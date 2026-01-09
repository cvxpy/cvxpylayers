from typing import Any, cast

import cvxpy as cp
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import scipy.sparse

import cvxpylayers.utils.parse_args as pa
from cvxpylayers.utils.parse_args import (
    apply_gp_log_transform,
    flatten_and_batch_params,
    recover_results,
)


class CvxpyLayer:
    """A differentiable convex optimization layer for JAX.

    This layer wraps a parametrized CVXPY problem, solving it in the forward pass
    and computing gradients via implicit differentiation. Compatible with
    ``jax.grad``, ``jax.jit``, and ``jax.vmap``.

    Example:
        >>> import cvxpy as cp
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from cvxpylayers.jax import CvxpyLayer
        >>>
        >>> # Define a simple QP
        >>> x = cp.Variable(2)
        >>> A = cp.Parameter((3, 2))
        >>> b = cp.Parameter(3)
        >>> problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
        >>>
        >>> # Create the layer
        >>> layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        >>>
        >>> # Solve and compute gradients
        >>> A_jax = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
        >>> b_jax = jax.random.normal(jax.random.PRNGKey(1), (3,))
        >>> (solution,) = layer(A_jax, b_jax)
        >>>
        >>> # Gradient computation
        >>> def loss_fn(A, b):
        ...     (x,) = layer(A, b)
        ...     return jnp.sum(x)
        >>> grads = jax.grad(loss_fn, argnums=[0, 1])(A_jax, b_jax)
    """

    def __init__(
        self,
        problem: cp.Problem,
        parameters: list[cp.Parameter],
        variables: list[cp.Variable],
        solver: str | None = None,
        gp: bool = False,
        verbose: bool = False,
        canon_backend: str | None = None,
        solver_args: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the differentiable optimization layer.

        Args:
            problem: A CVXPY Problem. Must be DPP-compliant (``problem.is_dpp()``
                must return True).
            parameters: List of CVXPY Parameters that will be filled with values
                at runtime. Order must match the order of arrays passed to __call__().
            variables: List of CVXPY Variables whose optimal values will be returned
                by __call__(). Order determines the order of returned arrays.
            solver: CVXPY solver to use (e.g., ``cp.CLARABEL``, ``cp.SCS``).
                If None, uses the default diffcp solver.
            gp: If True, problem is a geometric program. Parameters will be
                log-transformed before solving.
            verbose: If True, print solver output.
            canon_backend: Backend for canonicalization. Options are 'diffcp',
                'cuclarabel', or None (auto-select).
            solver_args: Default keyword arguments passed to the solver.
                Can be overridden per-call in __call__().

        Raises:
            AssertionError: If problem is not DPP-compliant.
            ValueError: If parameters or variables are not part of the problem.
        """
        if solver_args is None:
            solver_args = {}
        self.ctx = pa.parse_args(
            problem,
            variables,
            parameters,
            solver,
            gp=gp,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_args=solver_args,
        )
        if self.ctx.reduced_P.reduced_mat is not None:  # type: ignore[attr-defined]
            self.P = scipy_csr_to_jax_bcsr(self.ctx.reduced_P.reduced_mat)  # type: ignore[attr-defined]
        else:
            self.P = None
        self.q: jax.experimental.sparse.BCSR = scipy_csr_to_jax_bcsr(self.ctx.q.tocsr())  # type: ignore[assignment]
        self.A: jax.experimental.sparse.BCSR = scipy_csr_to_jax_bcsr(self.ctx.reduced_A.reduced_mat)  # type: ignore[attr-defined,assignment]

    def __call__(
        self, *params: jnp.ndarray, solver_args: dict[str, Any] | None = None
    ) -> tuple[jnp.ndarray, ...]:
        """Solve the optimization problem and return optimal variable values.

        Args:
            *params: Array values for each CVXPY Parameter, in the same order
                as the ``parameters`` argument to __init__(). Each array shape must
                match the corresponding Parameter shape, optionally with a batch
                dimension prepended. Batched and unbatched parameters can be mixed;
                unbatched parameters are broadcast.
            solver_args: Keyword arguments passed to the solver, overriding any
                defaults set in __init__().

        Returns:
            Tuple of arrays containing optimal values for each CVXPY Variable
            specified in the ``variables`` argument to __init__(). If inputs are
            batched, outputs will have matching batch dimensions.

        Raises:
            RuntimeError: If the solver fails to find a solution.

        Example:
            >>> # Single problem
            >>> (x_opt,) = layer(A_array, b_array)
            >>>
            >>> # Batched: solve 10 problems in parallel
            >>> A_batch = jax.random.normal(key, (10, 3, 2))
            >>> b_batch = jax.random.normal(key, (10, 3))
            >>> (x_batch,) = layer(A_batch, b_batch)  # x_batch.shape = (10, 2)
        """
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))

        # Apply log transformation to GP parameters
        params = apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = flatten_and_batch_params(params, self.ctx, batch)

        # Evaluate parametrized matrices
        P_eval = self.P @ p_stack if self.P is not None else None
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack

        # Store data and adjoint in closure for backward pass
        # This avoids JAX trying to trace through DIFFCP's Python-based solver
        data_container = {}

        @jax.custom_vjp
        def solve_problem(P_eval, q_eval, A_eval):
            # Forward pass: solve the optimization problem
            data = self.ctx.solver_ctx.jax_to_data(P_eval, q_eval, A_eval)  # type: ignore[attr-defined]
            primal, dual, adj_batch = data.jax_solve(solver_args)  # type: ignore[attr-defined]
            # Store for backward pass (outside JAX tracing)
            data_container["data"] = data
            data_container["adj_batch"] = adj_batch
            return primal, dual

        def solve_problem_fwd(P_eval, q_eval, A_eval):
            # Call forward to execute and populate container
            primal, dual = solve_problem(P_eval, q_eval, A_eval)
            # Return empty residuals (data is in closure)
            return (primal, dual), ()

        def solve_problem_bwd(res, g):
            # Backward pass using adjoint method
            dprimal, ddual = g
            data = data_container["data"]
            adj_batch = data_container["adj_batch"]
            dP, dq, dA = data.jax_derivative(dprimal, ddual, adj_batch)
            return dP, dq, dA

        solve_problem.defvjp(solve_problem_fwd, solve_problem_bwd)
        primal, dual = solve_problem(P_eval, q_eval, A_eval)

        # Recover results and apply GP inverse transform if needed
        return recover_results(primal, dual, self.ctx, batch)


def scipy_csr_to_jax_bcsr(
    scipy_csr: scipy.sparse.csr_array | None,
) -> jax.experimental.sparse.BCSR | None:
    if scipy_csr is None:
        return None
    # Use cast to help type checker understand scipy_csr is not None
    scipy_csr = cast(scipy.sparse.csr_array, scipy_csr)
    # Get the CSR format components
    values = scipy_csr.data
    col_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr
    num_rows, num_cols = scipy_csr.shape  # type: ignore[misc]

    # Create the JAX BCSR tensor
    jax_bcsr = jax.experimental.sparse.BCSR(
        (jnp.array(values), jnp.array(col_indices), jnp.array(row_ptr)),
        shape=(num_rows, num_cols),
    )

    return jax_bcsr
