from typing import Any, cast

import cvxpy as cp
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import numpy as np
import scipy.sparse

import cvxpylayers.utils.parse_args as pa


def _reshape_fortran(array: jnp.ndarray, shape: tuple) -> jnp.ndarray:
    """Reshape array using Fortran (column-major) order."""
    return jnp.reshape(array, shape, order="F")


def _apply_gp_log_transform(
    params: tuple[jnp.ndarray, ...],
    ctx: pa.LayersContext,
) -> tuple[jnp.ndarray, ...]:
    """Apply log transformation to geometric program (GP) parameters."""
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params
    return tuple(
        jnp.log(p) if ctx.parameters[i] in ctx.gp_param_to_log_param else p
        for i, p in enumerate(params)
    )


def _flatten_and_batch_params(
    params: tuple[jnp.ndarray, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> jnp.ndarray:
    """Flatten and batch parameters into a single stacked tensor."""
    flattened_params: list[jnp.ndarray | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        if ctx.batch_sizes[i] == 0 and batch:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            param_expanded = jnp.broadcast_to(jnp.expand_dims(param, 0), batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded, batch + (-1,)
            )
        else:
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param, batch + (-1,)
            )

    # Add constant 1.0 column for offset terms
    flattened_params[-1] = jnp.ones(batch + (1,), dtype=params[0].dtype)

    p_stack = jnp.concatenate([p for p in flattened_params if p is not None], -1)
    if batch:
        p_stack = p_stack.T
    return p_stack


def _svec_to_symmetric(
    svec: jnp.ndarray,
    n: int,
    batch: tuple,
    rows: np.ndarray,
    cols: np.ndarray,
    scale: np.ndarray | None = None,
) -> jnp.ndarray:
    """Convert vectorized form to full symmetric matrix."""
    rows_arr = jnp.array(rows)
    cols_arr = jnp.array(cols)
    data = svec * jnp.array(scale) if scale is not None else svec
    out_shape = batch + (n, n)
    result = jnp.zeros(out_shape, dtype=svec.dtype)
    result = result.at[..., rows_arr, cols_arr].set(data)
    result = result.at[..., cols_arr, rows_arr].set(data)
    return result


def _unpack_primal_svec(svec: jnp.ndarray, n: int, batch: tuple) -> jnp.ndarray:
    """Unpack symmetric primal variable from vectorized form (upper tri row-major)."""
    rows, cols = np.triu_indices(n)
    return _svec_to_symmetric(svec, n, batch, rows, cols)


def _unpack_svec(svec: jnp.ndarray, n: int, batch: tuple) -> jnp.ndarray:
    """Unpack scaled vectorized (svec) form to full symmetric matrix."""
    rows_rm, cols_rm = np.tril_indices(n)
    sort_idx = np.lexsort((rows_rm, cols_rm))
    rows = rows_rm[sort_idx]
    cols = cols_rm[sort_idx]
    scale = np.where(rows == cols, 1.0, 1.0 / np.sqrt(2.0))
    return _svec_to_symmetric(svec, n, batch, rows, cols, scale)


def _recover_results(
    primal: jnp.ndarray,
    dual: jnp.ndarray,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[jnp.ndarray, ...]:
    """Recover variable values from primal/dual solutions."""
    results = []
    for var in ctx.var_recover:
        batch_shape = tuple(primal.shape[:-1])
        if var.primal is not None:
            data = primal[..., var.primal]
            if var.is_symmetric:
                results.append(_unpack_primal_svec(data, var.shape[0], batch_shape))
            else:
                results.append(_reshape_fortran(data, batch_shape + var.shape))
        elif var.dual is not None:
            data = dual[..., var.dual]
            if var.is_psd_dual:
                results.append(_unpack_svec(data, var.shape[0], batch_shape))
            else:
                results.append(_reshape_fortran(data, batch_shape + var.shape))
        else:
            raise RuntimeError("Invalid VariableRecovery")

    # Apply exp transformation for GP primal variables
    if ctx.gp:
        results = [
            jnp.exp(r) if var.primal is not None else r for r, var in zip(results, ctx.var_recover)
        ]

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = [jnp.squeeze(r, 0) for r in results]

    return tuple(results)


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
        params = _apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = _flatten_and_batch_params(params, self.ctx, batch)

        # Evaluate parametrized matrices
        P_eval = self.P @ p_stack if self.P is not None else None
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack

        # Store data and adjoint in closure for backward pass
        # This avoids JAX trying to trace through DIFFCP's Python-based solver
        data_container: dict[str, Any] = {}

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
        return _recover_results(primal, dual, self.ctx, batch)


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
