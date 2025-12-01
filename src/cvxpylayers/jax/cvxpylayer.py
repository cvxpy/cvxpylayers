from typing import Any, Dict, List, Optional, Tuple, cast

import cvxpy as cp
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import scipy.sparse

import cvxpylayers.utils.parse_args as pa


def _apply_gp_log_transform(
    params: Tuple[jnp.ndarray, ...],
    ctx: pa.LayersContext,
) -> Tuple[jnp.ndarray, ...]:
    """Apply log transformation to geometric program (GP) parameters.

    Geometric programs are solved in log-space after conversion to DCP.
    This function applies log transformation to the appropriate parameters.

    Args:
        params: Tuple of parameter arrays in original GP space
        ctx: Layer context containing GP parameter mapping info

    Returns:
        Tuple of transformed parameters (log-space for GP params, unchanged otherwise)
    """
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params

    params_transformed = []
    for i, param in enumerate(params):
        cvxpy_param = ctx.parameters[i]
        if cvxpy_param in ctx.gp_param_to_log_param:
            # This parameter needs log transformation for GP
            params_transformed.append(jnp.log(param))
        else:
            params_transformed.append(param)
    return tuple(params_transformed)


def _flatten_and_batch_params(
    params: Tuple[jnp.ndarray, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> jnp.ndarray:
    """Flatten and batch parameters into a single stacked array.

    Converts a tuple of parameter arrays (potentially with mixed batched/unbatched)
    into a single concatenated array suitable for matrix multiplication with the
    parametrized problem matrices.

    Args:
        params: Tuple of parameter arrays
        ctx: Layer context with batch info and ordering
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Concatenated parameter array with shape (num_params, batch_size) or (num_params,)
    """
    flattened_params: List[Optional[jnp.ndarray]] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0 and batch:
            # Unbatched parameter - expand to match batch size
            param_expanded = jnp.expand_dims(param, 0)
            param_expanded = jnp.broadcast_to(param_expanded, batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = jnp.reshape(
                param_expanded,
                batch + (-1,),
                order="F",
            )
        else:
            # Already batched or no batch dimension needed
            flattened_params[ctx.user_order_to_col_order[i]] = jnp.reshape(
                param,
                batch + (-1,),
                order="F",
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = jnp.ones(batch + (1,), dtype=params[0].dtype)
    assert all(p is not None for p in flattened_params), "All parameters must be assigned"

    p_stack = jnp.concatenate(cast(List[jnp.ndarray], flattened_params), -1)
    # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
    if batch:
        p_stack = p_stack.T
    return p_stack


def _recover_results(
    primal: jnp.ndarray,
    dual: jnp.ndarray,
    ctx: pa.LayersContext,
    batch: tuple,
) -> Tuple[jnp.ndarray, ...]:
    """Recover variable values from primal/dual solutions.

    Extracts the requested variables from the solver's primal and dual
    solutions, applies inverse GP transformation if needed, and removes
    batch dimension for unbatched inputs.

    Args:
        primal: Primal solution from solver
        dual: Dual solution from solver
        ctx: Layer context with variable recovery info
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Tuple of recovered variable values
    """
    # Extract each variable using its slice from the solution vectors
    results = tuple(var.recover(primal, dual) for var in ctx.var_recover)

    # Apply exp transformation to recover from log-space for GP
    if ctx.gp:
        results = tuple(jnp.exp(r) for r in results)

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = tuple(jnp.squeeze(r, 0) for r in results)

    return results


class CvxpyLayer:
    def __init__(
        self,
        problem: cp.Problem,
        parameters: List[cp.Parameter],
        variables: List[cp.Variable],
        solver: Optional[str] = None,
        gp: bool = False,
        verbose: bool = False,
        canon_backend: Optional[str] = None,
        solver_args: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        self, *params: jnp.ndarray, solver_args: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, ...]:
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
        return _recover_results(primal, dual, self.ctx, batch)


def scipy_csr_to_jax_bcsr(
    scipy_csr: Optional[scipy.sparse.csr_array],
) -> Optional[jax.experimental.sparse.BCSR]:
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

