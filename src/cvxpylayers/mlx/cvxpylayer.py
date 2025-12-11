from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse
import mlx.core as mx

import cvxpylayers.utils.parse_args as pa


def _apply_gp_log_transform(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
) -> tuple[mx.array, ...]:
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
            params_transformed.append(mx.log(param))
        else:
            params_transformed.append(param)
    return tuple(params_transformed)


def _flatten_and_batch_params(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
    batch: tuple[int, ...],
) -> mx.array:
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
    flattened_params: list[mx.array | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0 and batch:
            # Unbatched parameter - expand to match batch size
            param_expanded = mx.expand_dims(param, axis=0)
            param_expanded = mx.broadcast_to(param_expanded, batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded,
                batch + (-1,),
            )
        else:
            # Already batched or no batch dimension needed
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param,
                batch + (-1,),
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = mx.ones(batch + (1,), dtype=params[0].dtype)
    assert all(p is not None for p in flattened_params), "All parameters must be assigned"

    p_stack = mx.concatenate(cast(list[mx.array], flattened_params), axis=-1)
    # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
    if batch:
        p_stack = mx.transpose(p_stack)
    return p_stack


def _recover_results(
    primal: mx.array,
    dual: mx.array,
    ctx: pa.LayersContext,
    batch: tuple[int, ...],
) -> tuple[mx.array, ...]:
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
    # Extract each variable using its slice and reshape
    results = tuple(
        var.recover(primal, dual, _reshape_fortran)
        for var in ctx.var_recover
    )

    # Apply exp transformation to recover from log-space for GP
    if ctx.gp:
        results = tuple(mx.exp(r) for r in results)

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = tuple(mx.squeeze(r, axis=0) for r in results)

    return results


def _reshape_fortran(x: mx.array, shape: tuple[int, ...]) -> mx.array:
    """Reshape array using Fortran (column-major) order.

    MLX doesn't support order='F' in reshape, so we emulate it by
    transposing before and after the reshape.
    """
    if len(x.shape) > 0:
        x = mx.transpose(x, axes=tuple(reversed(range(len(x.shape)))))
    reshaped = mx.reshape(x, tuple(reversed(shape)))
    if len(shape) > 0:
        reshaped = mx.transpose(reshaped, axes=tuple(reversed(range(len(shape)))))
    return reshaped


def _scipy_csr_to_dense(
    scipy_csr: scipy.sparse.csr_array | scipy.sparse.csr_matrix | None,
) -> np.ndarray | None:
    """Convert scipy sparse CSR matrix to dense numpy array.

    MLX does not currently support sparse linear algebra, so we convert
    to dense matrices for computation.
    """
    if scipy_csr is None:
        return None
    scipy_csr = cast(scipy.sparse.csr_matrix, scipy_csr)
    return np.asarray(scipy_csr.toarray())


class CvxpyLayer:
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
        # MLX doesn't support sparse LA, so we store dense numpy arrays
        # and convert to MLX arrays during forward pass
        if self.ctx.reduced_P.reduced_mat is not None:  # type: ignore[attr-defined]
            self._P_np = _scipy_csr_to_dense(self.ctx.reduced_P.reduced_mat)  # type: ignore[attr-defined]
        else:
            self._P_np = None
        self._q_np: np.ndarray = _scipy_csr_to_dense(self.ctx.q.tocsr())  # type: ignore[assignment]
        self._A_np: np.ndarray = _scipy_csr_to_dense(self.ctx.reduced_A.reduced_mat)  # type: ignore[attr-defined, assignment]

    def __call__(
        self,
        *params: mx.array,
        solver_args: dict[str, Any] | None = None,
    ) -> tuple[mx.array, ...]:
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))

        # Apply log transformation to GP parameters
        params = _apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = _flatten_and_batch_params(params, self.ctx, batch)

        # Get dtype from input parameters to ensure type matching
        param_dtype = params[0].dtype

        # Evaluate parametrized matrices (convert dense numpy to MLX)
        P_eval = mx.array(self._P_np, dtype=param_dtype) @ p_stack if self._P_np is not None else None
        q_eval = mx.array(self._q_np, dtype=param_dtype) @ p_stack
        A_eval = mx.array(self._A_np, dtype=param_dtype) @ p_stack

        # Solve optimization problem with custom VJP for gradients
        primal, dual = self._solve_with_vjp(P_eval, q_eval, A_eval, solver_args)

        # Recover results and apply GP inverse transform if needed
        return _recover_results(primal, dual, self.ctx, batch)

    def forward(
        self,
        *params: mx.array,
        solver_args: dict[str, Any] | None = None,
    ) -> tuple[mx.array, ...]:
        """Forward pass (alias for __call__)."""
        return self.__call__(*params, solver_args=solver_args)

    def _solve_with_vjp(
        self,
        P_eval: mx.array | None,
        q_eval: mx.array,
        A_eval: mx.array,
        solver_args: dict[str, Any],
    ) -> tuple[mx.array, mx.array]:
        """Solve the canonical problem with custom VJP for backpropagation."""
        ctx = self.ctx

        # Store data and adjoint in closure for backward pass
        data_container: dict[str, Any] = {}

        # Handle None P by using a dummy tensor (required for custom_function signature)
        param_dtype = q_eval.dtype
        P_arg = P_eval if P_eval is not None else mx.zeros((1,), dtype=param_dtype)
        has_P = P_eval is not None

        @mx.custom_function
        def solve_layer(P_tensor: mx.array, q_tensor: mx.array, A_tensor: mx.array):
            # Forward pass: solve the optimization problem
            quad_values = P_tensor if has_P else None
            data = ctx.solver_ctx.mlx_to_data(quad_values, q_tensor, A_tensor)  # type: ignore[attr-defined]
            primal, dual, adj_batch = data.mlx_solve(solver_args)  # type: ignore[attr-defined]
            # Store for backward pass (outside MLX tracing)
            data_container["data"] = data
            data_container["adj_batch"] = adj_batch
            data_container["has_P"] = has_P
            return primal, dual

        @solve_layer.vjp
        def solve_layer_vjp(primals, cotangents, outputs):  # noqa: F811
            # Backward pass using adjoint method
            if isinstance(cotangents, (tuple, list)):
                cot_list = list(cotangents)
            else:
                cot_list = [cotangents]

            dprimal = cot_list[0] if cot_list else mx.zeros_like(outputs[0])
            ddual = cot_list[1] if len(cot_list) >= 2 and cot_list[1] is not None else mx.zeros(outputs[1].shape, dtype=outputs[1].dtype)

            data = data_container["data"]
            adj_batch = data_container["adj_batch"]
            dP, dq, dA = data.mlx_derivative(dprimal, ddual, adj_batch)

            # Return zero gradient for P if problem has no quadratic term
            if not data_container["has_P"] or dP is None:
                grad_P = mx.zeros(primals[0].shape, dtype=primals[0].dtype)
            else:
                grad_P = dP

            return (grad_P, dq, dA)

        primal, dual = solve_layer(P_arg, q_eval, A_eval)
        return primal, dual
