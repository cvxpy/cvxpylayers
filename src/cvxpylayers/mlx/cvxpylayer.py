from typing import Any, cast

import cvxpy as cp
import mlx.core as mx
import numpy as np
import scipy.sparse

import cvxpylayers.utils.parse_args as pa


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


def _reshape_fortran(array: mx.array, shape: tuple) -> mx.array:
    """Reshape array using Fortran (column-major) order."""
    if len(array.shape) == 0:
        return mx.reshape(array, shape)
    x = mx.transpose(array, axes=tuple(reversed(range(len(array.shape)))))
    reshaped = mx.reshape(x, tuple(reversed(shape)))
    if len(shape) > 0:
        reshaped = mx.transpose(reshaped, axes=tuple(reversed(range(len(shape)))))
    return reshaped


def _apply_gp_log_transform(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
) -> tuple[mx.array, ...]:
    """Apply log transformation to geometric program (GP) parameters."""
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params
    return tuple(
        mx.log(p) if ctx.parameters[i] in ctx.gp_param_to_log_param else p
        for i, p in enumerate(params)
    )


def _flatten_and_batch_params(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> mx.array:
    """Flatten and batch parameters into a single stacked tensor."""
    flattened_params: list[mx.array | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        if ctx.batch_sizes[i] == 0 and batch:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            param_expanded = mx.broadcast_to(mx.expand_dims(param, axis=0), batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded, batch + (-1,)
            )
        else:
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param, batch + (-1,)
            )

    # Add constant 1.0 column for offset terms
    flattened_params[-1] = mx.ones(batch + (1,), dtype=params[0].dtype)

    p_stack = mx.concatenate([p for p in flattened_params if p is not None], axis=-1)
    if batch:
        p_stack = mx.transpose(p_stack)
    return p_stack


def _svec_to_symmetric(
    svec: mx.array,
    n: int,
    batch: tuple,
    rows: np.ndarray,
    cols: np.ndarray,
    scale: np.ndarray | None = None,
) -> mx.array:
    """Convert vectorized form to full symmetric matrix.

    MLX doesn't support advanced indexing like torch/jax, so we use numpy
    for the indexing operations and convert back to MLX.
    """
    if scale is not None:
        scale_mx = mx.array(scale, dtype=svec.dtype)
        data = svec * scale_mx
    else:
        data = svec

    out_shape = batch + (n, n)
    if batch:
        batch_size = int(np.prod(batch))
        data_flat = mx.reshape(data, (batch_size, -1))
        # Build result by iterating (MLX lacks advanced indexing)
        results = []
        for b in range(batch_size):
            data_b = data_flat[b]
            # Use numpy for indexing, then convert
            result_np = np.zeros((n, n), dtype=np.float64)
            data_np = np.array(data_b)
            result_np[rows, cols] = data_np
            result_np[cols, rows] = data_np
            results.append(mx.array(result_np, dtype=svec.dtype))
        result = mx.stack(results, axis=0)
        return mx.reshape(result, out_shape)
    else:
        # Unbatched: simple approach via numpy
        data_np = np.array(data)
        result_np = np.zeros((n, n), dtype=np.float64)
        result_np[rows, cols] = data_np
        result_np[cols, rows] = data_np
        return mx.array(result_np, dtype=svec.dtype)


def _unpack_primal_svec(svec: mx.array, n: int, batch: tuple) -> mx.array:
    """Unpack symmetric primal variable from vectorized form (upper tri row-major)."""
    rows, cols = np.triu_indices(n)
    return _svec_to_symmetric(svec, n, batch, rows, cols)


def _unpack_svec(svec: mx.array, n: int, batch: tuple) -> mx.array:
    """Unpack scaled vectorized (svec) form to full symmetric matrix."""
    rows_rm, cols_rm = np.tril_indices(n)
    sort_idx = np.lexsort((rows_rm, cols_rm))
    rows = rows_rm[sort_idx]
    cols = cols_rm[sort_idx]
    scale = np.where(rows == cols, 1.0, 1.0 / np.sqrt(2.0))
    return _svec_to_symmetric(svec, n, batch, rows, cols, scale)


def _recover_results(
    primal: mx.array,
    dual: mx.array,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[mx.array, ...]:
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
            mx.exp(r) if var.primal is not None else r for r, var in zip(results, ctx.var_recover)
        ]

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = [mx.squeeze(r, axis=0) for r in results]

    return tuple(results)


class CvxpyLayer:
    """A differentiable convex optimization layer for MLX.

    This layer wraps a parametrized CVXPY problem, solving it in the forward pass
    and computing gradients via implicit differentiation. Optimized for Apple
    Silicon (M1/M2/M3) with unified memory architecture.

    Example:
        >>> import cvxpy as cp
        >>> import mlx.core as mx
        >>> from cvxpylayers.mlx import CvxpyLayer
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
        >>> A_mx = mx.random.normal((3, 2))
        >>> b_mx = mx.random.normal((3,))
        >>> (solution,) = layer(A_mx, b_mx)
        >>>
        >>> # Gradient computation
        >>> def loss_fn(A, b):
        ...     (x,) = layer(A, b)
        ...     return mx.sum(x)
        >>> grad_fn = mx.grad(loss_fn, argnums=[0, 1])
        >>> grads = grad_fn(A_mx, b_mx)
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
            >>> A_batch = mx.random.normal((10, 3, 2))
            >>> b_batch = mx.random.normal((10, 3))
            >>> (x_batch,) = layer(A_batch, b_batch)  # x_batch.shape = (10, 2)
        """
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
        P_eval = (
            mx.array(self._P_np, dtype=param_dtype) @ p_stack if self._P_np is not None else None
        )
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
            ddual = (
                cot_list[1]
                if len(cot_list) >= 2 and cot_list[1] is not None
                else mx.zeros(outputs[1].shape, dtype=outputs[1].dtype)
            )

            data = data_container["data"]
            adj_batch = data_container["adj_batch"]
            dP, dq, dA = data.mlx_derivative(dprimal, ddual, adj_batch)

            # Return zero gradient for P if problem has no quadratic term
            if not data_container["has_P"] or dP is None:
                grad_P = mx.zeros(primals[0].shape, dtype=primals[0].dtype)
            else:
                grad_P = dP

            return (grad_P, dq, dA)

        primal, dual = solve_layer(P_arg, q_eval, A_eval)  # type: ignore[misc]
        return primal, dual
