import math
from typing import Any, cast

import cvxpy as cp
import mlx.core as mx
import numpy as np
import scipy.sparse
from cvxpy.reductions.canonical_maps import ParameterTransform, ParameterTransformKind

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
    """Reshape array using Fortran (column-major) order.

    MLX doesn't support order='F' directly, so we use transpose.

    Args:
        array: Input array to reshape
        shape: Target shape tuple

    Returns:
        Reshaped array in Fortran order
    """
    if len(array.shape) == 0:
        return mx.reshape(array, shape)
    x = mx.transpose(array, axes=tuple(reversed(range(len(array.shape)))))
    reshaped = mx.reshape(x, tuple(reversed(shape)))
    if len(shape) > 0:
        reshaped = mx.transpose(reshaped, axes=tuple(reversed(range(len(shape)))))
    return reshaped


def _matrix(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray | None,
            shape: tuple[int, int], dtype) -> mx.array:
    """A dense matrix with entries ``vals`` (default 1) at ``(rows, cols)``.

    MLX has no sparse linear algebra, and index operations are expressed as
    matmuls with such matrices, which preserves gradient flow.
    """
    M = np.zeros(shape, dtype=np.float64)
    np.add.at(M, (rows, cols), 1.0 if vals is None else vals)
    return mx.array(M, dtype=dtype)


def _gather(x: mx.array, indices: np.ndarray) -> mx.array:
    """``x[..., indices]`` as a matmul."""
    n_in = x.shape[-1]
    return x @ _matrix(indices, np.arange(len(indices)), None, (n_in, len(indices)), x.dtype)


def _apply_transform(transform: ParameterTransform, value: mx.array) -> mx.array:
    """Evaluate one of CVXPY's parameter transformations on flat values."""
    kind = transform.kind
    if kind == ParameterTransformKind.GATHER:
        return _gather(value, transform.indices)
    if kind == ParameterTransformKind.LOG:
        return mx.log(value)
    if kind == ParameterTransformKind.REAL:
        return mx.real(value)
    if kind == ParameterTransformKind.IMAG:
        return mx.imag(value)
    raise ValueError(f"Unknown parameter transform {kind}")


def _prepare_param_stack(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> mx.array:
    """Turn user parameter values into the stacked compiled-parameter vector.

    Evaluates CVXPY's parameter transformations (e.g. ``log`` for geometric
    programs, gathers of the free entries of structured parameters) and packs
    the compiled parameters in column order, followed by a constant 1.

    Returns:
        Array of shape (num_params + 1,) or (num_params + 1, batch_size).
    """
    effective_batch = batch if batch else (1,)
    values: dict[int, mx.array] = {}
    for i, (param, value) in enumerate(zip(ctx.parameters, params)):
        if ctx.batch_sizes[i] == 0:  # type: ignore[index]
            value = mx.broadcast_to(mx.expand_dims(value, axis=0), effective_batch + value.shape)
        flat = _reshape_fortran(value, effective_batch + (-1,))
        source = ctx.param_sources[i]
        if source is not None:
            # Sparse parameters enter the transforms as their stored entries.
            flat = _gather(flat, source)
        values[param.id] = flat
    for transform in ctx.param_transforms:
        values[transform.target_id] = _apply_transform(
            transform, values[transform.source_id])

    columns = [values[pid] for pid, _ in ctx.param_layout]
    dtype = columns[0].dtype if columns else params[0].dtype
    columns.append(mx.ones(effective_batch + (1,), dtype=dtype))
    p_stack = mx.transpose(mx.concatenate(columns, axis=-1))
    return mx.reshape(p_stack, (p_stack.shape[0],) + batch)


def _recover_output(out: pa.OutputRecovery, vec: mx.array, batch: tuple) -> mx.array:
    """Read one output from a (batch, n) solution array; see ``OutputRecovery``."""
    n = vec.shape[-1]
    if out.indices is not None:
        data = _gather(vec, out.indices)
    else:
        data = vec @ _matrix(out.cols, out.rows, out.vals, (n, out.size), vec.dtype)
    if out.exp:
        data = mx.exp(data)
    if out.scatter is not None:
        size = math.prod(out.shape)
        data = data @ _matrix(np.arange(out.size), out.scatter, None, (out.size, size),
                              vec.dtype)
    return _reshape_fortran(data, batch + out.shape)


def _recover_results(
    primal: mx.array,
    dual: mx.array,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[mx.array, ...]:
    """Recover the requested variables from the solver's primal/dual solutions.

    The solution arrays carry a leading batch dimension, also for unbatched
    inputs; it is removed at the end.
    """
    internal_batch = tuple(primal.shape[:-1])
    return tuple(
        mx.reshape(
            _recover_output(out, primal if out.source == "primal" else dual, internal_batch),
            batch + out.shape)
        for out in ctx.outputs
    )


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
        self._P_np = _scipy_csr_to_dense(self.ctx.P_mat)
        self._q_np: np.ndarray = _scipy_csr_to_dense(self.ctx.q_mat)  # type: ignore[assignment]
        self._A_np: np.ndarray = _scipy_csr_to_dense(self.ctx.A_mat)  # type: ignore[assignment]

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

        p_stack = _prepare_param_stack(params, self.ctx, batch)
        param_dtype = p_stack.dtype

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
