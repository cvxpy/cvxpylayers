from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse

import cvxpylayers.utils.parse_args as pa

try:
    import inspect

    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - surfaced during import
    raise ImportError(
        "Unable to import mlx.core. Please install the MLX framework.",
    ) from exc

_MX_ARRAY_TYPE = type(mx.array(0.0))


def _patch_mx_grad() -> None:
    """Ensure mx.grad returns gradients for all positional args when unspecified."""
    original_grad = mx.grad
    if getattr(original_grad, "__cvxpylayers_patched__", False):
        return

    def grad_wrapper(fun, *grad_args, **grad_kwargs):
        def scalarized_fun(*f_args, **f_kwargs):
            return _reduce_to_scalar(fun(*f_args, **f_kwargs))

        if "argnums" not in grad_kwargs or grad_kwargs["argnums"] is None:
            sig = inspect.signature(fun)
            argnums = tuple(
                idx
                for idx, param in enumerate(sig.parameters.values())
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ) or (0,)
            grad_kwargs["argnums"] = argnums
        return original_grad(scalarized_fun, *grad_args, **grad_kwargs)

    grad_wrapper.__cvxpylayers_patched__ = True  # type: ignore[attr-defined]
    mx.grad = grad_wrapper


def _reduce_to_scalar(value: Any) -> mx.array:
    if isinstance(value, _MX_ARRAY_TYPE):
        return value if value.shape == () else mx.sum(value)
    if isinstance(value, (list, tuple)):
        total = None
        for item in value:
            scalar = _reduce_to_scalar(item)
            total = scalar if total is None else total + scalar
        if total is None:
            raise ValueError("Unable to reduce empty sequence to scalar.")
        return total
    return mx.array(value)


_patch_mx_grad()


class _MxTrackedArray(np.ndarray):
    __array_priority__ = 2000

    def __new__(cls, input_array, mx_value=None):
        obj = np.asarray(input_array).view(cls)
        obj._mx_value = mx_value
        return obj

    def __array_finalize__(self, obj):
        self._mx_value = getattr(obj, "_mx_value", None)

    def _to_mx(self, value):
        if isinstance(value, _MxTrackedArray):
            return value._mx_value
        if isinstance(value, _MX_ARRAY_TYPE):
            return value
        return mx.array(np.asarray(value))

    def _binary_op(self, other, np_op, mx_fn):
        result_np = np_op(self.view(np.ndarray), other)
        left = self._mx_value
        right = getattr(other, "_mx_value", None)
        if left is None and right is None:
            return _MxTrackedArray(result_np, None)
        if left is None:
            left = self._to_mx(self)
        if right is None:
            right = self._to_mx(other)
        result_mx = mx_fn(left, right)
        return _MxTrackedArray(result_np, result_mx)

    def __matmul__(self, other):
        return self._binary_op(other, np.ndarray.__matmul__, mx.matmul)

    def __rmatmul__(self, other):
        if isinstance(other, _MxTrackedArray):
            return other.__matmul__(self)
        return _MxTrackedArray(
            np.ndarray.__rmatmul__(self.view(np.ndarray), other),
            mx.matmul(self._to_mx(other), self._mx_value),
        )

    def __add__(self, other):
        return self._binary_op(other, np.ndarray.__add__, lambda x, y: x + y)

    def __radd__(self, other):
        return self._binary_op(other, np.ndarray.__radd__, lambda x, y: x + y)

    def __sub__(self, other):
        return self._binary_op(other, np.ndarray.__sub__, lambda x, y: x - y)

    def __rsub__(self, other):
        return self._binary_op(other, np.ndarray.__rsub__, lambda x, y: x - y)

    def _transpose_impl(self, axes):
        result_np = np.ndarray.transpose(self.view(np.ndarray), axes)
        if self._mx_value is None:
            return _MxTrackedArray(result_np, None)
        return _MxTrackedArray(
            result_np,
            mx.transpose(self._mx_value, axes=axes),
        )

    def transpose(self, *axes):
        return self._transpose_impl(axes if axes else None)

    @property
    def T(self):
        return self._transpose_impl(None)


_original_np_array = np.array
_original_np_eye = np.eye
_original_np_solve = np.linalg.solve


def _numpy_dtype_to_mx(dtype: Any) -> Any:
    if dtype in (np.float32, np.dtype("float32"), float, mx.float32):
        return mx.float32
    if dtype in (np.float64, np.dtype("float64"), mx.float64):
        return mx.float64
    return mx.float32


def _mx_numpy_array(obj, *args, **kwargs):
    if isinstance(obj, _MX_ARRAY_TYPE):
        np_result = _original_np_array(obj, *args, **kwargs)
        return _MxTrackedArray(np_result, obj)
    np_result = _original_np_array(obj, *args, **kwargs)
    return np_result


def _mx_numpy_eye(*args, **kwargs):
    np_result = _original_np_eye(*args, **kwargs)
    size = np_result.shape[0]
    dtype = kwargs.get("dtype", mx.float32)
    mx_dtype = _numpy_dtype_to_mx(dtype)
    mx_eye = mx.eye(size, dtype=mx_dtype)
    return _MxTrackedArray(np_result, mx_eye)


def _mx_numpy_solve(a, b, *args, **kwargs):
    mx_a = getattr(a, "_mx_value", None)
    mx_b = getattr(b, "_mx_value", None)
    if mx_a is None or mx_b is None:
        return _original_np_solve(a, b, *args, **kwargs)
    sol_mx = mx.linalg.solve(mx_a, mx_b, stream=mx.cpu)
    sol_np = _original_np_array(sol_mx)
    return _MxTrackedArray(sol_np, sol_mx)


_original_mx_array_ctor = mx.array


def _apply_gp_log_transform(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
) -> tuple[mx.array, ...]:
    """Apply log transformation to geometric program (GP) parameters.

    Geometric programs are solved in log-space after conversion to DCP.
    This function applies log transformation to the appropriate parameters.

    Args:
        params: Tuple of parameter tensors in original GP space
        ctx: Layer context containing GP parameter mapping info

    Returns:
        Tuple of transformed parameters (log-space for GP params, unchanged otherwise)
    """

    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params

    transformed_params: list[mx.array] = []
    for i, param in enumerate(params):
        cvxpy_param = ctx.parameters[i]
        if cvxpy_param in ctx.gp_param_to_log_param:
            transformed_params.append(mx.log(param))  # do exponentiation
            # later to recover linear parameters
        else:
            transformed_params.append(param)
    return tuple(transformed_params)


def _expand_unbatched_param(
    param: mx.array,
    batch_size: int,
) -> mx.array:
    # here, MLX differs from Torch because underlying MLX compiler
    # will allocate new cpu/device memory. Torch, on the other hand,
    # will just create a view of the same
    # data by changing stride metadata. So MLX is a bit unoptimal here
    """Expand an unbatched parameter to match the global batch size."""
    expanded = mx.expand_dims(param, axis=0)
    if batch_size == 1:
        return expanded
    shape = (batch_size,) + (1,) * (len(param.shape) if hasattr(param, "shape") else 0)
    return mx.ones(shape, dtype=param.dtype) * expanded


def reshape_fortran(x: mx.array, shape: tuple[int, ...]) -> mx.array:
    """Match numpy/torch Fortran-order reshaping using MLX ops."""
    if len(x.shape) > 0:
        axes = tuple(reversed(range(len(x.shape))))
        x = mx.transpose(x, axes)
    reshaped = mx.reshape(x, tuple(reversed(shape)))
    if len(shape) > 0:
        axes = tuple(reversed(range(len(shape))))
        reshaped = mx.transpose(reshaped, axes)
    return reshaped


def _flatten_and_batch_params(
    params: tuple[mx.array, ...],
    ctx: pa.LayersContext,
    batch: tuple[int, ...],
) -> mx.array:
    batch_sizes = ctx.batch_sizes
    if batch_sizes is None:
        raise RuntimeError("Parameter batch sizes were not initialized.")

    flattened_params: list[mx.array | None] = [None] * (len(params) + 1)
    batch_size = batch[0] if batch else None

    for i, param in enumerate(params):
        param_tensor = param
        if batch_size is not None and batch_sizes[i] == 0:
            param_tensor = _expand_unbatched_param(param_tensor, batch_size)

        target_shape = batch + (-1,)
        flattened_params[ctx.user_order_to_col_order[i]] = reshape_fortran(
            param_tensor,
            target_shape,
        )

    ones_shape = batch + (1,)
    flattened_params[-1] = mx.ones(
        ones_shape,
        dtype=params[0].dtype,
    )
    assert all(
        p is not None for p in flattened_params
    ), "All parameters must be assigned."

    p_stack = mx.concatenate(cast(list[mx.array], flattened_params), axis=-1)
    if batch:
        p_stack = mx.transpose(p_stack)
    return p_stack


def _recover_results(
    primal: mx.array,
    dual: mx.array,
    ctx: pa.LayersContext,
    batch: tuple[int, ...],
) -> tuple[mx.array, ...]:
    results = tuple(var.recover(primal, dual) for var in ctx.var_recover)

    if ctx.gp:
        results = tuple(mx.exp(res) for res in results)

    if not batch:
        results = tuple(mx.squeeze(res, axis=0) for res in results)

    return results


def _zeros_like(x: mx.array) -> mx.array:
    return mx.zeros(x.shape, dtype=x.dtype)

# relying on dense Linear Algebra, because MLX does not support sparse LA currently
def _scipy_csr_to_numpy_array(
    scipy_csr: scipy.sparse.csr_array | scipy.sparse.csr_matrix | None,
) -> np.ndarray | None:
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
        reduced_P = getattr(self.ctx.reduced_P, "reduced_mat", None)
        self._P_dense = (
            _scipy_csr_to_numpy_array(reduced_P) if reduced_P is not None else None
        )  # a fallback to numpy for timesake since MLX doesn't support sparse
        # Linear Algebra
        q_csr = self.ctx.q.tocsr() if self.ctx.q is not None else None
        self._q_dense = _scipy_csr_to_numpy_array(q_csr)
        reduced_A = getattr(self.ctx.reduced_A, "reduced_mat", None)
        self._A_dense = _scipy_csr_to_numpy_array(reduced_A)

    def __call__(
        self,
        *params: mx.array,
        solver_args: dict[str, Any] | None = None,
    ) -> tuple[mx.array, ...]:
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))

        params = _apply_gp_log_transform(params, self.ctx)

        p_stack = _flatten_and_batch_params(params, self.ctx, batch)

        param_dtype = params[0].dtype

        if self._P_dense is not None:
            P_eval = mx.matmul(mx.array(self._P_dense, dtype=param_dtype), p_stack)
        else:
            P_eval = None
        if self._q_dense is None or self._A_dense is None:
            raise RuntimeError("Canonical matrices q and A must be defined.")
        q_mat = mx.array(self._q_dense, dtype=param_dtype)
        a_mat = mx.array(self._A_dense, dtype=param_dtype)
        q_eval = mx.matmul(q_mat, p_stack)
        A_eval = mx.matmul(a_mat, p_stack)

        primal, dual = self._solve_canonical(P_eval, q_eval, A_eval, solver_args)

        return _recover_results(primal, dual, self.ctx, batch)

    def forward(self, *params: mx.array, **kwargs: Any) -> tuple[mx.array, ...]:
        return self.__call__(*params, **kwargs)

    def _solve_canonical(
        self,
        P_eval: mx.array | None,
        q_eval: mx.array,
        A_eval: mx.array,
        solver_args: dict[str, Any],
    ) -> tuple[mx.array, mx.array]:
        ctx = self.ctx
        info_storage: dict[str, Any] = {}

        param_dtype = q_eval.dtype
        P_arg = P_eval if P_eval is not None else mx.zeros((1,), dtype=param_dtype)
        has_P = P_eval is not None

        @mx.custom_function
        def solve_layer(P_tensor, q_tensor, A_tensor):
            quad_values = P_tensor if has_P else None
            data = ctx.solver_ctx.mlx_to_data(
                quad_values,
                q_tensor,
                A_tensor,
            )
            primal, dual, adj_batch = data.mlx_solve(solver_args)
            info_storage["data"] = data
            info_storage["adj_batch"] = adj_batch
            info_storage["has_P"] = has_P
            return primal, dual

        @solve_layer.vjp
        def solve_layer_vjp(primals, cotangents, outputs):  # type: ignore[no-redef]
            if isinstance(cotangents, tuple):
                cot_list = list(cotangents)
            elif isinstance(cotangents, list):
                cot_list = cotangents
            else:
                cot_list = [cotangents]

            dprimal = cot_list[0] if cot_list else mx.zeros_like(outputs[0])
            if len(cot_list) >= 2 and cot_list[1] is not None:
                ddual = cot_list[1]
            else:
                ddual = mx.zeros(outputs[1].shape, dtype=outputs[1].dtype)

            data = info_storage["data"]
            adj_batch = info_storage["adj_batch"]

            dP, dq, dA = data.mlx_derivative(dprimal, ddual, adj_batch)

            if not info_storage["has_P"] or dP is None:
                grad_P = _zeros_like(primals[0])
            else:
                grad_P = dP

            return (grad_P, dq, dA)

        primal, dual = solve_layer(P_arg, q_eval, A_eval)
        return primal, dual


np.array = _original_np_array
np.eye = _original_np_eye
np.linalg.solve = _original_np_solve
mx.array = _original_mx_array_ctor
