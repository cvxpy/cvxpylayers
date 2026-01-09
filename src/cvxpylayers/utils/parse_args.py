from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import cvxpy as cp
import cvxpy.constraints
import numpy as np
import scipy.sparse
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg

import cvxpylayers.interfaces

if TYPE_CHECKING:
    import torch

T = TypeVar("T")


class SolverData(Protocol):
    """Protocol for data objects returned by solver context."""

    def torch_solve(
        self, solver_args: dict[str, Any] | None = None
    ) -> tuple["torch.Tensor", "torch.Tensor", Any]:
        """Solve the problem using torch backend.

        Returns:
            tuple of (primal, dual, adj_batch) where primal and dual are torch tensors
            and adj_batch is a solver-specific adjoint/backward object.
        """
        ...

    def torch_derivative(
        self, primal: "torch.Tensor", dual: "torch.Tensor", adj_batch: Any
    ) -> tuple["torch.Tensor | None", "torch.Tensor", "torch.Tensor"]:
        """Compute derivatives using torch backend.

        Returns:
            tuple of (dP, dq, dA) gradients as torch tensors (dP can be None).
        """
        ...


class SolverContext(Protocol):
    """Protocol for solver context objects."""

    def torch_to_data(
        self,
        quad_obj_values: "torch.Tensor | None",
        lin_obj_values: "torch.Tensor",
        con_values: "torch.Tensor",
    ) -> SolverData:
        """Convert torch tensors to solver data format."""
        ...


# Default reshape function for numpy/jax arrays
def _default_reshape(array, shape):
    return array.reshape(shape, order="F")


def reshape_fortran(array: T, shape: tuple) -> T:
    """Reshape array using Fortran (column-major) order.

    Works with torch, jax, mlx, and numpy arrays. JAX uses native order='F',
    while torch and mlx use permutation-based approach.

    Args:
        array: Input tensor/array to reshape
        shape: Target shape tuple

    Returns:
        Reshaped array in Fortran order
    """
    type_name = type(array).__module__.split(".")[0]

    if type_name in ("jax", "jaxlib"):
        import jax.numpy as jnp

        return jnp.reshape(array, shape, order="F")  # type: ignore[return-value]

    # Torch and MLX use permutation approach
    if len(array.shape) == 0:  # type: ignore[union-attr]
        if type_name == "torch":
            return array.reshape(shape)  # type: ignore[return-value,union-attr]
        elif type_name == "mlx":
            import mlx.core as mx

            return mx.reshape(array, shape)  # type: ignore[return-value]
        else:
            return np.reshape(array, shape, order="F")  # type: ignore[return-value]

    if type_name == "torch":
        x = array.permute(*reversed(range(len(array.shape))))  # type: ignore[union-attr]
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))  # type: ignore[return-value]

    elif type_name == "mlx":
        import mlx.core as mx

        x = mx.transpose(array, axes=tuple(reversed(range(len(array.shape)))))  # type: ignore[arg-type]
        reshaped = mx.reshape(x, tuple(reversed(shape)))
        if len(shape) > 0:
            reshaped = mx.transpose(reshaped, axes=tuple(reversed(range(len(shape)))))
        return reshaped  # type: ignore[return-value]

    else:
        # Numpy fallback
        return np.reshape(array, shape, order="F")  # type: ignore[return-value]


def apply_gp_log_transform(
    params: tuple[T, ...],
    ctx: "LayersContext",
) -> tuple[T, ...]:
    """Apply log transformation to geometric program (GP) parameters.

    Geometric programs are solved in log-space after conversion to DCP.
    This function applies log transformation to the appropriate parameters.
    Works with torch, jax, mlx, and numpy arrays.

    Args:
        params: Tuple of parameter tensors/arrays in original GP space
        ctx: Layer context containing GP parameter mapping info

    Returns:
        Tuple of transformed parameters (log-space for GP params, unchanged otherwise)
    """
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params

    type_name = type(params[0]).__module__.split(".")[0]

    # Get appropriate log function
    if type_name == "torch":
        import torch

        log_fn = torch.log
    elif type_name in ("jax", "jaxlib"):
        import jax.numpy as jnp

        log_fn = jnp.log
    elif type_name == "mlx":
        import mlx.core as mx

        log_fn = mx.log
    else:
        log_fn = np.log

    params_transformed = []
    for i, param in enumerate(params):
        cvxpy_param = ctx.parameters[i]
        if cvxpy_param in ctx.gp_param_to_log_param:
            params_transformed.append(log_fn(param))  # type: ignore[arg-type]
        else:
            params_transformed.append(param)
    return tuple(params_transformed)


def flatten_and_batch_params(
    params: tuple[T, ...],
    ctx: "LayersContext",
    batch: tuple,
) -> T:
    """Flatten and batch parameters into a single stacked tensor.

    Converts a tuple of parameter tensors (potentially with mixed batched/unbatched)
    into a single concatenated tensor suitable for matrix multiplication with the
    parametrized problem matrices. Works with torch, jax, mlx arrays.

    Args:
        params: Tuple of parameter tensors/arrays
        ctx: Layer context with batch info and ordering
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Concatenated parameter tensor with shape (num_params, batch_size) or (num_params,)
    """
    type_name = type(params[0]).__module__.split(".")[0]

    # Get framework-specific operations
    if type_name == "torch":
        import torch

        def expand_broadcast(p: T, batch_shape: tuple) -> T:
            return p.unsqueeze(0).expand(batch_shape + p.shape)  # type: ignore[union-attr,return-value]

        def ones_fn(shape: tuple, dtype, device) -> T:
            return torch.ones(shape, dtype=dtype, device=device)  # type: ignore[return-value]

        def cat_fn(tensors: list) -> T:
            return torch.cat(tensors, -1)  # type: ignore[return-value]

        def transpose_fn(t: T) -> T:
            return t.T  # type: ignore[union-attr,return-value]

        def get_device(p: T):  # type: ignore[misc]
            return p.device  # type: ignore[union-attr]

    elif type_name in ("jax", "jaxlib"):
        import jax.numpy as jnp

        def expand_broadcast(p: T, batch_shape: tuple) -> T:
            return jnp.broadcast_to(jnp.expand_dims(p, 0), batch_shape + p.shape)  # type: ignore[return-value,union-attr]

        def ones_fn(shape: tuple, dtype, device) -> T:
            return jnp.ones(shape, dtype=dtype)  # type: ignore[return-value]

        def cat_fn(tensors: list) -> T:
            return jnp.concatenate(tensors, -1)  # type: ignore[return-value]

        def transpose_fn(t: T) -> T:
            return t.T  # type: ignore[union-attr,return-value]

        def get_device(p: T):  # type: ignore[misc]
            return None

    elif type_name == "mlx":
        import mlx.core as mx

        def expand_broadcast(p: T, batch_shape: tuple) -> T:
            return mx.broadcast_to(mx.expand_dims(p, axis=0), batch_shape + p.shape)  # type: ignore[return-value,union-attr]

        def ones_fn(shape: tuple, dtype, device) -> T:
            return mx.ones(shape, dtype=dtype)  # type: ignore[return-value]

        def cat_fn(tensors: list) -> T:
            return mx.concatenate(tensors, axis=-1)  # type: ignore[return-value]

        def transpose_fn(t: T) -> T:
            return mx.transpose(t)  # type: ignore[return-value]

        def get_device(p: T):  # type: ignore[misc]
            return None

    else:
        # Numpy fallback (used by JAX's check_grads for numerical differentiation)
        def expand_broadcast(p: T, batch_shape: tuple) -> T:
            return np.broadcast_to(np.expand_dims(p, 0), batch_shape + p.shape)  # type: ignore[return-value,union-attr]

        def ones_fn(shape: tuple, dtype, device) -> T:
            return np.ones(shape, dtype=dtype)  # type: ignore[return-value]

        def cat_fn(tensors: list) -> T:
            return np.concatenate(tensors, -1)  # type: ignore[return-value]

        def transpose_fn(t: T) -> T:
            return t.T  # type: ignore[union-attr,return-value]

        def get_device(p: T):  # type: ignore[misc]
            return None

    flattened_params: list[T | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0 and batch:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            param_expanded = expand_broadcast(param, batch)
            flattened_params[ctx.user_order_to_col_order[i]] = reshape_fortran(
                param_expanded,
                batch + (-1,),
            )
        else:
            # Already batched or no batch dimension needed
            flattened_params[ctx.user_order_to_col_order[i]] = reshape_fortran(
                param,
                batch + (-1,),
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = ones_fn(
        batch + (1,),
        params[0].dtype,  # type: ignore[union-attr]
        get_device(params[0]),
    )
    assert all(p is not None for p in flattened_params), "All parameters must be assigned"

    p_stack = cat_fn([p for p in flattened_params if p is not None])
    # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
    if batch:
        p_stack = transpose_fn(p_stack)
    return p_stack


def recover_results(
    primal: T,
    dual: T,
    ctx: "LayersContext",
    batch: tuple,
) -> tuple[T, ...]:
    """Recover variable values from primal/dual solutions.

    Extracts the requested variables from the solver's primal and dual
    solutions, applies inverse GP transformation if needed, and removes
    batch dimension for unbatched inputs. Works with torch, jax, mlx arrays.

    Args:
        primal: Primal solution from solver
        dual: Dual solution from solver
        ctx: Layer context with variable recovery info
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Tuple of recovered variable values
    """
    type_name = type(primal).__module__.split(".")[0]

    # Get framework-specific operations
    if type_name == "torch":
        import torch

        exp_fn = torch.exp

        def squeeze_fn(t: T) -> T:
            return t.squeeze(0)  # type: ignore[union-attr,return-value]

        rf = reshape_fortran

    elif type_name in ("jax", "jaxlib"):
        import jax.numpy as jnp

        exp_fn = jnp.exp

        def squeeze_fn(t: T) -> T:
            return jnp.squeeze(t, 0)  # type: ignore[return-value]

        rf = None  # JAX uses default reshape with order="F"

    elif type_name == "mlx":
        import mlx.core as mx

        exp_fn = mx.exp

        def squeeze_fn(t: T) -> T:
            return mx.squeeze(t, axis=0)  # type: ignore[return-value]

        rf = reshape_fortran

    else:
        # Numpy fallback (used by JAX's check_grads for numerical differentiation)
        exp_fn = np.exp

        def squeeze_fn(t: T) -> T:
            return np.squeeze(t, 0)  # type: ignore[return-value]

        rf = None  # Numpy uses default reshape with order="F"

    # Extract each variable using its slice and reshape
    if rf is not None:
        results = tuple(var.recover(primal, dual, rf) for var in ctx.var_recover)
    else:
        results = tuple(var.recover(primal, dual) for var in ctx.var_recover)

    # Apply exp transformation to recover primal variables from log-space for GP
    # (dual variables stay in original space - no transformation needed)
    if ctx.gp:
        results = tuple(
            exp_fn(r) if var.primal is not None else r  # type: ignore[arg-type]
            for r, var in zip(results, ctx.var_recover)
        )

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = tuple(squeeze_fn(r) for r in results)  # type: ignore[arg-type]

    return results  # type: ignore[return-value]


@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None
    shape: tuple[int, ...]
    is_symmetric: bool = False  # True if primal variable is symmetric (requires svec unpacking)
    is_psd_dual: bool = False  # True if this is a PSD constraint dual (requires svec unpacking)

    def recover(self, primal_sol: T, dual_sol: T, reshape_fn=_default_reshape) -> T:
        """Extract and reshape variable from primal or dual solution.

        Args:
            primal_sol: Primal solution array
            dual_sol: Dual solution array
            reshape_fn: Function to reshape array with Fortran order semantics.
                        Defaults to numpy-style reshape with order="F".
        """
        batch = tuple(primal_sol.shape[:-1])  # type: ignore[attr-defined]
        if self.primal is not None:
            data = primal_sol[..., self.primal]  # type: ignore[index]
            if self.is_symmetric:
                # Unpack symmetric primal variable from svec format
                return _unpack_primal_svec(data, self.shape[0], batch)
            return reshape_fn(data, batch + self.shape)  # type: ignore[index]
        if self.dual is not None:
            data = dual_sol[..., self.dual]  # type: ignore[index]
            if self.is_psd_dual:
                # Unpack scaled vectorized form (svec) to full symmetric matrix
                return _unpack_svec(data, self.shape[0], batch)
            return reshape_fn(data, batch + self.shape)  # type: ignore[index]
        raise RuntimeError(
            "Invalid VariableRecovery: both primal and dual slices are None. "
            "At least one must be set to recover variable values."
        )


def _svec_to_symmetric(
    svec: T,
    n: int,
    batch: tuple,
    rows: np.ndarray,
    cols: np.ndarray,
    scale: np.ndarray | None = None,
) -> T:
    """Convert vectorized form to full symmetric matrix.

    Args:
        svec: Vectorized form, shape (*batch, n*(n+1)/2)
        n: Matrix dimension
        batch: Batch dimensions
        rows: Row indices for unpacking
        cols: Column indices for unpacking
        scale: Optional scaling factors for each element (for svec format with sqrt(2) scaling)

    Returns:
        Full symmetric matrix, shape (*batch, n, n)
    """
    # Detect framework and dispatch
    type_name = type(svec).__module__.split(".")[0]

    if type_name == "torch":
        import torch

        rows_t = torch.tensor(rows, dtype=torch.long, device=svec.device)  # type: ignore[union-attr]
        cols_t = torch.tensor(cols, dtype=torch.long, device=svec.device)  # type: ignore[union-attr]
        if scale is not None:
            scale_t = torch.tensor(scale, dtype=svec.dtype, device=svec.device)  # type: ignore[union-attr]
            data = svec * scale_t  # type: ignore[operator]
        else:
            data = svec

        out_shape = batch + (n, n)
        if batch:
            batch_size = int(np.prod(batch))
            data_flat = data.reshape(batch_size, -1)  # type: ignore[union-attr]
            result = torch.zeros(batch_size, n, n, dtype=svec.dtype, device=svec.device)  # type: ignore[union-attr]
            result[:, rows_t, cols_t] = data_flat
            result[:, cols_t, rows_t] = data_flat
            return result.reshape(out_shape)  # type: ignore[return-value]
        else:
            result = torch.zeros(n, n, dtype=svec.dtype, device=svec.device)  # type: ignore[union-attr]
            result[rows_t, cols_t] = data  # type: ignore[index]
            result[cols_t, rows_t] = data  # type: ignore[index]
            return result  # type: ignore[return-value]

    elif type_name == "jax" or type_name == "jaxlib":
        import jax.numpy as jnp

        rows_arr = jnp.array(rows)
        cols_arr = jnp.array(cols)
        data = svec * jnp.array(scale) if scale is not None else svec  # type: ignore[operator]
        out_shape = batch + (n, n)
        result = jnp.zeros(out_shape, dtype=svec.dtype)  # type: ignore[union-attr]
        result = result.at[..., rows_arr, cols_arr].set(data)
        result = result.at[..., cols_arr, rows_arr].set(data)
        return result  # type: ignore[return-value]

    elif type_name == "mlx":
        import mlx.core as mx

        # MLX doesn't support advanced indexing like torch/jax, use scatter approach
        if scale is not None:
            scale_mx = mx.array(scale, dtype=svec.dtype)  # type: ignore[union-attr]
            data = svec * scale_mx  # type: ignore[operator]
        else:
            data = svec

        out_shape = batch + (n, n)
        if batch:
            batch_size = int(np.prod(batch))
            data_flat = mx.reshape(data, (batch_size, -1))  # type: ignore[arg-type]
            # Build result by iterating (MLX lacks advanced indexing)
            results = []
            for b in range(batch_size):
                data_b = data_flat[b]
                # Use numpy for indexing, then convert
                result_np = np.zeros((n, n), dtype=np.float64)
                data_np = np.array(data_b)
                result_np[rows, cols] = data_np
                result_np[cols, rows] = data_np
                results.append(mx.array(result_np, dtype=svec.dtype))  # type: ignore[union-attr]
            result = mx.stack(results, axis=0)
            return mx.reshape(result, out_shape)  # type: ignore[return-value]
        else:
            # Unbatched: simple approach via numpy
            data_np = np.array(data)
            result_np = np.zeros((n, n), dtype=np.float64)
            result_np[rows, cols] = data_np
            result_np[cols, rows] = data_np
            return mx.array(result_np, dtype=svec.dtype)  # type: ignore[return-value,union-attr]

    else:
        # Numpy path
        svec_np = np.asarray(svec)
        data = svec_np * scale if scale is not None else svec_np
        out_shape = batch + (n, n)
        result = np.zeros(out_shape, dtype=svec_np.dtype)
        result[..., rows, cols] = data
        result[..., cols, rows] = data
        return result  # type: ignore[return-value]


def _unpack_primal_svec(svec: T, n: int, batch: tuple) -> T:
    """Unpack symmetric primal variable from vectorized form.

    CVXPY stores symmetric variables in upper triangular row-major order:
    [X[0,0], X[0,1], ..., X[0,n-1], X[1,1], X[1,2], ..., X[n-1,n-1]]
    """
    # Vectorized: upper triangular row-major indices
    rows, cols = np.triu_indices(n)
    return _svec_to_symmetric(svec, n, batch, rows, cols)


def _unpack_svec(svec: T, n: int, batch: tuple) -> T:
    """Unpack scaled vectorized (svec) form to full symmetric matrix.

    The svec format stores a symmetric n x n matrix as a vector of length n*(n+1)/2,
    with off-diagonal elements scaled by sqrt(2). Uses column-major lower triangular
    ordering: (0,0), (1,0), (1,1), (2,0), ...
    """
    # Vectorized: lower triangular column-major indices
    # np.tril_indices returns row-major order, need to reorder to column-major
    rows_rm, cols_rm = np.tril_indices(n)
    # Sort by column first, then by row for column-major order
    sort_idx = np.lexsort((rows_rm, cols_rm))
    rows = rows_rm[sort_idx]
    cols = cols_rm[sort_idx]
    # Scale: 1.0 for diagonal, 1/sqrt(2) for off-diagonal
    scale = np.where(rows == cols, 1.0, 1.0 / np.sqrt(2.0))
    return _svec_to_symmetric(svec, n, batch, rows, cols, scale)


@dataclass
class LayersContext:
    parameters: list[cp.Parameter]
    reduced_P: scipy.sparse.csr_array
    q: scipy.sparse.csr_array | None
    reduced_A: scipy.sparse.csr_array
    cone_dims: dict[str, int | list[int]]
    solver_ctx: SolverContext
    solver: str
    var_recover: list[VariableRecovery]
    user_order_to_col_order: dict[int, int]
    batch_sizes: list[int] | None = (
        None  # Track which params are batched (0=unbatched, N=batch size)
    )
    # GP (Geometric Programming) support
    gp: bool = False
    # Maps original GP parameters to their log-space DCP parameters
    # Used to determine which parameters need log transformation in forward pass
    gp_param_to_log_param: dict[cp.Parameter, cp.Parameter] | None = None

    def validate_params(self, values: list) -> tuple:
        if len(values) != len(self.parameters):
            raise ValueError(
                f"A tensor must be provided for each CVXPY parameter; "
                f"received {len(values)} tensors, expected {len(self.parameters)}",
            )

        # Determine batch size from all parameters
        batch_sizes = []
        for i, (value, param) in enumerate(zip(values, self.parameters, strict=False)):
            # Check if value has the right shape (with or without batch dimension)
            if len(value.shape) == len(param.shape):
                # No batch dimension for this parameter
                if value.shape != param.shape:
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected: {param.shape}, Got: {value.shape}",
                    )
                batch_sizes.append(0)
            elif len(value.shape) == len(param.shape) + 1:
                # Has batch dimension
                if value.shape[1:] != param.shape:
                    shape_str = ", ".join(map(str, param.shape))
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected batched shape: (batch_size, {shape_str}), "
                        f"Got: {value.shape}",
                    )
                batch_sizes.append(value.shape[0])
            else:
                raise ValueError(
                    f"Invalid parameter dimensionality for parameter {i}. "
                    f"Expected {len(param.shape)} or {len(param.shape) + 1} dimensions, "
                    f"Got: {len(value.shape)} dimensions",
                )

        # Check that all non-zero batch sizes are the same
        nonzero_batch_sizes = [b for b in batch_sizes if b > 0]
        if nonzero_batch_sizes:
            batch_size = nonzero_batch_sizes[0]
            if not all(b == batch_size for b in nonzero_batch_sizes):
                raise ValueError(
                    f"Inconsistent batch sizes. Expected all batched parameters to have "
                    f"the same batch size, but got: {batch_sizes}",
                )
            # Store batch_sizes for use in forward pass
            self.batch_sizes = batch_sizes
            return (batch_size,)
        self.batch_sizes = batch_sizes
        return ()


def _find_parent_constraint(var: cp.Variable, problem: cp.Problem) -> cp.Constraint | None:
    """Find the constraint whose dual_variables contains this variable."""
    for con in problem.constraints:
        if any(var.id == dv.id for dv in con.dual_variables):
            return con
    return None


def _dual_var_offset(var: cp.Variable, constraint: cp.Constraint) -> int:
    """Get the offset of a dual variable within its parent constraint's dual vector."""
    offset = 0
    for dv in constraint.dual_variables:
        if dv.id == var.id:
            return offset
        offset += dv.size
    raise ValueError(f"Variable {var} not found in constraint {constraint}")


def _build_primal_recovery(var: cp.Variable, param_prob: ParamConeProg) -> VariableRecovery:
    """Build recovery info for a primal variable."""
    start = param_prob.var_id_to_col[var.id]
    is_sym = hasattr(var, "is_symmetric") and var.is_symmetric() and len(var.shape) >= 2

    if is_sym:
        n = var.shape[0]  # type: ignore[index]
        svec_size = n * (n + 1) // 2
        return VariableRecovery(
            primal=slice(start, start + svec_size),
            dual=None,
            shape=var.shape,
            is_symmetric=True,
        )
    return VariableRecovery(
        primal=slice(start, start + var.size),
        dual=None,
        shape=var.shape,
    )


def _build_dual_recovery(
    var: cp.Variable,
    parent_con: cp.Constraint,
    constr_id_to_slice: dict[int, slice],
) -> VariableRecovery:
    """Build recovery info for a dual variable."""
    constr_slice = constr_id_to_slice[parent_con.id]
    offset = _dual_var_offset(var, parent_con)
    dual_start = constr_slice.start + offset

    is_psd = isinstance(parent_con, cvxpy.constraints.PSD)
    # PSD duals are stored in svec format: n*(n+1)//2 elements
    if is_psd:
        n = var.shape[0]
        dual_size = n * (n + 1) // 2
    else:
        dual_size = var.size

    return VariableRecovery(
        primal=None,
        dual=slice(dual_start, dual_start + dual_size),
        shape=var.shape,
        is_psd_dual=is_psd,
    )


def _build_constr_id_to_slice(param_prob: ParamConeProg) -> dict[int, slice]:
    """Build mapping from constraint ID to slice in dual solution vector.

    The dual solution vector is ordered by cone type:
    Zero (equalities) -> NonNeg (inequalities) -> SOC -> ExpCone -> PSD -> PowCone3D

    Args:
        param_prob: CVXPY's parametrized cone program

    Returns:
        Dictionary mapping constraint ID to slice in dual solution vector
    """
    constr_id_to_slice: dict[int, slice] = {}
    cur_idx = 0

    # Process each cone type in canonical order
    cone_types = [
        cvxpy.constraints.Zero,
        cvxpy.constraints.NonNeg,
        cvxpy.constraints.SOC,
        cvxpy.constraints.ExpCone,
        cvxpy.constraints.PSD,
        cvxpy.constraints.PowCone3D,
    ]

    for cone_type in cone_types:
        for c in param_prob.constr_map.get(cone_type, []):
            # PSD constraints use scaled vectorization (svec) in the dual
            # For an n x n PSD constraint, svec size is n*(n+1)//2
            if cone_type is cvxpy.constraints.PSD:
                n = c.shape[0]  # Matrix dimension
                cone_size = n * (n + 1) // 2
            else:
                cone_size = c.size
            constr_id_to_slice[c.id] = slice(cur_idx, cur_idx + cone_size)
            cur_idx += cone_size

    return constr_id_to_slice


def _validate_problem(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    gp: bool,
) -> None:
    """Validate that the problem is DPP-compliant and inputs are well-formed.

    Args:
        problem: CVXPY problem to validate
        variables: List of CVXPY variables to track (can include constraint dual variables)
        parameters: List of CVXPY parameters
        gp: Whether this is a geometric program (GP)

    Raises:
        ValueError: If problem is not DPP-compliant or inputs are invalid
    """
    # Check if problem follows disciplined parametrized programming (DPP) rules
    if gp:
        if not problem.is_dgp(dpp=True):  # type: ignore[call-arg]
            raise ValueError("Problem must be DPP for geometric programming.")
    else:
        if not problem.is_dcp(dpp=True):  # type: ignore[call-arg]
            raise ValueError("Problem must be DPP.")

    # Validate parameters match problem definition
    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match problem.parameters")
    if not isinstance(parameters, list) and not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as a list or tuple")
    if not isinstance(variables, list) and not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as a list or tuple")

    # Validate variables: each must be either a primal variable or a constraint dual variable
    primal_vars = set(problem.variables())
    for v in variables:
        if v in primal_vars:
            continue  # Valid primal variable
        parent_con = _find_parent_constraint(v, problem)
        if parent_con is None:
            raise ValueError(
                f"Variable {v} must be a subset of problem.variables or a constraint dual variable"
            )


def _build_user_order_mapping(
    parameters: list[cp.Parameter],
    param_prob: ParamConeProg,
    gp: bool,
    gp_param_to_log_param: dict[cp.Parameter, cp.Parameter] | None,
) -> dict[int, int]:
    """Build mapping from user parameter order to column order.

    CVXPY internally reorders parameters when canonicalizing problems. This
    creates a mapping from the user's parameter order to the internal column
    order used in the canonical form.

    Args:
        parameters: List of CVXPY parameters in user order
        param_prob: CVXPY's parametrized problem object
        gp: Whether this is a geometric program
        gp_param_to_log_param: Mapping from GP params to log-space DCP params

    Returns:
        Dictionary mapping user parameter index to column order index
    """
    # For GP problems, we need to use the log-space DCP parameter IDs
    if gp and gp_param_to_log_param:
        # Map user order index to column using log-space DCP parameters
        user_order_to_col = {
            i: param_prob.param_id_to_col[
                gp_param_to_log_param[p].id if p in gp_param_to_log_param else p.id
            ]
            for i, p in enumerate(parameters)
        }
    else:
        # Standard DCP problem - use original parameters
        user_order_to_col = {
            i: col
            for col, i in sorted(
                [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(parameters)],
            )
        }

    # Convert column indices to sequential order mapping
    user_order_to_col_order = {}
    for j, i in enumerate(user_order_to_col.keys()):
        user_order_to_col_order[i] = j

    return user_order_to_col_order


def parse_args(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    solver: str | None,
    gp: bool = False,
    verbose: bool = False,
    canon_backend: str | None = None,
    solver_args: dict[str, Any] | None = None,
) -> LayersContext:
    # Validate problem is DPP (disciplined parametrized programming)
    _validate_problem(problem, variables, parameters, gp)

    if solver is None:
        solver = "DIFFCP"

    # Handle GP problems using native CVXPY reduction (cvxpy >= 1.7.4)
    gp_param_to_log_param = None
    if gp:
        # Apply native CVXPY DGP→DCP reduction
        dgp2dcp = cp.reductions.Dgp2Dcp(problem)  # type: ignore[attr-defined]
        dcp_problem, _ = dgp2dcp.apply(problem)

        # Extract parameter mapping from the reduction
        gp_param_to_log_param = dgp2dcp.canon_methods._parameters

        # Get problem data from the already-transformed DCP problem
        data, _, _ = dcp_problem.get_problem_data(
            solver=solver,
            gp=False,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_args,
        )
    else:
        # Standard DCP path
        data, _, _ = problem.get_problem_data(
            solver=solver,
            gp=False,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_args,
        )

    param_prob = data[cp.settings.PARAM_PROB]  # type: ignore[attr-defined]
    cone_dims = data["dims"]

    # Create solver context
    solver_ctx = cvxpylayers.interfaces.get_solver_ctx(
        solver,
        param_prob,
        cone_dims,
        data,
        solver_args,
    )

    # Build parameter ordering mapping
    user_order_to_col_order = _build_user_order_mapping(
        parameters, param_prob, gp, gp_param_to_log_param
    )

    q = getattr(param_prob, "q", getattr(param_prob, "c", None))

    # Build variable recovery info for each requested variable
    constr_id_to_slice = _build_constr_id_to_slice(param_prob)
    primal_vars = set(problem.variables())

    var_recover = []
    for v in variables:
        if v in primal_vars:
            var_recover.append(_build_primal_recovery(v, param_prob))
        else:
            parent_con = _find_parent_constraint(v, problem)
            assert parent_con is not None  # Already validated
            var_recover.append(_build_dual_recovery(v, parent_con, constr_id_to_slice))

    return LayersContext(
        parameters,
        param_prob.reduced_P,
        q,
        param_prob.reduced_A,
        cone_dims,
        solver_ctx,  # type: ignore[arg-type]
        solver,
        var_recover=var_recover,
        user_order_to_col_order=user_order_to_col_order,
        gp=gp,
        gp_param_to_log_param=gp_param_to_log_param,
    )
