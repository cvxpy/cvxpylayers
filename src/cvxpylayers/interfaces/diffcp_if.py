from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import diffcp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # Type alias for multi-framework tensor types
    TensorLike = Union[torch.Tensor, jnp.ndarray, np.ndarray, mx.array]
else:
    TensorLike = Any


def _detect_batch_size(con_values: TensorLike) -> Tuple[int, bool]:
    """Detect batch size and whether input was originally unbatched.

    Handles both PyTorch tensors and JAX arrays by checking the number
    of dimensions.

    Args:
        con_values: Constraint values (torch.Tensor or jnp.ndarray)

    Returns:
        Tuple of (batch_size, originally_unbatched) where:
            - batch_size: Number of batch elements (1 if unbatched)
            - originally_unbatched: True if input had no batch dimension
    """
    # Handle both torch tensors (.dim()) and jax/numpy/mlx arrays (.ndim)
    ndim = (
            con_values.dim() if hasattr(con_values, "dim")
            else con_values.ndim  # type: ignore[attr-defined]
            )
    if ndim == 1:
        return 1, True  # Unbatched input
    else:
        return con_values.shape[1], False  # Batched input


def _build_diffcp_matrices(
    con_values: TensorLike,
    lin_obj_values: TensorLike,
    A_structure: Tuple[np.ndarray, np.ndarray],
    A_shape: Tuple[int, int],
    b_idx: np.ndarray,
    batch_size: int,
) -> Tuple[List[sp.csc_matrix],
           List[np.ndarray],
           List[np.ndarray],
           List[np.ndarray]]:
    """Build DIFFCP matrices from constraint and objective values.

    Converts parameter values into the conic form required by DIFFCP solver:
        minimize c^T x subject to Ax + s = b, s in K
    where K is a product of cones.

    Args:
        con_values: Constraint coefficient values (batched)
        lin_obj_values: Linear objective coefficient values (batched)
        A_structure: Sparse matrix structure (indices, indptr)
        A_shape: Shape of augmented constraint matrix
        b_idx: Indices for extracting RHS from last column
        batch_size: Number of batch elements

    Returns:
        Tuple of (As, bs, cs, b_idxs) where:
            - As: List of constraint matrices (one per batch element)
            - bs: List of RHS vectors (one per batch element)
            - cs: List of cost vectors (one per batch element)
            - b_idxs: List of RHS index arrays (one per batch element)
    """
    As, bs, cs, b_idxs = [], [], [], []

    for i in range(batch_size):
        # Convert to numpy - handles both torch tensors and jax arrays
        con_vals_i = np.array(con_values[:, i])
        lin_vals_i = np.array(lin_obj_values[:-1, i])

        # Build augmented matrix [A | b] from sparse structure
        A_aug = sp.csc_matrix(
            (con_vals_i, *A_structure),
            shape=A_shape,
        )
        # Extract A and b, negating A to match DIFFCP convention
        As.append(-A_aug[:, :-1])
        bs.append(A_aug[:, -1].toarray().flatten())
        cs.append(lin_vals_i)
        b_idxs.append(b_idx)

    return As, bs, cs, b_idxs


class DIFFCP_ctx:
    c_slice: slice

    A_idxs: np.ndarray
    b_idx: np.ndarray
    A_structure: Tuple[np.ndarray, np.ndarray]
    A_shape: Tuple[int, int]

    G_idxs: np.ndarray
    h_slice: slice
    G_structure: Tuple[np.ndarray, np.ndarray]
    G_shape: Tuple[int, int]

    solver: Callable

    def __init__(
        self,
        objective_structure,
        constraint_structure,
        dims,
        lower_bounds,
        upper_bounds,
        options=None,
    ):
        con_indices, con_ptr, (m, np1) = constraint_structure

        self.A_structure = (con_indices, con_ptr)
        self.A_shape = (m, np1)
        self.b_idx = con_indices[con_ptr[-2]: con_ptr[-1]]

        self.dims = dims

    def torch_to_data(self, quad_obj_values,
                      lin_obj_values, con_values) -> "DIFFCP_data":
        batch_size, originally_unbatched = _detect_batch_size(con_values)

        # Add batch dimension for uniform handling if needed
        if originally_unbatched:
            con_values = con_values.unsqueeze(1)
            lin_obj_values = lin_obj_values.unsqueeze(1)

        # Build matrices
        As, bs, cs, b_idxs = _build_diffcp_matrices(
            con_values,
            lin_obj_values,
            self.A_structure,
            self.A_shape,
            self.b_idx,
            batch_size,
        )

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            b_idxs=b_idxs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )

    def jax_to_data(self, quad_obj_values,
                    lin_obj_values, con_values) -> "DIFFCP_data":
        if jnp is None:
            raise ImportError(
                "JAX interface requires 'jax' package to be installed. "
                "Install with: pip install jax"
            )

        batch_size, originally_unbatched = _detect_batch_size(con_values)

        # Add batch dimension for uniform handling if needed
        if originally_unbatched:
            con_values = jnp.expand_dims(con_values, 1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, 1)

        # Build matrices
        As, bs, cs, b_idxs = _build_diffcp_matrices(
            con_values,
            lin_obj_values,
            self.A_structure,
            self.A_shape,
            self.b_idx,
            batch_size,
        )

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            b_idxs=b_idxs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )

    def mlx_to_data(self, quad_obj_values,
                    lin_obj_values, con_values) -> "DIFFCP_data":

        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        # Convert to numpy arrays if they're MLX arrays
        if isinstance(con_values, np.ndarray):
            con_values_np = con_values
        else:
            con_values_np = np.array(con_values, dtype=np.float32)

        if isinstance(lin_obj_values, np.ndarray):
            lin_obj_values_np = lin_obj_values
        else:
            lin_obj_values_np = np.array(lin_obj_values, dtype=np.float32)

        batch_size, originally_unbatched = _detect_batch_size(con_values_np)

        # Add batch dimension for uniform handling if needed
        if originally_unbatched:
            con_values_np = np.expand_dims(con_values_np, 1)
            lin_obj_values_np = np.expand_dims(lin_obj_values_np, 1)

        # Build matrices
        As, bs, cs, b_idxs = _build_diffcp_matrices(
            con_values_np,
            lin_obj_values_np,
            self.A_structure,
            self.A_shape,
            self.b_idx,
            batch_size,
        )

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            b_idxs=b_idxs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )


def _compute_gradients(
    adj_batch: Callable,
    dprimal: TensorLike,
    ddual: TensorLike,
    bs: List[np.ndarray],
    b_idxs: List[np.ndarray],
    batch_size: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute gradients using DIFFCP's adjoint method.

    Uses implicit differentiation to compute gradients of the optimization
    solution with respect to problem parameters. The adjoint method efficiently
    computes these gradients by solving the adjoint system.

    Args:
        adj_batch: DIFFCP's batch adjoint function
        dprimal: Incoming gradients w.r.t. primal solution
        ddual: Incoming gradients w.r.t. dual solution
        bs: List of RHS vectors from forward pass
        b_idxs: List of RHS indices from forward pass
        batch_size: Number of batch elements

    Returns:
        Tuple of (dq_batch, dA_batch) where:
            - dq_batch: List of gradients w.r.t. linear objective coefficients
            - dA_batch: List of gradients w.r.t. constraint coefficients
    """
    # Convert incoming gradients to lists for DIFFCP
    dxs = [np.array(dprimal[i]) for i in range(batch_size)]
    dys = [np.array(ddual[i]) for i in range(batch_size)]
    dss = [np.zeros_like(bs[i]) for i in range(batch_size)]

    # Call DIFFCP's batch adjoint to get gradients w.r.t. problem data
    dAs, dbs, dcs = adj_batch(dxs, dys, dss)

    # Aggregate gradients from each batch element
    dq_batch = []
    dA_batch = []
    for i in range(batch_size):
        # Negate dA because A was negated in forward pass,
        #but not db (b was not negated)
        con_grad = np.hstack([-dAs[i].data, dbs[i][b_idxs[i]]])
        # Add zero gradient for constant offset term
        lin_grad = np.hstack([dcs[i], np.array([0.0])])
        dA_batch.append(con_grad)
        dq_batch.append(lin_grad)

    return dq_batch, dA_batch


@dataclass
class DIFFCP_data:
    As: List[sp.csc_matrix]
    bs: List[np.ndarray]
    cs: List[np.ndarray]
    b_idxs: List[np.ndarray]
    cone_dict: Dict[str, Union[int, List[int]]]
    batch_size: int
    originally_unbatched: bool

    def torch_solve(self, solver_args=None):
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package."
                "Install with: pip install torch"
            )

        if solver_args is None:
            solver_args = {}

        # Always use batch solve
        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **solver_args,
        )
        # Stack results into batched tensors
        primal = torch.stack([torch.from_numpy(x) for x in xs])
        dual = torch.stack([torch.from_numpy(y) for y in ys])
        return primal, dual, adj_batch

    def torch_derivative(self, primal, dual, adj_batch):
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package."
                "Install with: pip install torch"
            )

        # Compute gradients
        dq_batch, dA_batch = _compute_gradients(
            adj_batch, primal, dual, self.bs, self.b_idxs, self.batch_size
        )

        # Stack into shape (num_entries, batch_size)
        dq_stacked = torch.stack([torch.from_numpy(g) for g in dq_batch]).T
        dA_stacked = torch.stack([torch.from_numpy(g) for g in dA_batch]).T

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dq_stacked = dq_stacked.squeeze(1)
            dA_stacked = dA_stacked.squeeze(1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )

    def jax_solve(self, solver_args=None):
        if solver_args is None:
            solver_args = {}

        # Always use batch solve
        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **solver_args,
        )

        # Stack results into batched arrays
        primal = jnp.stack([jnp.array(x) for x in xs])
        dual = jnp.stack([jnp.array(y) for y in ys])

        # Return primal, dual, and adjoint function for backward pass
        return primal, dual, adj_batch

    def jax_derivative(self, dprimal, ddual, adj_batch):
        # Compute gradients
        dq_batch, dA_batch = _compute_gradients(
            adj_batch, dprimal, ddual, self.bs, self.b_idxs, self.batch_size
        )

        # Stack into shape (num_entries, batch_size)
        dq_stacked = jnp.stack([jnp.array(g) for g in dq_batch]).T
        dA_stacked = jnp.stack([jnp.array(g) for g in dA_batch]).T

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dq_stacked = jnp.squeeze(dq_stacked, 1)
            dA_stacked = jnp.squeeze(dA_stacked, 1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )

    def mlx_solve(self, solver_args=None):
        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        if solver_args is None:
            solver_args = {}

        # Always use batch solve
        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **solver_args,
        )
        # Stack results into batched arrays
        # MLX doesn't support float64 on GPU, use float32 instead
        primal = mx.stack([mx.array(x, dtype=mx.float32) for x in xs])
        dual = mx.stack([mx.array(y, dtype=mx.float32) for y in ys])
        return primal, dual, adj_batch

    def mlx_derivative(self, dprimal, ddual, adj_batch):
        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        # Convert MLX arrays to numpy for gradient computation
        # Handle both batched and unbatched cases
        dprimal_np = np.array(dprimal, dtype=np.float32)
        ddual_np = np.array(ddual, dtype=np.float32)

        # Ensure proper shape for _compute_gradients
        # which expects indexable arrays
        # If unbatched, add batch dimension
        if dprimal_np.ndim == 1:
            dprimal_np = dprimal_np[np.newaxis, :]
        if ddual_np.ndim == 1:
            ddual_np = ddual_np[np.newaxis, :]

        # Compute gradients
        dq_batch, dA_batch = _compute_gradients(
            adj_batch, dprimal_np, ddual_np, self.bs, self.b_idxs, self.batch_size
        )

        # Stack into shape (num_entries, batch_size) and convert to MLX
        # MLX doesn't support float64 on GPU, use float32 instead
        dq_stacked = mx.stack([mx.array(g, dtype=mx.float32) for g in dq_batch])
        dq_stacked = mx.transpose(dq_stacked)
        dA_stacked = mx.stack([mx.array(g, dtype=mx.float32) for g in dA_batch])
        dA_stacked = mx.transpose(dA_stacked)

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dq_stacked = mx.squeeze(dq_stacked, 1)
            dA_stacked = mx.squeeze(dA_stacked, 1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )
