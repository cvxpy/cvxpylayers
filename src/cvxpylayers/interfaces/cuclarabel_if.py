from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import diffqcp  # type: ignore
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.cuclarabel_conif import dims_to_solver_cones
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

from cvxpylayers.utils.solver_utils import (
    JuliaCuVector2CuPyArray,
    convert_csc_structure_to_csr_structure,
)

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

import cupy
from cupyx.scipy.sparse import csr_matrix as cucsr_matrix
from juliacall import Main as jl  # type: ignore

if TYPE_CHECKING:
    # Type alias for multi-framework tensor types
    TensorLike = torch.Tensor | jnp.ndarray | np.ndarray | cupy.ndarray
else:
    TensorLike = Any


def _detect_batch_size(con_values: TensorLike) -> tuple[int, bool]:
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
    # Handle both torch tensors (.dim()) and jax/numpy arrays (.ndim)
    ndim = con_values.dim() if hasattr(con_values, "dim") else con_values.ndim  # type: ignore[attr-defined]

    if ndim == 1:
        return 1, True  # Unbatched input
    else:
        return con_values.shape[1], False  # Batched input


def _build_cuclarabel_matrices(
    con_values: TensorLike,
    quad_obj_values: TensorLike,
    lin_obj_values: TensorLike,
    P_idx: cupy.ndarray,
    P_structure: tuple[cupy.ndarray, cupy.ndarray],
    P_shape: tuple[int, int],
    A_idx: cupy.ndarray,
    A_structure: tuple[cupy.ndarray, cupy.ndarray],
    A_shape: tuple[int, int],
    b_idx: cupy.ndarray,
    batch_size: int,
) -> tuple[
    list[cucsr_matrix], list[np.ndarray], list[cucsr_matrix], list[np.ndarray], list[np.ndarray]
]:
    """Build CUCLARABEL matrices from constraint and objective values.

    Converts parameter values into the conic form required by CUCLARABEL:
        minimize 1/2 x^T P x + q^T x subject to Ax + s = b, s in K
    where K is a product of cones.

    Args:
        con_values: Constraint coefficient values (batched)
        quad_obj_values: Quadratic objective coefficient values (batched)
        lin_obj_values: Linear objective coefficient values (batched)
        P_structure: Sparse matrix structure for objective (indices, indptr)
        P_shape: Shape of objective matrix, must be square
        A_structure: Sparse matrix structure (indices, indptr)
        A_shape: Shape of augmented constraint matrix
        b_idx: Indices for extracting RHS from last column
        batch_size: Number of batch elements

    Returns:
        Tuple of (Ps, As, bs, cs, b_idxs) where:
            - Ps: List of objective matrices (one per batch element)
            - qs: List of objective linear part vectors (one per batch element)
            - As: List of constraint matrices (one per batch element)
            - bs: List of RHS vectors (one per batch element)
            - b_idxs: List of RHS index arrays (one per batch element)
    """
    Ps, qs, As, bs, b_idxs = [], [], [], [], []

    for i in range(batch_size):
        # Convert to cupy - handles both torch tensors and jax arrays
        quad_vals_i = cupy.array(quad_obj_values[:, i]) if quad_obj_values is not None else None
        con_vals_i = cupy.array(con_values[:, i])
        lin_vals_i = cupy.array(lin_obj_values[:-1, i])

        P = (
            cucsr_matrix(
                (quad_vals_i[P_idx], *P_structure),
                shape=P_shape,
            )
            if P_idx is not None and quad_vals_i is not None
            else cucsr_matrix((cupy.array([]), *P_structure), shape=P_shape)
        )

        A = cucsr_matrix(
            (con_vals_i[A_idx], *A_structure),
            shape=A_shape,
        )

        b = cupy.zeros(A_shape[0])
        b[b_idx] = con_vals_i[-b_idx.size :]
        # Extract A and b, negating A to match CUCLARABEL convention
        Ps.append(P)
        As.append(-A)
        bs.append(b)
        qs.append(lin_vals_i)
        b_idxs.append(b_idx)

    return Ps, qs, As, bs, b_idxs


def _call_cuclarabel(Ps, qs, As, bs, cones):
    xs = []
    ys = []
    ss = []

    for i in range(len(Ps)):
        P, q, A, b = Ps[i], qs[i], As[i], bs[i]
        if P.nnz != 0:
            jl.P = jl.Clarabel.cupy_to_cucsrmat(
                jl.Float64,
                int(P.data.data.ptr),
                int(P.indices.data.ptr),
                int(P.indptr.data.ptr),
                *P.shape,
                P.nnz,
            )
        else:
            nvars = P.shape[0]
            jl.seval(f"""
            P = CuSparseMatrixCSR(sparse(Float64[], Float64[], Float64[], {nvars}, {nvars}))
            """)
        jl.q = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(q.data.ptr), q.size)

        jl.A = jl.Clarabel.cupy_to_cucsrmat(
            jl.Float64,
            int(A.data.data.ptr),
            int(A.indices.data.ptr),
            int(A.indptr.data.ptr),
            *A.shape,
            A.nnz,
        )
        jl.b = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(b.data.ptr), b.size)

        dims_to_solver_cones(jl, cones)

        jl.seval("""
        settings = Clarabel.Settings(direct_solve_method = :cudss)
        solver   = Clarabel.Solver(settings)
        solver   = Clarabel.setup!(solver, P,q,A,b,cones)
        Clarabel.solve!(solver)
        """)
        xs.append(JuliaCuVector2CuPyArray(jl, jl.solver.solution.x))
        ys.append(JuliaCuVector2CuPyArray(jl, jl.solver.solution.z))
        ss.append(JuliaCuVector2CuPyArray(jl, jl.solver.solution.s))
    return (
        xs,
        ys,
        ss,
        {
            "P": Ps,
            "q": qs,
            "A": As,
            "b": bs,
            "x": xs,
            "y": ys,
            "s": ss,
        },
    )


class CUCLARABEL_ctx:
    P_idxs: cupy.ndarray | None
    P_structure: tuple[cupy.ndarray, cupy.ndarray]
    P_shape: tuple[int, int]

    A_idxs: cupy.ndarray
    b_idx: cupy.ndarray
    A_structure: tuple[cupy.ndarray, cupy.ndarray]
    A_shape: tuple[int, int]

    def __init__(
        self,
        objective_structure,
        constraint_structure,
        dims,
        lower_bounds,
        upper_bounds,
        options=None,
    ):
        A_shuffle, A_structure, A_shape, b_idx = convert_csc_structure_to_csr_structure(
            constraint_structure, True
        )

        if objective_structure is not None:
            P_shuffle, P_structure, P_shape = convert_csc_structure_to_csr_structure(
                objective_structure, False
            )
            assert P_shape[0] == P_shape[1]
        else:
            P_shuffle = None
            P_structure = (np.array([], dtype=int), np.array([0] * (A_shape[1] + 1)))
            P_shape = (A_shape[1], A_shape[1])

        assert P_shape[0] == A_shape[1]

        self.P_idxs = cupy.array(P_shuffle) if P_shuffle is not None else None
        self.P_structure = tuple(cupy.array(arr) for arr in P_structure)
        self.P_shape = P_shape

        self.A_idxs = cupy.array(A_shuffle)
        self.A_structure = tuple(cupy.array(arr) for arr in A_structure)
        self.A_shape = A_shape
        self.b_idx = cupy.array(b_idx)

        self.dims = dims

        jl.seval("""using Clarabel, LinearAlgebra, SparseArrays""")
        jl.seval("""using CUDA, CUDA.CUSPARSE""")

    def torch_to_data(self, quad_obj_values, lin_obj_values, con_values) -> "CUCLARABEL_data":
        batch_size, originally_unbatched = _detect_batch_size(con_values)

        # Add batch dimension for uniform handling if needed
        if originally_unbatched:
            con_values = con_values.unsqueeze(1)
            lin_obj_values = lin_obj_values.unsqueeze(1)
            quad_obj_values = quad_obj_values.unsqueeze(1) if quad_obj_values is not None else None

        # Build matrices
        Ps, qs, As, bs, b_idxs = _build_cuclarabel_matrices(
            con_values,
            quad_obj_values,
            lin_obj_values,
            self.P_idxs,
            self.P_structure,
            self.P_shape,
            self.A_idxs,
            self.A_structure,
            self.A_shape,
            self.b_idx,
            batch_size,
        )

        return CUCLARABEL_data(
            Ps=Ps,
            qs=qs,
            As=As,
            bs=bs,
            b_idxs=b_idxs,
            cones=self.dims,
            scs_cones=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )

    def jax_to_data(self, quad_obj_values, lin_obj_values, con_values) -> "CUCLARABEL_data":
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
            quad_obj_values = jnp.expand_dims(quad_obj_values, 1)

        # Build matrices
        Ps, qs, As, bs, b_idxs = _build_cuclarabel_matrices(
            con_values,
            quad_obj_values,
            lin_obj_values,
            self.P_idxs,
            self.P_structure,
            self.P_shape,
            self.A_idxs,
            self.A_structure,
            self.A_shape,
            self.b_idx,
            batch_size,
        )

        return CUCLARABEL_data(
            Ps=Ps,
            qs=qs,
            As=As,
            bs=bs,
            b_idxs=b_idxs,
            cones=self.dims,
            scs_cones=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )


def _compute_gradients(
    dprimal: TensorLike,
    ddual: TensorLike,
    Ps: list[cucsr_matrix],
    qs: list[cupy.ndarray],
    As: list[cucsr_matrix],
    bs: list[cupy.ndarray],
    scs_cones: dict[str, int | list[int]],
    xs: list[cupy.ndarray],
    ys: list[cupy.ndarray],
    ss: list[cupy.ndarray],
    b_idxs: list[cupy.ndarray],
    batch_size: int,
) -> tuple[list[cupy.ndarray], list[cupy.ndarray], list[cupy.ndarray]]:
    """Compute gradients using DIFFQCP's adjoint method.

    Uses implicit differentiation to compute gradients of the optimization
    solution with respect to problem parameters. The adjoint method efficiently
    computes these gradients by solving the adjoint system.

    Args:
        TODO

    Returns:
        Tuple of (dq_batch, dA_batch) where:
            - dq_batch: List of gradients w.r.t. linear objective coefficients
            - dA_batch: List of gradients w.r.t. constraint coefficients
    """
    # Convert incoming gradients to lists for DIFFQCP
    dxs = [np.array(dprimal[i]) for i in range(batch_size)]
    dys = [np.array(ddual[i]) for i in range(batch_size)]
    dss = [np.zeros_like(bs[i]) for i in range(batch_size)]  # No gradient w.r.t. slack

    dP_batch = []
    dq_batch = []
    dA_batch = []
    one_zero = cupy.array([0.0])
    # Call diffqcp
    for i in range(batch_size):
        structure = diffqcp.QCPStructureGPU(Ps[i], As[i], scs_cones)
        qcp = diffqcp.DeviceQCP(Ps[i], As[i], qs[i], bs[i], xs[i], ys[i], ss[i], structure)
        dP, dA, dq, db = qcp.vjp(dxs[i], dys[i], dss[i])

        con_grad = np.hstack([-dA.data, db[b_idxs[i]]])
        # Add zero gradient for constant offset term
        lin_grad = np.hstack([dq, one_zero])
        dA_batch.append(con_grad)
        dq_batch.append(lin_grad)
        dP_batch.append(dP.data)

    return dP_batch, dq_batch, dA_batch


@dataclass
class CUCLARABEL_data:
    Ps: list[cucsr_matrix]
    qs: list[cupy.ndarray]
    As: list[cucsr_matrix]
    bs: list[cupy.ndarray]
    b_idxs: list[cupy.ndarray]
    cones: list
    scs_cones: dict[str, int | list[int]]
    batch_size: int
    originally_unbatched: bool

    def torch_solve(self, solver_args=None):
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        if solver_args is None:
            solver_args = {}

        # Always use batch solve
        xs, ys, _, deriv_info = _call_cuclarabel(
            self.Ps,
            self.qs,
            self.As,
            self.bs,
            self.cones,
        )
        # Stack results into batched tensors
        primal = torch.stack([torch.tensor(x) for x in xs])
        dual = torch.stack([torch.tensor(y) for y in ys])
        return primal, dual, deriv_info

    def torch_derivative(self, primal, dual, deriv_info):
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        # Compute gradients
        dP_batch, dq_batch, dA_batch = _compute_gradients(
            primal,
            dual,
            deriv_info["P"],
            deriv_info["q"],
            deriv_info["A"],
            deriv_info["b"],
            self.scs_cones,
            deriv_info["x"],
            deriv_info["y"],
            deriv_info["s"],
            self.b_idxs,
            self.batch_size,
        )

        # Stack into shape (num_entries, batch_size)
        dP_stacked = torch.stack([torch.tensor(g) for g in dP_batch]).T
        dq_stacked = torch.stack([torch.tensor(g) for g in dq_batch]).T
        dA_stacked = torch.stack([torch.tensor(g) for g in dA_batch]).T

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dP_stacked = dP_stacked.squeeze(1)
            dq_stacked = dq_stacked.squeeze(1)
            dA_stacked = dA_stacked.squeeze(1)

        return (
            dP_stacked,
            dq_stacked,
            dA_stacked,
        )

    def jax_solve(self, solver_args=None):
        if solver_args is None:
            solver_args = {}

        # Always use batch solve
        xs, ys, _, deriv_info = _call_cuclarabel(
            self.Ps,
            self.qs,
            self.As,
            self.bs,
            self.cones,
        )

        # Stack results into batched arrays
        primal = jnp.stack([jnp.array(x) for x in xs])
        dual = jnp.stack([jnp.array(y) for y in ys])

        # Return primal, dual, and adjoint function for backward pass
        return primal, dual, deriv_info

    def jax_derivative(self, dprimal, ddual, deriv_info):
        # Compute gradients
        dP_batch, dq_batch, dA_batch = _compute_gradients(
            dprimal,
            ddual,
            deriv_info["P"],
            deriv_info["q"],
            deriv_info["A"],
            deriv_info["b"],
            self.scs_cones,
            deriv_info["x"],
            deriv_info["y"],
            deriv_info["s"],
            self.b_idxs,
            self.batch_size,
        )

        # Stack into shape (num_entries, batch_size)
        dP_stacked = jnp.stack([jnp.array(g) for g in dP_batch]).T
        dq_stacked = jnp.stack([jnp.array(g) for g in dq_batch]).T
        dA_stacked = jnp.stack([jnp.array(g) for g in dA_batch]).T

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dP_stacked = jnp.squeeze(dP_stacked, 1)
            dq_stacked = jnp.squeeze(dq_stacked, 1)
            dA_stacked = jnp.squeeze(dA_stacked, 1)

        return (
            dP_stacked,
            dq_stacked,
            dA_stacked,
        )
