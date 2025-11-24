from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import dims_to_solver_cones
from cvxpy.reductions.solvers.conic_solvers.cuclarabel_conif import dims_to_solver_cones as dims_to_cusolver_cones

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jaxlib._jax import Device
    import jax.experimental.sparse as jsparse
    # from diffqcp import QCPStructureLayers, DeviceQCP, HostQCP
    from diffqcp import DeviceQCP, QCPStructureGPU
except ImportError:
    pass


if TYPE_CHECKING:
    import torch
    import cupy as cp
    from cupy.sparse import csr_matrix
    from jaxtyping import Float, Integer
    TensorLike = jax.Array | cp.ndarray | csr_matrix


@dataclass
class GpuDataMatrices:
    Pjxs: list[jsparse.BCSR]
    Pcps: list[csr_matrix]
    Ajxs: list[jsparse.BCSR]
    Acps: list[csr_matrix]
    qjxs: list[jax.Array]
    qcps: list[cp.ndarray]
    bjxs: list[jax.Array]
    bcps: list[cp.ndarray]


def _build_gpu_cqp_matrices(
    con_values: Float[jax.Array, "m n batch"],
    quad_obj_values: Float[jax.Array, " n batch"] | None,
    lin_obj_values: Float[jax.Array, "n batch"],
    obj_csc_to_csr_permutation: jnp.ndarray,
    obj_structure: tuple[jnp.ndarray, jnp.ndarray],
    obj_shape: tuple[int, int],
    con_csc_to_csr_subset_and_permutation: jnp.ndarray,
    con_structure: tuple[jnp.ndarray, jnp.ndarray],
    con_shape: tuple[int, int],
    b_idxs: jnp.ndarray,
    batch_size: int
) -> GpuDataMatrices:
    """Build conic quadratic program matrices from constraint and objective values.

    Converts parameter values into the conic form required by CUCLARABEL and (gpu) diffqcp.
        minimize 1/2 x^T P x + q^T x subject to Ax + s = b, s in K
    where K is a product of cones.

    TODO(quill): in future you can probably just store jax values as single
    BCSR array.
    """
    import cupy as cp
    from cupy.sparse import csr_matrix

    Pjxs, qjxs, Ajxs, bjxs = [], [], [], []
    Pcps, qcps, Acps, bcps = [], [], [], []

    quad_obj_values = quad_obj_values[obj_csc_to_csr_permutation, :]
    con_values = con_values[con_csc_to_csr_subset_and_permutation, :]

    for i in range(batch_size):
        quad_vals_i = (quad_obj_values[:, i]
                       if quad_obj_values is not None else None)
        con_vals_i = con_values[:, i]
        lin_vals_i = lin_obj_values[:-1, i]

        if quad_vals_i is None:
            Pjx = jsparse.empty(shape=obj_shape, sparse_format="bcsr")
            Pcp = csr_matrix(obj_shape)
        else:
            Pjx = jsparse.BCSR((
                quad_vals_i,
                *obj_structure
            ), shape=obj_shape)
            Pcp_data = cp.from_dlpack(quad_vals_i)
            Pcp = csr_matrix((
                Pcp_data,
                cp.from_dlpack(obj_structure[0]),
                cp.from_dlpack(obj_structure[1]),
            ), shape=obj_shape)
        
        Ajx = jsparse.BCSR((
            con_vals_i[con_csc_to_csr_subset_and_permutation],
            *con_structure
        ), shape=con_shape)
        Acp_data = cp.from_dlpack(Ajx.data)
        Acp = csr_matrix((
            Acp_data,
            cp.from_dlpack(con_structure[0]),
            cp.from_dlpack(con_structure[1])
        ), shape=con_shape)

        bjx = jnp.zeros(con_shape[0])
        bjx = bjx.at[b_idxs].set(con_vals_i[-jnp.size(b_idxs):])
        bcp = cp.from_dlpack(bjx)

        Pjxs.append(Pjx)
        Pcps.append(Pcp)
        qjxs.append(lin_vals_i)
        qcps.append(cp.from_dlpack(lin_vals_i))
        Ajxs.append(Ajx)
        Acps.append(Acp)
        bjxs.append(bjx)
        bcps.append(bcp)

    return GpuDataMatrices(
        Pjxs=Pjxs, Pcps=Pcps, Ajxs=Ajxs, Acps=Acps,
        qjxs=qjxs, qcps=qcps, bjxs=bjxs, bcps=bcps
    )


class DIFFQCP_CTX:

    csr_objective_structure: tuple[jnp.ndarray, jnp.ndarray]
    coo_objective_structure: tuple[jnp.ndarray, jnp.ndarray]
    obj_csc_to_csr_permutation: jnp.ndarray
    obj_csr_to_csc_permutation: jnp.ndarray
    P_shape: tuple[int, int]

    last_col_start: int
    last_col_end: int
    last_col_indices: jnp.ndarray
    csr_con_structure: tuple[jnp.ndarray, jnp.ndarray]
    con_csc_to_csr_subset_and_permutation: jnp.ndarray
    con_csr_to_csc_subset_and_permutation: jnp.ndarray
    A_shape: tuple[int, int]

    dims: dict
    diffqcp_problem_struc: QCPStructureGPU | None = None
    julia_ctx: Julia_CTX | None = None

    def __init__(
        self,
        objective_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        constraint_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        data: dict,
        options: dict | None = None
    ):
        obj_indices, obj_ptr, (n, _) = objective_structure
        
        obj_csr = sp.csc_array(
            (np.arange(obj_indices.size), obj_indices, obj_ptr),
            shape=(n,n),
        ).tocsr()
        self.csr_objective_structure = jnp.array(obj_csr.indices), jnp.array(obj_csr.indptr)
        self.obj_csc_to_csr_permutation = jnp.array(obj_csr.data)
        self.obj_csr_to_csc_permutation = jnp.empty_like(self.obj_csc_to_csr_permutation)
        self.obj_csr_to_csc_permutation[self.obj_csc_to_csr_permutation] = jnp.arange(jnp.size(self.obj_csc_to_csr_permutation))

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1
        self.A_shape = (m, n)

        self.b_idxs = jnp.ndarray(con_indices[con_ptr[-2] : con_ptr[-1]])

        # Now construct the structure for just the A matrix as expected by `diffqcp`
        con_csr = sp.csc_array(
            (np.arange(con_indices.size), con_indices, con_ptr[:-1]),
            shape=(m, n),
        ).tocsr()
        self.csr_con_structure = jnp.array(con_csr.indices), jnp.array(con_csr.indptr)
        self.con_csc_to_csr_subset_and_permutation = jnp.array(con_csr.data)
        self.con_csr_to_csc_subset_and_permutation = jnp.empty_like(self.con_csc_to_csr_subset_and_permutation)
        self.con_csr_to_csc_subset_and_permutation[self.con_csc_to_csr_subset_and_permutation] = jnp.arange(
            jnp.size(self.con_csc_to_csr_subset_and_permutation)
        )

        self.dims = data["dims"]
        # TODO(quill): for future
        # self.diffqcp_problem_struc = QCPStructureLayers(
        #     data,
        #     *self.csr_objective_structure,
        #     *self.csr_con_structure
        # )

    def jax_to_data(
        self,
        quad_obj_values: Float[jax.Array, "_ *batch"] | None,
        lin_obj_values: Float[jax.Array, "_ *batch"],
        con_values: Float[jax.Array, "_ *batch"]
    ) -> DIFFQCP_data:
        
        if jnp.ndim(con_values) == 1:
            originally_unbatched = True
            batch_size = 1
            con_values = jnp.expand_dims(con_values, axis=1)
            quad_obj_values = jnp.expand_dims(quad_obj_values, axis=1) if quad_obj_values is not None else None
            lin_obj_values = jnp.expand_dims(lin_obj_values, axis=1)
        else:
            originally_unbatched = False
            batch_size = con_values.shape[1]
        
        device: Device = quad_obj_values.device
        
        if device.platform == "cpu":
            raise ValueError("diffqcp currently can only run on GPU-enabled workflows.")
        elif device.platform == "gpu":
            if self.julia_ctx is None:
                self.julia_ctx = Julia_CTX(self.dims)

            data_matrices = _build_gpu_cqp_matrices(
                con_values, quad_obj_values, lin_obj_values, self.obj_csc_to_csr_permutation,
                self.csr_objective_structure, self.P_shape, self.con_csc_to_csr_subset_and_permutation,
                self.csr_con_structure, self.A_shape, self.last_col_indices, batch_size
            )

            if self.diffqcp_problem_struc is None:
                self.diffqcp_problem_struc = QCPStructureGPU(
                    data_matrices.Pjxs[0], data_matrices.Ajxs[0], self.dims
                )

            return DIFFQCP_gpu_data(
                data_matrices=data_matrices,
                qcp_structure=self.diffqcp_problem_struc,
                P_csr_csc_perm=self.obj_csr_to_csc_permutation,
                A_csr_csc_perm=self.con_csr_to_csc_subset_and_permutation,
                b_idxs = self.b_idxs,
                julia_ctx=self.julia_ctx,
                originally_unbatched=originally_unbatched
            )

        else:
            raise ValueError("CVXPYlayers does not currently support operations "
                             f"on {device.platform}s.")
    
    def torch_to_data(
        self,
        quad_obj_values: Float[torch.Tensor, "_ *batch"] | None,
        lin_obj_values: Float[torch.Tensor, "_ *batch"],
        con_values: Float[torch.Tensor, "_ *batch"]
    ) -> DIFFQCP_data:

        return self.jax_to_data(
            quad_obj_values=(jax.dlpack.from_dlpack(quad_obj_values) if quad_obj_values is not None
                             else None),
            lin_obj_values=jax.dlpack.from_dlpack(lin_obj_values),
            con_values=jax.dlpack.from_dlpack(con_values)
        )


def _solve_gpu(
    data_matrices: GpuDataMatrices,
    qcp_struc,
    julia_ctx: Julia_CTX
) -> tuple[list[Float[jax.Array, " n"]], list[Float[jax.Array, " m"]], list[Callable]]:

    Pjxs = data_matrices.Pjxs
    Pcps = data_matrices.Pcps
    Ajxs = data_matrices.Ajxs
    Acps = data_matrices.Acps
    qjxs = data_matrices.qjxs
    qcps = data_matrices.qcps
    bjxs = data_matrices.bjxs
    bcps = data_matrices.bcps

    xs, ys, vjps = [], [], []
    
    for i in range(len(Pjxs)):
        # NOTE(quill): in this case I totally could do in place
        #   updates after the firt Julia solve.
        #   Unless we want to use CUDA streams
        xcp, ycp, scp = julia_ctx.solve(
            P=Pcps[i], A=Acps[i], q=qcps[i], b=bcps[i]
        )
        xjx = jax.dlpack.from_dlpack(xcp)
        yjx = jax.dlpack.from_dlpack(ycp)
        sjx = jax.dlpack.from_dlpack(scp)
        qcp_struc = QCPStructureGPU(Pjxs[i], Ajxs[i], )
        qcp = DeviceQCP(
            Pjxs[i], Ajxs[i], qjxs[i], bjxs[i],
            xjx, yjx, sjx, qcp_struc
        )
        xs.append(xjx)
        ys.append(yjx)
        vjps.append(qcp.vjp)

    return xs, ys, vjps


def _compute_gradients(
    dprimal: Float[jax.Array, "batch n"],
    ddual: Float[jax.Array, "batch m"],
    P_csr_csc_perm: Integer[jax.Array, "..."],
    A_csr_csc_perm: Integer[jax.Array, "..."],
    b_idxs: Integer[jax.Array, "..."],
    vjps: list[Callable],
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array]]:
    
    dP_batch = []
    dq_batch = []
    dA_batch = []
    num_batches = jnp.shape(dprimal)[0]

    ds = jnp.zeros_like(ddual[0]) # No gradient w.r.t. slack

    for i in range(num_batches):
        # TODO(quill): add ability to pass parameers to `vjp`
        dP, dA, dq, db = vjps[i](dprimal[i], ddual[i], ds)
        
        con_grad = jnp.hstack([-dA.data[A_csr_csc_perm], db[b_idxs]])
        lin_grad = jnp.hstack([dq, jnp.array([0.0])])
        dA_batch.append(con_grad)
        dq_batch.append(lin_grad)
        dP_batch.append(dP.data[P_csr_csc_perm])

    return dP_batch, dq_batch, dA_batch


@dataclass
class DIFFQCP_cpu_data:
    
    batch_size: int
    originally_unbatched: bool

    def _solve(self, solver_args=None):
        raise NotImplementedError


@dataclass
class DIFFQCP_gpu_data:
    
    data_matrices: GpuDataMatrices
    qcp_structure: QCPStructureGPU# QCPStructureLayers
    P_csr_csc_perm: Integer[jax.Array, "..."]
    A_csr_csc_perm: Integer[jax.Array, "..."]
    b_idxs: Integer[jax.Array, "..."]
    julia_ctx: Julia_CTX
    originally_unbatched: bool

    def jax_solve(self, solver_args=None):
        
        if solver_args is None:
            solver_args = {}

        xs, ys, vjps = _solve_gpu(
            self.data_matrices,
            qcp_struc=self.qcp_structure,
            julia_ctx=self.julia_ctx
        )

        primal = jnp.stack([x for x in xs])
        dual = jnp.stack([y for y in ys])

        return primal, dual, vjps

    def jax_derivative(
        self,
        dprimal: Float[jax.Array, "batch_size n"],
        ddual: Float[jax.Array, "batch_size m"],
        vjps: list[Callable]
    ):
        dP_batch, dq_batch, dA_batch = _compute_gradients(
            dprimal=dprimal,
            ddual=ddual,
            P_csr_csc_perm=self.P_csr_csc_perm,
            A_csr_csc_perm=self.A_csr_csc_perm,
            b_idxs=self.b_idxs,
            vjps=vjps
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

    def torch_solve(self, solver_args=None):
        
        if solver_args is None:
            solver_args = {}

        xs, ys, vjps = _solve_gpu(
            self.data_matrices,
            qcp_struc=self.qcp_structure,
            julia_ctx=self.julia_ctx
        )

        primal = torch.stack([torch.from_dlpack(x) for x in xs])
        dual = torch.stack([torch.from_dlpack(y) for y in ys])
        return primal, dual, vjps

    def torch_derivative(
        self,
        dprimal: Float[torch.Tensor, "batch_size n"],
        ddual: Float[torch.Tensor, "batch_size m"],
        vjps: list[Callable]
    ):
        dP_batch, dq_batch, dA_batch = _compute_gradients(
            dprimal=jax.dlpack.from_dlpack(dprimal),
            ddual=jax.dlpack.from_dlpack(ddual),
            P_csr_csc_perm=self.P_csr_csc_perm,
            A_csr_csc_perm=self.A_csr_csc_perm,
            b_idxs=self.b_idxs,
            vjps=vjps
        )

        # Stack into shape (num_entries, batch_size)
        dP_stacked = torch.stack([torch.from_dlpack(g) for g in dP_batch]).T
        dq_stacked = torch.stack([torch.from_dlpack(g) for g in dq_batch]).T
        dA_stacked = torch.stack([torch.from_dlpack(g) for g in dA_batch]).T

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


type DIFFQCP_data = DIFFQCP_cpu_data | DIFFQCP_gpu_data
        

class Julia_CTX:
    
    jl: Any
    was_solved_once: bool

    def __init__(
        self,
        dims: dict
    ):
        from juliacall import Main as jl
        self.jl = jl
        self.jl.seval("using Clarabel, LinearAlgebra, SparseArrays")
        self.jl.seval("using CUDA, CUDA.CUSPARSE")
        
        dims_to_cusolver_cones(self.jl, dims)

        self.jl.seval("""
        settings = Clarabel.Settings(direct_solve_method = :cudss)
        settings.verbose = False
        solver   = Clarabel.Solver(settings)
        """)

        self.was_solved_once = False

    def _solve_first_time(
        self,
        P: cp.csr_matrix,
        A: cp.csr_matrix,
        q: cp.ndarray,
        b: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Taken from `cvxpy`'s `clarabel_conif.py`"""
        nvars = q.size
        if P.nnz != 0:
            self.jl.P = self.jl.Clarabel.cupy_to_cucsrmat(
                self.jl.Float64, int(P.data.data.ptr), int(P.indices.data.ptr),
                int(P.indptr.data.ptr), *P.shape, P.nnz
            )
        else:
            self.jl.seval(f"""
            P = CuSparseMatrixCSR(sparse(Float64[], Float64[], Float64[], {nvars}, {nvars}))
            """)
        self.jl.q = self.jl.Clarabel.cupy_to_cuvector(self.jl.Float64, int(q.data.ptr), nvars)

        self.jl.A = self.jl.Clarabel.cupy_to_cucsrmat(
            self.jl.Float64, int(A.data.data.ptr), int(A.indices.data.ptr),
            int(A.indptr.data.ptr), *A.shape, A.nnz
        )
        self.b = self.jl.Clarabel.cupy_to_cuvector(self.jl.Float64, int(b.data.ptr), b.size)

        self.jl.seval("""solver = Clarabel.setup!(solver, P,q,A,b,cones)""")
        self.jl.Clarabel.solve_b(self.jl.solver)

        x = JuliaCuVector2CuPyArray(self.jl, self.jl.solver.solution.x)
        y = JuliaCuVector2CuPyArray(self.jl, self.jl.solver.solution.z)
        s = JuliaCuVector2CuPyArray(self.jl, self.jl.solver.solution.s)
        
        return x, y, s

    def _solve_np1_time(
        self,
        P: cp.csr_matrix,
        A: cp.csr_matrix,
        q: cp.ndarray,
        b: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        pass
    
    def solve(
        self,
        P: cp.csr_matrix,
        A: cp.csr_matrix,
        q: cp.ndarray,
        b: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        
        # TODO(quill): determine the feasibility of the following.
        # - We'd probably have to cache the data matrices
        #   (note how in diffqcp gpu experiment you don't update the julia data itself,
        #   but instead just increment the data it shares with CuPy array.)
        # - Also remember that when you weren't careful and let CuPy arrays out of scope,
        #   Julia / CUDA ended up yelling.

        # if not self.was_solved_once:
        #     self.was_solved_once = True
        #     return self._solve_first_time(P, A, q, b)
        # else:
        #     return self._solve_np1_time(P, A, q, b)

        return self._solve_first_time(P, A, q, b)
        

def JuliaCuVector2CuPyArray(jl, jl_arr):
    """Taken from https://github.com/cvxgrp/CuClarabel/blob/main/src/python/jl2py.py.
    """
    import cupy as cp
    # Get the device pointer from Julia
    pDevice = jl.Int(jl.pointer(jl_arr))

    # Get array length and element type
    span = jl.size(jl_arr)
    dtype = jl.eltype(jl_arr)

    # Map Julia type to CuPy dtype
    if dtype == jl.Float64:
        dtype = cp.float64
    else:
        dtype = cp.float32

    # Compute memory size in bytes (assuming 1D vector)
    size_bytes = int(span[0] * cp.dtype(dtype).itemsize)

    # Create CuPy memory view from the Julia pointer
    mem = cp.cuda.UnownedMemory(pDevice, size_bytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)

    # Wrap into CuPy ndarray
    arr = cp.ndarray(shape=span, dtype=dtype, memptr=memptr)
    return arr