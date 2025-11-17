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
    from diffqcp import QCPStructureLayers, DeviceQCP, HostQCP
except ImportError:
    pass


if TYPE_CHECKING:
    import torch
    import cupy as cp
    from jaxtyping import Float


class DIFFQCP_CTX:

    n: int
    csc_objective_structure: tuple[jnp.ndarray, jnp.ndarray]
    csr_objective_structure: tuple[jnp.ndarray, jnp.ndarray]
    coo_objective_structure: tuple[jnp.ndarray, jnp.ndarray]
    obj_csc_to_csr_permutation: jnp.ndarray

    m: int
    last_col_start: int
    last_col_end: int
    last_col_indices: jnp.ndarray
    csc_con_structure: tuple[jnp.ndarray, jnp.ndarray]
    csr_con_structure: tuple[jnp.ndarray, jnp.ndarray]
    con_csc_to_csr_subset_and_permutation: jnp.ndarray

    dims: dict
    diffqcp_problem_struc: QCPStructureLayers
    julia_ctx: Julia_CTX | None = None

    def __init__(
        self,
        objective_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        constraint_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        data: dict,
        options: dict | None = None
    ):
        obj_indices, obj_ptr, (n, _) = objective_structure
        
        self.csc_objective_structure = jnp.array(obj_indices), jnp.array(obj_ptr)
        obj_csr = sp.csc_array(
            (np.arange(obj_indices.size), obj_indices, obj_ptr),
            shape=(n,n),
        ).tocsr()
        self.n = n
        self.csr_objective_structure = jnp.array(obj_csr.indices), jnp.array(obj_csr.indptr)
        self.obj_csc_to_csr_permutation = jnp.array(obj_csr.data)

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1

        # NOTE(quill): stole the following from `mpax_if.py`
        self.last_col_start = con_ptr[-2]
        self.last_col_end = con_ptr[-1]
        self.last_col_indices = jnp.ndarray(con_indices[self.last_col_start : self.last_col_end])
        self.m = m

        # Now construct the structure for just the A matrix as expected by `diffqcp`
        con_csr = sp.csc_array(
            (np.arange(con_indices.size), con_indices, con_ptr[:-1]),
            shape=(m, n),
        ).tocsr()
        self.csr_con_structure = jnp.array(con_csr.indices), jnp.array(con_csr.indptr)
        self.con_csc_to_csr_subset_and_permutation = jnp.array(con_csr.data)

        self.dims = data["dims"]
        self.diffqcp_problem_struc = QCPStructureLayers(
            data,
            *self.csr_objective_structure,
            *self.csr_con_structure
        )

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
            raise ValueError("The diffqcp backend for CVXPYlayers does not "
                             "currently support batched problems.")
            # originally_unbatched = False
            # batch_size = con_values.shape[1]
        
        device: Device = quad_obj_values.device
        
        if device.platform == "cpu":
            pass
        elif device.platform == "gpu":
            if self.julia_ctx is None:
                self.julia_ctx = Julia_CTX(self.dims)
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
    

@dataclass
class DIFFQCP_cpu_data:
    
    batch_size: int
    originally_unbatched: bool

    def _solve(self, solver_args=None):
        raise NotImplementedError


@dataclass
class DIFFQCP_gpu_data:
    
    ctx: DIFFQCP_CTX # Reference to context with structure info
    quad_obj_values: Float[jax.Array, "_ batch"] | None
    lin_obj_values: Float[jax.Array, "_ batch"]
    con_values: Float[jax.Array, "_ batch"]
    batch_size: int
    originally_unbatched: bool

    def _solve(self, solver_args=None):
        # NOTE(quill) skip batching for now; try with CUDA streams when adding batch support
        import cupy as cp
        from cupy.sparse import csr_matrix
        
        P_shape = (self.ctx.n, self.ctx.n)
        if self.quad_obj_values is None:
            Pjx = jsparse.empty(shape=P_shape, sparse_format="bcsr")
            Pcp = csr_matrix(P_shape)
        else:
            Pjx_data = self.quad_obj_values[self.ctx.obj_csc_to_csr_permutation, 0]
            Pjx = jsparse.BCSR((
                Pjx_data,
                self.ctx.csr_objective_structure[0],
                self.ctx.csr_objective_structure[1]
            ), shape=P_shape)
            Pcp_data = cp.from_dlpack(Pjx_data)
            Pcp = csr_matrix((
                Pcp_data,
                cp.from_dlpack(self.ctx.csr_objective_structure[0]),
                cp.from_dlpack(self.ctx.csr_objective_structure[1])
            ), shape=P_shape)
        # use c for linear part of objective since `qcp` is used in context of `diffqcp` module
        cjx = self.lin_obj_values[:-1, 0]
        ccp = cp.from_dlpack(cjx)
        
        b_values = self.con_values[self.ctx.last_col_start : self.ctx.last_col_end, 0]
        bjx = jnp.zeros(self.ctx.m)
        bjx = bjx.at[self.ctx.last_col_indices].set(b_values)
        bcp = cp.from_dlpack(bjx)
        
        A_shape = (self.ctx.m, self.ctx.n)
        # Negate A to match DIFFQCP/DIFFCP convention.
        Ajx_data = self.con_values[self.ctx.con_csc_to_csr_subset_and_permutation, 0]
        Ajx = jsparse.BCSR((
            Ajx_data,
            self.ctx.csr_con_structure[0],
            self.ctx.csr_con_structure[1]
        ), shape=A_shape)
        Acp_data = cp.from_dlpack(Ajx_data)
        Acp = csr_matrix((
            Acp_data,
            cp.from_dlpack(self.ctx.csr_con_structure[0]),
            cp.from_dlpack(self.ctx.csr_con_structure[1])
        ), shape=A_shape)

        xcp, ycp, scp = self.ctx.julia_ctx.solve(Pcp, Acp, ccp, bcp)
        x = jax.dlpack.from_dlpack(xcp)
        y = jax.dlpack.from_dlpack(ycp)
        s = jax.dlpack.from_dlpack(scp)

        qcp = DeviceQCP(Pjx, Ajx, cjx, bjx, x, y, s, self.ctx.diffqcp_problem_struc)

        # TODO(quill): put back in batch form
        # TODO(quill): determine where you want to JIT
        #   (remember you cannot JIT whole `vjp` if using nvmath for solve)
        return x, y, qcp.vjp
    
    def _derivative(
        self,
        primal: Float[jax.Array, "n batch"],
        dual: Float[jax.Array, "m batch"],
        adj_batch: Callable
    ):
        dx = primal[0]
        dy = dual[0]
        ds = jnp.zeros(self.ctx.m)

        dP, dA, dq, db = adj_batch(dx, dy, ds)


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