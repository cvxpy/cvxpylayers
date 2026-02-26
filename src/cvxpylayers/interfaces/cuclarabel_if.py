from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cvxpy.reductions.solvers.conic_solvers.cuclarabel_conif import (
    dims_to_solver_cones as dims_to_cuclarabel_cones,
)
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (
    dims_to_solver_dict as scs_dims_to_solver_dict,
)

try:
    import jax

    # NOTE(quill): following will only work if this is the beginning of a JAX program.
    #   Or (obviously) 64 bit arithmetic has already been set.
    jax.config.update("jax_enable_x64", True)
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp
    from diffqcp import DeviceQCP, QCPStructureGPU
    from jaxlib._jax import Device
    from jaxtyping import Float, Integer  # jaxtyping is a `diffqcp` requirement
except ImportError:
    pass

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


if TYPE_CHECKING:
    import cupy
    import cvxpylayers.utils.parse_args as pa
    from cupyx.scipy.sparse import csr_matrix

    TensorLike = jax.Array | cupy.ndarray | csr_matrix


if torch is not None:
    class _CvxpyLayer(torch.autograd.Function):
        @staticmethod
        def forward(
            P_eval: torch.Tensor | None,
            q_eval: torch.Tensor,
            A_eval: torch.Tensor,
            cl_ctx: "pa.LayersContext",
            solver_args: dict[str, Any],
            needs_grad: bool = True,
            warm_start: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
            # Convert torch to jax and use jax_to_data path
            solver_ctx = cl_ctx.solver_ctx
            data = solver_ctx.jax_to_data(
                quad_obj_values=(
                    jax.dlpack.from_dlpack(P_eval.detach())
                    if P_eval is not None
                    else None
                ),
                lin_obj_values=jax.dlpack.from_dlpack(q_eval.detach()),
                con_values=jax.dlpack.from_dlpack(A_eval.detach()),
            )

            if solver_args is None:
                solver_args = {}

            xs, ys, vjps = _solve_gpu(
                data.data_matrices, qcp_struc=data.qcp_structure, julia_ctx=data.julia_ctx
            )

            primal = torch.stack([torch.from_dlpack(x) for x in xs])
            dual = torch.stack([torch.from_dlpack(y) for y in ys])

            return primal, dual, vjps, data

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
            _, _, vjps, data = outputs
            ctx.vjps = vjps
            ctx.data = data

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, dprimal: torch.Tensor, ddual: torch.Tensor, _vjps: Any, _data: Any
        ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None, None, None, None]:
            data = ctx.data

            dP_batch, dq_batch, dA_batch = _compute_gradients(
                dprimal=jax.dlpack.from_dlpack(dprimal.detach()),
                ddual=jax.dlpack.from_dlpack(ddual.detach()),
                b_idxs=data.b_idxs,
                vjps=ctx.vjps,
            )

            # Stack into shape (num_entries, batch_size)
            dP_stacked = torch.stack([torch.from_dlpack(g) for g in dP_batch]).T
            dq_stacked = torch.stack([torch.from_dlpack(g) for g in dq_batch]).T
            dA_stacked = torch.stack([torch.from_dlpack(g) for g in dA_batch]).T

            # Squeeze batch dimension only if input was originally unbatched
            if data.originally_unbatched:
                dP_stacked = dP_stacked.squeeze(1)
                dq_stacked = dq_stacked.squeeze(1)
                dA_stacked = dA_stacked.squeeze(1)

            return dP_stacked, dq_stacked, dA_stacked, None, None, None, None


@dataclass
class GpuDataMatrices:
    Pjxs: list[jsparse.BCSR]
    Pcps: list[csr_matrix]
    Ajxs: list[jsparse.BCSR]
    Acps: list[csr_matrix]
    qjxs: list[jax.Array]
    qcps: list[cupy.ndarray]
    bjxs: list[jax.Array]
    bcps: list[cupy.ndarray]


def _build_gpu_cqp_matrices(
    con_values: Float[jax.Array, "m n batch"],
    quad_obj_values: Float[jax.Array, " n batch"] | None,
    lin_obj_values: Float[jax.Array, "n batch"],
    P_structure: tuple[Integer[jax.Array, "_"], Integer[jax.Array, "_"]],
    P_shape: tuple[int, int],
    nnz_A: int,
    A_structure: tuple[Integer[jax.Array, "_"], Integer[jax.Array, "_"]],
    A_shape: tuple[int, int],
    b_idxs: Integer[jax.Array, "_"],
    batch_size: int,
) -> GpuDataMatrices:
    """Build conic quadratic program matrices from constraint and objective values.

    Converts parameter values into the conic form required by CUCLARABEL and (gpu) diffqcp.
        minimize 1/2 x^T P x + q^T x subject to Ax + s = b, s in K
    where K is a product of cones.

    Parametrization matrices have been pre-permuted so that matrix multiplication
    directly produces values in CSR order — no runtime shuffle indexing needed.

    TODO(quill): in future you can probably just store jax values as single
    BCSR array.

    Args:
        `con_values`: Constraint coefficient values (batched), already in CSR order
        `quad_obj_values`: Quadratic objective coefficient values (batched), already in CSR order
        `lin_obj_values`: Linear objective coefficient values (batched)
        `P_structure`: Sparse matrix structure (CSR layout) for objective: (indices, indptr)
        `P_shape`: Shape of objective matrix. Must be square
        `nnz_A`: Number of non-zeros in the A matrix
        `A_structure`: Sparse matrix structure (CSR layout) for constraint matrix-
            -**NOT augmented constraint matrix**: (indices, indptr).
        `A_shape`: Shape of the constraint matrix. **NOT the augmented constraint matrix**.
        `b_idxs`: Array of RHS indices. (These are the non-zero elements in the b in R^m
            constraint vector.)

    Returns:
        `GpuDataMatrices` object consisting of lists JAX sparse matrices and vectors and
        corresponding lists of CuPy reference objects. (i.e., zero-copied data.)
    """
    import cupy
    from cupyx.scipy.sparse import csr_matrix

    Pjxs, qjxs, Ajxs, bjxs = [], [], [], []
    Pcps, qcps, Acps, bcps = [], [], [], []

    for i in range(batch_size):
        quad_vals_i = quad_obj_values[:, i] if quad_obj_values is not None else None
        con_vals_i = con_values[:, i]
        lin_vals_i = lin_obj_values[:-1, i]

        if quad_vals_i is None:
            Pjx = jsparse.empty(shape=P_shape, sparse_format="bcsr")
            Pcp = csr_matrix(P_shape)
        else:
            Pjx = jsparse.BCSR((quad_vals_i, *P_structure), shape=P_shape)
            Pcp_data = cupy.from_dlpack(quad_vals_i)
            Pcp = csr_matrix(
                (
                    Pcp_data,
                    cupy.from_dlpack(P_structure[0]),
                    cupy.from_dlpack(P_structure[1]),
                ),
                shape=P_shape,
            )

        # Negate A to match CuClarabel / diffqcp convention.
        # con_vals_i[:nnz_A] are already in CSR order from pre-permuted matrices.
        Ajx_data = -con_vals_i[:nnz_A]
        Ajx = jsparse.BCSR((Ajx_data, *A_structure), shape=A_shape)
        Acp_data = cupy.from_dlpack(Ajx_data)
        Acp = csr_matrix(
            (Acp_data, cupy.from_dlpack(A_structure[0]), cupy.from_dlpack(A_structure[1])),
            shape=A_shape,
        )

        bjx = jnp.zeros(A_shape[0])
        bjx = bjx.at[b_idxs].set(con_vals_i[-jnp.size(b_idxs) :])
        bcp = cupy.from_dlpack(bjx)

        Pjxs.append(Pjx)
        Pcps.append(Pcp)
        qjxs.append(lin_vals_i)
        qcps.append(cupy.from_dlpack(lin_vals_i))
        Ajxs.append(Ajx)
        Acps.append(Acp)
        bjxs.append(bjx)
        bcps.append(bcp)

    return GpuDataMatrices(
        Pjxs=Pjxs, Pcps=Pcps, Ajxs=Ajxs, Acps=Acps, qjxs=qjxs, qcps=qcps, bjxs=bjxs, bcps=bcps
    )


class CUCLARABEL_ctx:
    P_structure: tuple[Integer[jax.Array, "_"], Integer[jax.Array, "_"]]
    P_shape: tuple[int, int]

    A_structure: tuple[Integer[jax.Array, "_"], Integer[jax.Array, "_"]]
    A_shape: tuple[int, int]
    nnz_A: int
    b_idxs: Integer[jax.Array, "_"]

    dims: dict
    options: dict | None
    diffqcp_problem_struc: QCPStructureGPU | None = None
    julia_ctx: Julia_CTX | None = None

    def __init__(
        self,
        csr,
        cone_dims,
        options=None,
    ):
        self.A_shape = csr.A_shape
        self.nnz_A = csr.nnz_A
        self.b_idxs = jnp.array(csr.b_idx)

        self.A_structure = jnp.array(csr.A_csr_structure[0]), jnp.array(csr.A_csr_structure[1])

        if csr.P_csr_structure is not None:
            self.P_shape = csr.P_shape
            self.P_structure = jnp.array(csr.P_csr_structure[0]), jnp.array(csr.P_csr_structure[1])
        else:
            self.P_shape = (csr.A_shape[1], csr.A_shape[1])
            self.P_structure = (jnp.array([], dtype=int), jnp.array([0] * (csr.A_shape[1] + 1)))

        self.options = options
        self.dims = cone_dims

    def jax_to_data(
        self,
        quad_obj_values: Float[jax.Array, "_ *batch"] | None,
        lin_obj_values: Float[jax.Array, "_ *batch"],
        con_values: Float[jax.Array, "_ *batch"],
    ) -> CUCLARABEL_data:
        if jnp.ndim(con_values) == 1:
            originally_unbatched = True
            batch_size = 1
            con_values = jnp.expand_dims(con_values, axis=1)
            quad_obj_values = (
                jnp.expand_dims(quad_obj_values, axis=1) if quad_obj_values is not None else None
            )
            lin_obj_values = jnp.expand_dims(lin_obj_values, axis=1)
        else:
            originally_unbatched = False
            batch_size = con_values.shape[1]

        device: Device = con_values.device

        if device.platform == "cpu":
            gpu_device = (
                self.options.pop("device", next(d for d in jax.devices() if d.platform == "gpu"))
                if self.options is not None
                else next(d for d in jax.devices() if d.platform == "gpu")
            )
            con_values = jax.device_put(con_values, device=gpu_device)
            quad_obj_values = (
                jax.device_put(quad_obj_values, device=gpu_device)
                if quad_obj_values is not None
                else None
            )
            lin_obj_values = jax.device_put(lin_obj_values, device=gpu_device)

        if self.julia_ctx is None:
            verbose = self.options.get("verbose", False) if self.options else False
            self.julia_ctx = Julia_CTX(self.dims, verbose=verbose)

        data_matrices = _build_gpu_cqp_matrices(
            con_values=con_values,
            quad_obj_values=quad_obj_values,
            lin_obj_values=lin_obj_values,
            P_structure=self.P_structure,
            P_shape=self.P_shape,
            nnz_A=self.nnz_A,
            A_structure=self.A_structure,
            A_shape=self.A_shape,
            b_idxs=self.b_idxs,
            batch_size=batch_size,
        )

        if self.diffqcp_problem_struc is None:
            self.diffqcp_problem_struc = QCPStructureGPU(
                data_matrices.Pjxs[0], data_matrices.Ajxs[0], scs_dims_to_solver_dict(self.dims)
            )

        return CUCLARABEL_data(
            data_matrices=data_matrices,
            qcp_structure=self.diffqcp_problem_struc,
            b_idxs=self.b_idxs,
            julia_ctx=self.julia_ctx,
            originally_unbatched=originally_unbatched,
        )



def _solve_gpu(
    data_matrices: GpuDataMatrices, qcp_struc: QCPStructureGPU, julia_ctx: Julia_CTX
) -> tuple[list[Float[jax.Array, " n"]], list[Float[jax.Array, " m"]], list[DeviceQCP]]:
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
        xcp, ycp, scp = julia_ctx.solve(P=Pcps[i], A=Acps[i], q=qcps[i], b=bcps[i])
        xjx = jax.dlpack.from_dlpack(xcp)
        yjx = jax.dlpack.from_dlpack(ycp)
        sjx = jax.dlpack.from_dlpack(scp)
        qcp = DeviceQCP(Pjxs[i], Ajxs[i], qjxs[i], bjxs[i], xjx, yjx, sjx, qcp_struc)
        xs.append(xjx)
        ys.append(yjx)
        vjps.append(qcp)

    return xs, ys, vjps


def _compute_gradients(
    dprimal: Float[jax.Array, "batch n"],
    ddual: Float[jax.Array, "batch m"],
    vjps: list[DeviceQCP],
    b_idxs: Integer[jax.Array, "..."],
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array]]:
    """Compute gradients using DIFFQCP's adjoint method.

    Uses implicit differentiation to compute gradients of the optimization
    solution with respect to problem parameters. The adjoint method efficiently
    computes these gradients by solving the adjoint system.

    Parametrization matrices have been pre-permuted so gradient data from
    DIFFQCP (in CSR order) is already in the correct order — no CSR→CSC
    permutation needed.

    Args:
        `dprimal`: Incoming gradients w.r.t. primal solution
        `ddual`: Incoming gradients w.r.t. dual solution
        `vjps`: A list of DIFFQCP's vector-Jacobian function.
        `b_idxs`: RHS indices from forward pass

    Returns:
        Tuple of (`dP_batch`, `dq_batch`, `dA_batch`) where
            - `dP_batch`: List of gradients w.r.t. quadratic objective coefficients
            - `dq_batch`: List of gradients w.r.t. linear objective coefficients
            - `dA_batch`: List of gradients w.r.t. constraint coefficients
    """

    dP_batch = []
    dq_batch = []
    dA_batch = []
    num_batches = jnp.shape(dprimal)[0]

    dslack = jnp.zeros_like(ddual[0])  # No gradient w.r.t. slack

    import equinox as eqx

    # NOTE(quill): doing the following to enforce only one comilation of QCP.vjp
    @eqx.filter_jit
    def _compute_vjp(qcp_module_instance: DeviceQCP, dx, dy, ds):
        return qcp_module_instance.vjp(dx, dy, ds)

    for i in range(num_batches):
        # TODO(quill): add ability to pass parameters to `vjp`
        dP, dA, dq, db = _compute_vjp(vjps[i], dprimal[i], ddual[i], dslack)

        con_grad = jnp.hstack([-dA.data, db[b_idxs]])
        lin_grad = jnp.hstack([dq, jnp.array([0.0])])
        dA_batch.append(con_grad)
        dq_batch.append(lin_grad)
        dP_batch.append(dP.data)

    return dP_batch, dq_batch, dA_batch


@dataclass
class CUCLARABEL_data:
    data_matrices: GpuDataMatrices
    qcp_structure: QCPStructureGPU  # QCPStructureLayers
    b_idxs: Integer[jax.Array, "..."]
    julia_ctx: Julia_CTX
    originally_unbatched: bool

    def jax_solve(self, solver_args=None):
        if solver_args is None:
            solver_args = {}

        xs, ys, vjps = _solve_gpu(
            self.data_matrices, qcp_struc=self.qcp_structure, julia_ctx=self.julia_ctx
        )

        primal = jnp.stack([x for x in xs])
        dual = jnp.stack([y for y in ys])

        return primal, dual, vjps

    def jax_derivative(
        self,
        dprimal: Float[jax.Array, "batch_size n"],
        ddual: Float[jax.Array, "batch_size m"],
        vjps: list[DeviceQCP],
    ):
        dP_batch, dq_batch, dA_batch = _compute_gradients(
            dprimal=dprimal,
            ddual=ddual,
            b_idxs=self.b_idxs,
            vjps=vjps,
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



class Julia_CTX:
    jl: Any
    was_solved_once: bool

    def __init__(self, dims: dict, verbose: bool = False):
        from juliacall import Main as jl

        self.jl = jl
        # self.jl.seval('import Pkg; Pkg.develop(url="https://github.com/oxfordcontrol/Clarabel.jl.git")')
        # self.jl.seval('import Pkg; Pkg.develop(url="https://github.com/PTNobel/Clarabel.jl.git")')
        # self.jl.seval('Pkg.add("CUDA")')
        self.jl.seval("using Clarabel, LinearAlgebra, SparseArrays")
        self.jl.seval("using CUDA, CUDA.CUSPARSE")

        dims_to_cuclarabel_cones(self.jl, dims)

        # Set verbose from Python
        self.jl.verbose_flag = verbose
        self.jl.seval("""
        settings = Clarabel.Settings(direct_solve_method = :cudss)
        settings.verbose = verbose_flag
        solver   = Clarabel.Solver(settings)
        """)

        self.was_solved_once = False

    def _solve_first_time(
        self,
        P: cupy.csr_matrix,
        A: cupy.csr_matrix,
        q: cupy.ndarray,
        b: cupy.ndarray,
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """Taken from `cvxpy`'s `cuclarabel_conif.py`"""
        nvars = q.size
        self.jl.q = self.jl.Clarabel.cupy_to_cuvector(self.jl.Float64, int(q.data.ptr), nvars)
        if P.nnz != 0:
            self.jl.P = self.jl.Clarabel.cupy_to_cucsrmat(
                self.jl.Float64,
                int(P.data.data.ptr),
                int(P.indices.data.ptr),
                int(P.indptr.data.ptr),
                *P.shape,
                P.nnz,
            )
        else:
            self.jl.seval(f"""
            P = CuSparseMatrixCSR(sparse(Float64[], Float64[], Float64[], {nvars}, {nvars}))
            """)

        self.jl.A = self.jl.Clarabel.cupy_to_cucsrmat(
            self.jl.Float64,
            int(A.data.data.ptr),
            int(A.indices.data.ptr),
            int(A.indptr.data.ptr),
            *A.shape,
            A.nnz,
        )
        self.jl.b = self.jl.Clarabel.cupy_to_cuvector(self.jl.Float64, int(b.data.ptr), b.size)

        self.jl.seval("""solver = Clarabel.setup!(solver, P,q,A,b,cones)""")
        self.jl.Clarabel.solve_b(self.jl.solver)

        x = JuliaCuVector2CuPyArray(self.jl, self.jl.solver.solution.x)
        y = JuliaCuVector2CuPyArray(self.jl, self.jl.solver.solution.z)
        s = JuliaCuVector2CuPyArray(self.jl, self.jl.solver.solution.s)

        return x, y, s

    def solve(
        self, P: cupy.csr_matrix, A: cupy.csr_matrix, q: cupy.ndarray, b: cupy.ndarray
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
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
    """Taken from https://github.com/cvxgrp/CuClarabel/blob/main/src/python/jl2py.py."""
    import cupy

    # Get the device pointer from Julia
    pDevice = jl.Int(jl.pointer(jl_arr))

    # Get array length and element type
    span = jl.size(jl_arr)
    dtype = jl.eltype(jl_arr)

    # Map Julia type to CuPy dtype
    if dtype == jl.Float64:
        dtype = cupy.float64
    else:
        dtype = cupy.float32

    # Compute memory size in bytes (assuming 1D vector)
    size_bytes = int(span[0] * cupy.dtype(dtype).itemsize)

    # Create CuPy memory view from the Julia pointer
    mem = cupy.cuda.UnownedMemory(pDevice, size_bytes, owner=None)
    memptr = cupy.cuda.MemoryPointer(mem, 0)

    # Wrap into CuPy ndarray
    arr = cupy.ndarray(shape=span, dtype=dtype, memptr=memptr)
    return arr
