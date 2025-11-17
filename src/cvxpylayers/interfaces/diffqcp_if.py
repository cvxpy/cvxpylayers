from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    import jax.experimental.sparse as jsp
    from diffqcp import QCPStructureLayers, DeviceQCP, HostQCP
except ImportError:
    pass


if TYPE_CHECKING:
    import torch
    from jaxtyping import Float


class DIFFQCP_CTX:

    csc_objective_structure: tuple[np.ndarray, np.ndarray]
    csr_objective_structure: tuple[np.ndarray, np.ndarray]
    coo_objective_structure: tuple[np.ndarray, np.ndarray]
    objective_csc_to_csr_permutation: np.ndarray

    dims: dict
    diffqcp_problem_structure: QCPStructureLayers
    julia_ctx: Julia_CTX | None = None

    def __init__(
        self,
        objective_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        constraint_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        data: dict,
        options: dict | None = None
    ):
        """
        We'll setup the data for both 
        """
        obj_indices, obj_ptr, (n, _) = objective_structure
        
        obj_csr = sp.csc_array(
            (np.arange(obj_indices.size), obj_indices, obj_ptr),
            shape=(n,n),
        ).tocsr()
        self.csr_objective_structure = obj_csr.indices, obj_csr.indptr
        self.csc_to_csr_permutation = obj_csr.data

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1

        self.dims = data["dims"]

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
            quad_obj_values = jnp.expand_dims(quad_obj_values, axis=1)
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
        # seems like this is where we can provide info such as type of
        # least squares solver for `diffqcp`

        # So we need to 

        pass


@dataclass
class DIFFQCP_gpu_data:
    
    diffqcp_ctx: DIFFQCP_CTX # Reference to context with structure info
    quad_obj_values: Float[jax.Array, "_ batch"] | None
    lin_obj_values: Float[jax.Array, "_ batch"]
    con_values: Float[jax.Array, "_ batch"]
    batch_size: int
    originally_unbatched: bool

    def _solve(self, solver_args=None):
        pass


type DIFFQCP_data = DIFFQCP_cpu_data | DIFFQCP_gpu_data
        

class Julia_CTX:
    julia_caller: Any

    def __init__(
        self,
        dims: dict
    ):
        from juliacall import Main as jl
        self.julia_caller = jl
        self.julia_caller.seval("using Clarabel, LinearAlgebra, SparseArrays")
        self.julia_caller.seval("using CUDA, CUDA.CUSPARSE")
        
        dims_to_cusolver_cones(self.julia_caller, dims)

        self.julia_caller.seval("""
        settings = Clarabel.Settings(direct_solve_method = :cudss)
        settings.verbose = False
        solver   = Clarabel.Solver(settings)
        """)