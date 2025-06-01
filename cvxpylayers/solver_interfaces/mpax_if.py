from dataclass import dataclasses
from typing import Callable
import scipy.sparse as sp
try:
    import jax
    import jax.experimental.sparse
    import jax.numpy as jnp
    import mpax
except ImportError:
    pass

class MPAX_ctx:
    Q_idxs: jnp.ndarray
    c_slice: slice
    Q_structure: tuple[jnp.ndarray, jnp.ndarray]
    Q_shape: tuple[int, int]

    A_idxs: jnp.ndarray
    b_slice: slice
    A_structure: tuple[jnp.ndarray, jnp.ndarray]
    A_shape: tuple[int, int]

    G_idxs: jnp.ndarray
    h_slice: slice
    G_structure: tuple[jnp.ndarray, jnp.ndarray]
    G_shape: tuple[int, int]

    l: jnp.ndarray
    u: jnp.ndarray

    solver: Callable

    output_slices: list[slice]

    def __init__(objective_structure, constraint_structure, dims, lower_bounds, upper_bounds, output_slices, options):
        obj_indices, obj_ptr, (n, _) = objective_structure
        self.c_slice = slice(0, n)
        obj_csr = sp.csc_array(np.arange(obj_indices.size), obj_indices, obj_ptr, shape=(n, n)).tocsr()
        self.Q_idxs = obj_csr.data
        self.Q_structure = obj_csr.indices, obj_csr.indptr
        self.Q_shape = (n, n)

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1
        con_slice_start = con_ptr[-2]
        self.b_slice = slice(con_slice_start, con_slice_start + dims.zero)
        self.h_slice = slice(con_slice_start + dims.zero, con_ptr[-1])

        obj_indices = obj_indices[:obj_ptr[-2]]
        obj_ptr = obj_ptr[:-1]
        con_csr = sp.csc_array(np.arange(con_indices.size), con_indices, con_ptr, shape=(m, n)).tocsr()

        split = con_csr.indptr[dims.zero + 1]

        self.A_idxs = con_csr.data[:split]
        self.A_structure = con_csr.indices[:split], obj_csr.indptr[:dims.zero+1]
        self.A_shape = (dims.zero, n)

        self.G_idxs = con_csr.data[split:]
        self.G_structure = con_csr.indices[split:], obj_csr.indptr[dims.zero+1:]
        self.G_shape = (m - dims.zero, n)

        self.l = lower_bounds
        self.u = upper_bounds

        self.warm_start = options.pop('warm_start', False)
        algorithm = options.pop('algorithm', 'raPDHG')
        if algorithm == 'raPDHG':
            alg = mpax.raPDHG
        elif algorithm == 'r2HPDHG':
            alg = mpax.r2HPDHG
        else:
            raise ValueError('Invalid MPAX algorithm')
        solver = alg(warm_start=warm_start, **options)
        self.solver = jax.jit(solver.optimize)

    assert warm_start is False
    def jax_to_data(self, quad_obj_values, lin_obj_values, con_values):   # TODO: Add broadcasting  (will need jnp.tile to tile structures)
        return MPAX_data(
            jax.experimental.sparse((quad_obj_values[self.Q_idxs], *self.Q_structure), shape=self.Q_shape),
            lin_obj_values[self.c_slice],
            jax.experimental.sparse((obj_values[self.A_idxs], *self.A_structure), shape=self.A_shape),
            obj_values[self.b_slice],
            jax.experimental.sparse((obj_values[self.G_idxs], *self.G_structure), shape=self.G_shape),
            obj_values[self.h_slice],
            self.l,
            self.u,
            self.solver
        )

    def solution_to_outputs(self, solution):
        return (solution.x[s] for s in self.output_slices)


@dataclass
class MPAX_data:
    Q: jnp.ndarray | jax.experimental.sparse.BCOO | jax.experimental.sparse.BCSR
    c: jnp.ndarray
    A: jnp.ndarray | jax.experimental.sparse.BCOO | jax.experimental.sparse.BCSR
    b: jnp.ndarray
    G: jnp.ndarray | jax.experimental.sparse.BCOO | jax.experimental.sparse.BCSR
    h: jnp.ndarray
    l: jnp.ndarray
    u: jnp.ndarray
    solver: Callable

    def solve(self):
        solution = self.solver(
            self.Q,
            self.c,
            self.A,
            self.b,
            self.G,
            self.h,
            self.l,
            self.u,
        )
        return solution

    def derivative(self):
        return 
    

class MPAX_CL_if:

    def jax_to_data(self, objective_values, constraints_values, context):
        obj_array = 

    def torch_to_data(self):
        ...

    def solve_with_data(self):
        ...

    def batch(self):
        ...

    def derivative(self):
        ...
