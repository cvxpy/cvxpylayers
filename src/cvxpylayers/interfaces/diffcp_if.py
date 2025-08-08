from dataclasses import dataclass
from typing import Callable
import scipy.sparse as sp
import numpy as np
import diffcp

class DIFFCP_ctx:
    Q_idxs: np.ndarray
    c_slice: slice
    Q_structure: tuple[np.ndarray, np.ndarray]
    Q_shape: tuple[int, int]

    A_idxs: np.ndarray
    b_slice: slice
    A_structure: tuple[np.ndarray, np.ndarray]
    A_shape: tuple[int, int]

    G_idxs: np.ndarray
    h_slice: slice
    G_structure: tuple[np.ndarray, np.ndarray]
    G_shape: tuple[int, int]

    l: np.ndarray
    u: np.ndarray

    solver: Callable

    output_slices: list[slice]

    def __init__(self, objective_structure, constraint_structure, dims, lower_bounds, upper_bounds, output_slices, options):
        obj_indices, obj_ptr, (n, _) = objective_structure
        self.c_slice = slice(0, n)
        obj_csr = sp.csc_array((np.arange(obj_indices.size), obj_indices, obj_ptr), shape=(n, n)).tocsr()
        self.Q_idxs = obj_csr.data
        self.Q_structure = obj_csr.indices, obj_csr.indptr
        self.Q_shape = (n, n)

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1
        con_slice_start = con_ptr[-2]
        self.b_slice = slice(con_slice_start, con_slice_start + dims.zero)
        self.h_slice = slice(con_slice_start + dims.zero, con_ptr[-1])

        con_csr = sp.csc_array((np.arange(con_indices.size), con_indices, con_ptr[:-1]), shape=(m, n)).tocsr()
        split = con_csr.indptr[dims.zero]

        self.A_idxs = con_csr.data[:split]
        self.A_structure = con_csr.indices[:split], con_csr.indptr[:dims.zero+1]
        self.A_shape = (dims.zero, n)

        self.G_idxs = con_csr.data[split:]
        self.G_structure = con_csr.indices[split:], con_csr.indptr[dims.zero:] - split
        self.G_shape = (m - dims.zero, n)

        self.l = lower_bounds if lower_bounds is not None else -np.inf * np.ones(n)
        self.u = upper_bounds if upper_bounds is not None else np.inf * np.ones(n)

        self.warm_start = options.pop('warm_start', False)
        assert self.warm_start is False
        algorithm = options.pop('algorithm', 'raPDHG')
        if algorithm == 'raPDHG':
            alg = mpax.raPDHG
        elif algorithm == 'r2HPDHG':
            alg = mpax.r2HPDHG
        else:
            raise ValueError('Invalid MPAX algorithm')
        solver = alg(warm_start=self.warm_start, **options)
        self.solver = jax.jit(solver.optimize)
        self.output_slices = output_slices

    def jax_to_data(self, quad_obj_values, lin_obj_values, con_values):   # TODO: Add broadcasting  (will need np.tile to tile structures)
        model = mpax.create_qp(
            P:=jax.experimental.sparse.BCSR((quad_obj_values[self.Q_idxs], *self.Q_structure), shape=self.Q_shape),
            q:=lin_obj_values[self.c_slice],
            A:=jax.experimental.sparse.BCSR((con_values[self.A_idxs], *self.A_structure), shape=self.A_shape),
            b:=con_values[self.b_slice],
            G:=jax.experimental.sparse.BCSR((con_values[self.G_idxs], *self.G_structure), shape=self.G_shape),
            h:=con_values[self.h_slice],
            self.l,
            self.u,
        )
        return MPAX_data(
            model,
            self.solver
        )

    def solution_to_outputs(self, solution):
        return (solution.primal_solution[s] for s in self.output_slices)


@dataclass
class MPAX_data:
    model: mpax.utils.QuadraticProgrammingProblem
    solver: Callable

    def solve(self):
        solution = self.solver(
            self.model
        )
        return solution.x, solution.y

    def derivative(self, primal, dual):
        return 

    def torch_derivative(self, primal, dual):
        import torch
        quad, lin, con = derivative(self, np.array(primal), np.array(dual))
        return torch.tensor(quad), torch.tensor(lin), torch.tensor(con)
