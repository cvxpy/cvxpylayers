from dataclasses import dataclass
from typing import Callable
import scipy.sparse as sp
import numpy as np
import diffcp

class DIFFCP_ctx:
    c_slice: slice

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

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1
        con_slice_start = con_ptr[-2]
        self.b_slice = slice(con_slice_start, con_slice_start + dims.zero)
        self.h_slice = slice(con_slice_start + dims.zero, con_ptr[-1])

        con_csc = sp.csc_array((np.arange(con_indices.size), con_indices, con_ptr[:-1]), shape=(m, n))

        self.A_idxs = con_csc.data
        self.A_structure = con_csc.indices, con_csr.indptr
        self.A_shape = (m, n)

        self.dims = dims

        self.warm_start = options.pop('warm_start', False)
        assert self.warm_start is False

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
        return DIFFCP_data(
            model,
            self.solver
        )

    def solution_to_outputs(self, solution):
        return (solution.primal_solution[s] for s in self.output_slices)


@dataclass
class DIFFCP_data:
    A: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    cone_dicts: dict[str, int | list[int]]

    def solve(self):
        x, y, s, self.diff, self.adj = diffcp.solve_and_derivative(self.A, self.b, self.c, self.cone_dict)
        return x, y

    def derivative(self, primal, dual):
        dA, db, dc = self.adj(primal, dual, np.zeros_like(self.b))
        return dA, db, dc

    def torch_derivative(self, primal, dual):
        import torch
        con_mat, con_vec, lin = derivative(self, np.array(primal), np.array(dual))
        return None, torch.tensor(lin), torch.tensor(con_mat), torch.tensor(con_vec)
