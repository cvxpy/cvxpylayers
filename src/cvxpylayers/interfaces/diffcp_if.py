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

    def __init__(self, objective_structure, constraint_structure, dims, lower_bounds, upper_bounds, output_slices, options=None):
        con_indices, con_ptr, (m, np1) = constraint_structure

        self.A_structure = constraint_structure
        self.A_shape = (m, np1)

        self.dims = dims


    def torch_to_data(self, quad_obj_values, lin_obj_values, con_values):
        A_aug = sp.csc_matrix((self.con_values.cpu().numpy(), self.A_structure), shape=self.A_shape)
        return DIFFCP_data(
            A=A_aug[:, :-1],
            b=A_aug[:, -1].toarray(),
            c=lin_obj_values,
            cone_dicts=self.dims,
        )

    def solution_to_outputs(self, solution):
        return (solution.primal_solution[s] for s in self.output_slices)


@dataclass
class DIFFCP_data:
    A: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    cone_dicts: dict[str, int | list[int]]

    def torch_solve(self):
        import torch
        x, y, s, self.diff, self.adj = diffcp.solve_and_derivative(self.A, self.b, self.c, self.cone_dict)
        return torch.from_numpy(x), torch.from_numpy(y)

    def derivative(self, primal, dual):
        dA, db, dc = self.adj(primal, dual, np.zeros_like(self.b))
        return dA, db, dc

    def torch_derivative(self, primal, dual):
        import torch
        con_mat, con_vec, lin = derivative(self, np.array(primal), np.array(dual))
        return None, torch.tensor(lin), torch.tensor(con_mat), torch.tensor(con_vec)
