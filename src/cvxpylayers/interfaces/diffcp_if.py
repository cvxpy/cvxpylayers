from dataclasses import dataclass
from typing import Callable
import scipy.sparse as sp
import numpy as np
import diffcp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict


class DIFFCP_ctx:
    c_slice: slice

    A_idxs: np.ndarray
    b_idx: np.ndarray
    A_structure: tuple[np.ndarray, np.ndarray]
    A_shape: tuple[int, int]

    G_idxs: np.ndarray
    h_slice: slice
    G_structure: tuple[np.ndarray, np.ndarray]
    G_shape: tuple[int, int]

    solver: Callable

    output_slices: list[slice]

    def __init__(
        self,
        objective_structure,
        constraint_structure,
        dims,
        lower_bounds,
        upper_bounds,
        output_slices,
        options=None,
    ):
        con_indices, con_ptr, (m, np1) = constraint_structure

        self.A_structure = (con_indices, con_ptr)
        self.A_shape = (m, np1)
        self.b_idx = con_indices[con_ptr[-2] : con_ptr[-1]]

        self.dims = dims

    def torch_to_data(self, quad_obj_values, lin_obj_values, con_values):
        A_aug = sp.csc_matrix(
            (con_values.cpu().numpy(), *self.A_structure), shape=self.A_shape
        )
        return DIFFCP_data(
            A=A_aug[:, :-1],
            b=A_aug[:, -1].toarray().flatten(),
            c=lin_obj_values[:-1].cpu().numpy(),
            b_idx=self.b_idx,
            cone_dict=dims_to_solver_dict(self.dims),
        )

    def solution_to_outputs(self, solution):
        return (solution.primal_solution[s] for s in self.output_slices)


@dataclass
class DIFFCP_data:
    A: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    b_idx: np.ndarray
    cone_dict: dict[str, int | list[int]]

    def torch_solve(self):
        import torch

        print(self.cone_dict)
        x, y, s, _, adj = diffcp.solve_and_derivative(
            self.A, self.b, self.c, self.cone_dict
        )
        return torch.from_numpy(x), torch.from_numpy(y), adj

    def derivative(self, primal, dual, adj):
        dA, db, dc = adj(primal, dual, np.zeros_like(self.b))
        return np.hstack([dA.data, db[self.b_idx]]), dc

    def torch_derivative(self, primal, dual, adj):
        import torch

        con, lin = self.derivative(primal.numpy(), dual.numpy(), adj)
        return (
            None,
            torch.hstack([torch.from_numpy(lin), torch.tensor(0.0)]),
            torch.from_numpy(con),
        )
