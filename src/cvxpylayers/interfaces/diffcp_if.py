from collections.abc import Callable
from dataclasses import dataclass

import diffcp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


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

    def __init__(
        self,
        objective_structure,
        constraint_structure,
        dims,
        lower_bounds,
        upper_bounds,
        options=None,
    ):
        con_indices, con_ptr, (m, np1) = constraint_structure

        self.A_structure = (con_indices, con_ptr)
        self.A_shape = (m, np1)
        self.b_idx = con_indices[con_ptr[-2] : con_ptr[-1]]

        self.dims = dims

    def torch_to_data(self, quad_obj_values, lin_obj_values, con_values) -> "DIFFCP_data":
        # Detect batch size and whether input was originally unbatched
        if con_values.dim() == 1:
            originally_unbatched = True
            batch_size = 1
            # Add batch dimension for uniform handling
            con_values = con_values.unsqueeze(1)
            lin_obj_values = lin_obj_values.unsqueeze(1)
        else:
            originally_unbatched = False
            batch_size = con_values.shape[1]

        # Build lists for all batch elements
        As, bs, cs, b_idxs = [], [], [], []
        for i in range(batch_size):
            A_aug = sp.csc_matrix(
                (con_values[:, i].cpu().numpy(), *self.A_structure),
                shape=self.A_shape,
            )
            As.append(-A_aug[:, :-1])  # Negate A to match DIFFCP convention
            bs.append(A_aug[:, -1].toarray().flatten())
            cs.append(lin_obj_values[:-1, i].cpu().numpy())
            b_idxs.append(self.b_idx)

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            b_idxs=b_idxs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )

    def jax_to_data(self, quad_obj_values, lin_obj_values, con_values) -> "DIFFCP_data":
        if jnp is None:
            raise ImportError(
                "JAX interface requires 'jax' package to be installed. "
                "Install with: pip install jax"
            )
        # Detect batch size and whether input was originally unbatched
        if con_values.ndim == 1:
            originally_unbatched = True
            batch_size = 1
            # Add batch dimension for uniform handling
            con_values = jnp.expand_dims(con_values, 1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, 1)
        else:
            originally_unbatched = False
            batch_size = con_values.shape[1]

        # Build lists for all batch elements
        As, bs, cs, b_idxs = [], [], [], []
        for i in range(batch_size):
            A_aug = sp.csc_matrix(
                (np.array(con_values[:, i]), *self.A_structure),
                shape=self.A_shape,
            )
            As.append(-A_aug[:, :-1])  # Negate A to match DIFFCP convention
            bs.append(A_aug[:, -1].toarray().flatten())
            cs.append(np.array(lin_obj_values[:-1, i]))
            b_idxs.append(self.b_idx)

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            b_idxs=b_idxs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )


@dataclass
class DIFFCP_data:
    As: list[sp.csc_matrix]
    bs: list[np.ndarray]
    cs: list[np.ndarray]
    b_idxs: list[np.ndarray]
    cone_dict: dict[str, int | list[int]]
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
        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **solver_args,
        )
        # Stack results into batched tensors
        primal = torch.stack([torch.from_numpy(x) for x in xs])
        dual = torch.stack([torch.from_numpy(y) for y in ys])
        return primal, dual, adj_batch

    def torch_derivative(self, primal, dual, adj_batch):
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        # Split batched tensors into lists
        dxs = [primal[i].numpy() for i in range(self.batch_size)]
        dys = [dual[i].numpy() for i in range(self.batch_size)]
        dss = [np.zeros_like(self.bs[i]) for i in range(self.batch_size)]

        # Call batch adjoint
        dAs, dbs, dcs = adj_batch(dxs, dys, dss)

        # Aggregate gradients from each batch element
        dq_batch = []
        dA_batch = []
        for i in range(self.batch_size):
            # Negate dA because A was negated in forward pass, but not db (b was not negated)
            con_grad = np.hstack([-dAs[i].data, dbs[i][self.b_idxs[i]]])
            lin_grad = np.hstack([dcs[i], np.array([0.0])])
            dA_batch.append(con_grad)
            dq_batch.append(lin_grad)

        # Stack into shape (num_entries, batch_size)
        dq_stacked = torch.stack([torch.from_numpy(g) for g in dq_batch]).T
        dA_stacked = torch.stack([torch.from_numpy(g) for g in dA_batch]).T

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dq_stacked = dq_stacked.squeeze(1)
            dA_stacked = dA_stacked.squeeze(1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )

    def jax_solve(self, solver_args=None):
        if solver_args is None:
            solver_args = {}

        # Always use batch solve
        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **solver_args,
        )

        # Stack results into batched arrays
        primal = jnp.stack([jnp.array(x) for x in xs])
        dual = jnp.stack([jnp.array(y) for y in ys])

        # Return primal, dual, and adjoint function for backward pass
        return primal, dual, adj_batch

    def jax_derivative(self, dprimal, ddual, adj_batch):
        # Split batched arrays into lists
        dxs = [np.array(dprimal[i]) for i in range(self.batch_size)]
        dys = [np.array(ddual[i]) for i in range(self.batch_size)]
        dss = [np.zeros_like(self.bs[i]) for i in range(self.batch_size)]

        # Call batch adjoint
        dAs, dbs, dcs = adj_batch(dxs, dys, dss)

        # Aggregate gradients from each batch element
        dq_batch = []
        dA_batch = []
        for i in range(self.batch_size):
            # Negate dA because A was negated in forward pass, but not db (b was not negated)
            con_grad = np.hstack([-dAs[i].data, dbs[i][self.b_idxs[i]]])
            lin_grad = np.hstack([dcs[i], np.array([0.0])])
            dA_batch.append(con_grad)
            dq_batch.append(lin_grad)

        # Stack into shape (num_entries, batch_size)
        dq_stacked = jnp.stack([jnp.array(g) for g in dq_batch]).T
        dA_stacked = jnp.stack([jnp.array(g) for g in dA_batch]).T

        # Squeeze batch dimension only if input was originally unbatched
        if self.originally_unbatched:
            dq_stacked = jnp.squeeze(dq_stacked, 1)
            dA_stacked = jnp.squeeze(dA_stacked, 1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )
