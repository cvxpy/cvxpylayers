"""Moreau solver interface for CVXPYLayers.

Moreau is a conic optimization solver that solves problems of the form:
    minimize    (1/2)x'Px + q'x
    subject to  Ax + s = b
                s in K

where K is a product of cones.

Supports both CPU and GPU (CUDA) tensors via moreau.torch.Solver:
- CUDA tensors: Uses moreau.torch.Solver(device='cuda') for GPU operations
- CPU tensors: Uses moreau.torch.Solver(device='cpu') with efficient batch solving
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

from cvxpylayers.utils.solver_utils import convert_csc_structure_to_csr_structure

# Moreau unified package (provides both NumPy and PyTorch interfaces)
try:
    import moreau
    import moreau.torch as moreau_torch
except ImportError:
    moreau = None  # type: ignore[assignment]
    moreau_torch = None  # type: ignore[assignment]

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import cvxpylayers.utils.parse_args as pa

    TensorLike = torch.Tensor | jnp.ndarray | np.ndarray
else:
    TensorLike = Any


def _detect_batch_size(con_values: TensorLike) -> tuple[int, bool]:
    """Detect batch size and whether input was originally unbatched."""
    ndim = con_values.dim() if hasattr(con_values, "dim") else con_values.ndim
    if ndim == 1:
        return 1, True
    else:
        return con_values.shape[1], False


def _cvxpy_dims_to_moreau_cones(dims: dict):
    """Convert CVXPYLayers cone dimensions to Moreau Cones object."""
    if moreau is None:
        raise ImportError(
            "Moreau solver requires 'moreau' package. "
            "Install with: pip install moreau"
        )

    cones = moreau.Cones()
    cones.num_zero_cones = dims.get("z", 0)
    cones.num_nonneg_cones = dims.get("l", 0)
    cones.soc_dims = list(dims.get("q", []))
    cones.num_exp_cones = dims.get("ep", 0)
    cones.power_alphas = list(dims.get("p", []))

    return cones


class MOREAU_ctx:
    """Context class for Moreau solver."""

    P_idx: np.ndarray | None
    P_col_indices: np.ndarray
    P_row_offsets: np.ndarray
    P_shape: tuple[int, int]

    A_idx: np.ndarray
    A_col_indices: np.ndarray
    A_row_offsets: np.ndarray
    A_shape: tuple[int, int]
    b_idx: np.ndarray

    dims: dict

    def __init__(
        self,
        objective_structure,
        constraint_structure,
        dims,
        lower_bounds,
        upper_bounds,
        options=None,
    ):
        # Convert constraint matrix from CSC to CSR
        A_shuffle, A_structure, A_shape, b_idx = convert_csc_structure_to_csr_structure(
            constraint_structure, True
        )

        # Convert objective matrix from CSC to CSR
        if objective_structure is not None:
            P_shuffle, P_structure, P_shape = convert_csc_structure_to_csr_structure(
                objective_structure, False
            )
            assert P_shape[0] == P_shape[1]
        else:
            P_shuffle = None
            P_structure = (np.array([], dtype=np.int64), np.zeros(A_shape[1] + 1, dtype=np.int64))
            P_shape = (A_shape[1], A_shape[1])

        assert P_shape[0] == A_shape[1]

        self.P_idx = P_shuffle
        self.P_col_indices = P_structure[0].astype(np.int64)
        self.P_row_offsets = P_structure[1].astype(np.int64)
        self.P_shape = P_shape

        self.A_idx = A_shuffle
        self.A_col_indices = A_structure[0].astype(np.int64)
        self.A_row_offsets = A_structure[1].astype(np.int64)
        self.A_shape = A_shape
        self.b_idx = b_idx

        self.dims = dims
        self.options = options or {}

        self._cones = None
        self._torch_solver_cuda = None
        self._torch_solver_cpu = None

    @property
    def cones(self):
        """Get moreau.Cones (unified for NumPy and PyTorch paths)."""
        if self._cones is None:
            self._cones = _cvxpy_dims_to_moreau_cones(dims_to_solver_dict(self.dims))
        return self._cones

    def _get_settings(self):
        """Get moreau.Settings configured from self.options."""
        settings = moreau.Settings()
        for key, value in self.options.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return settings

    def get_torch_solver(self, device: str):
        """Get moreau.torch.Solver for the specified device (lazy init)."""
        if device == 'cuda':
            if self._torch_solver_cuda is None:
                if moreau_torch is None or not moreau.device_available('cuda'):
                    raise ImportError(
                        "Moreau CUDA backend requires 'moreau' package with CUDA support. "
                        "Install with: pip install moreau[cuda]"
                    )
                self._torch_solver_cuda = moreau_torch.Solver(
                    n=self.P_shape[0],
                    m=self.A_shape[0],
                    P_row_offsets=torch.tensor(self.P_row_offsets, dtype=torch.int64),
                    P_col_indices=torch.tensor(self.P_col_indices, dtype=torch.int64),
                    A_row_offsets=torch.tensor(self.A_row_offsets, dtype=torch.int64),
                    A_col_indices=torch.tensor(self.A_col_indices, dtype=torch.int64),
                    cones=self.cones,
                    settings=self._get_settings(),
                    device='cuda',
                )
            return self._torch_solver_cuda
        else:
            if self._torch_solver_cpu is None:
                if moreau_torch is None:
                    raise ImportError(
                        "Moreau solver requires 'moreau' package. "
                        "Install with: pip install moreau"
                    )
                self._torch_solver_cpu = moreau_torch.Solver(
                    n=self.P_shape[0],
                    m=self.A_shape[0],
                    P_row_offsets=torch.tensor(self.P_row_offsets, dtype=torch.int64),
                    P_col_indices=torch.tensor(self.P_col_indices, dtype=torch.int64),
                    A_row_offsets=torch.tensor(self.A_row_offsets, dtype=torch.int64),
                    A_col_indices=torch.tensor(self.A_col_indices, dtype=torch.int64),
                    cones=self.cones,
                    settings=self._get_settings(),
                    device='cpu',
                )
            return self._torch_solver_cpu

    def jax_to_data(
        self, quad_obj_values, lin_obj_values, con_values
    ) -> "MOREAU_data_jax":
        """Prepare data for JAX solve."""
        if jnp is None:
            raise ImportError(
                "JAX interface requires 'jax' package. Install with: pip install jax"
            )

        batch_size, originally_unbatched = _detect_batch_size(con_values)

        if originally_unbatched:
            con_values = jnp.expand_dims(con_values, 1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, 1)
            quad_obj_values = (
                jnp.expand_dims(quad_obj_values, 1) if quad_obj_values is not None else None
            )

        P_vals_list, q_list, A_vals_list, b_list = [], [], [], []

        for i in range(batch_size):
            con_vals_i = np.array(con_values[:, i])
            lin_vals_i = np.array(lin_obj_values[:-1, i])
            quad_vals_i = (
                np.array(quad_obj_values[:, i]) if quad_obj_values is not None else None
            )

            if self.P_idx is not None and quad_vals_i is not None:
                P_vals = quad_vals_i[self.P_idx]
            else:
                P_vals = np.array([], dtype=np.float64)

            A_vals = con_vals_i[self.A_idx]

            b = np.zeros(self.A_shape[0], dtype=np.float64)
            b[self.b_idx] = con_vals_i[-self.b_idx.size:]

            P_vals_list.append(P_vals)
            A_vals_list.append(-A_vals)
            b_list.append(b)
            q_list.append(lin_vals_i)

        return MOREAU_data_jax(
            P_vals_list=P_vals_list,
            q_list=q_list,
            A_vals_list=A_vals_list,
            b_list=b_list,
            cones=self.cones,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
            P_col_indices=self.P_col_indices,
            P_row_offsets=self.P_row_offsets,
            A_col_indices=self.A_col_indices,
            A_row_offsets=self.A_row_offsets,
            n=self.P_shape[0],
            m=self.A_shape[0],
            options=self.options,
        )


@dataclass
class MOREAU_data_jax:
    """Data class for JAX Moreau solver."""

    P_vals_list: list
    q_list: list
    A_vals_list: list
    b_list: list
    cones: Any
    batch_size: int
    originally_unbatched: bool
    P_col_indices: np.ndarray
    P_row_offsets: np.ndarray
    A_col_indices: np.ndarray
    A_row_offsets: np.ndarray
    n: int
    m: int
    options: dict

    def jax_solve(self, solver_args=None):
        """Solve using moreau (CPU/GPU via numpy)."""
        if jnp is None:
            raise ImportError(
                "JAX interface requires 'jax' package. Install with: pip install jax"
            )
        if moreau is None:
            raise ImportError(
                "Moreau solver requires 'moreau' package. Install with: pip install moreau"
            )

        if solver_args is None:
            solver_args = {}

        settings = moreau.Settings()
        settings.verbose = solver_args.get("verbose", self.options.get("verbose", False))

        solver = moreau.Solver(
            n=self.n,
            m=self.m,
            P_row_offsets=self.P_row_offsets,
            P_col_indices=self.P_col_indices,
            A_row_offsets=self.A_row_offsets,
            A_col_indices=self.A_col_indices,
            cones=self.cones,
            settings=settings,
        )

        if self.batch_size == 1:
            P_values = self.P_vals_list[0][np.newaxis, :]
            A_values = self.A_vals_list[0][np.newaxis, :]
            q = self.q_list[0][np.newaxis, :]
            b = self.b_list[0][np.newaxis, :]
        else:
            if self.P_vals_list[0].size > 0:
                P_values = np.stack(self.P_vals_list)
            else:
                P_values = np.zeros((self.batch_size, 0), dtype=np.float64)
            A_values = np.stack(self.A_vals_list)
            q = np.stack(self.q_list)
            b = np.stack(self.b_list)

        result = solver.solve(P_values, A_values, q, b)

        primal = jnp.array(result["x"])
        dual = jnp.array(result["z"])

        return primal, dual, None

    def jax_derivative(self, dprimal, ddual, backwards_info):
        """Compute gradients. NOT YET IMPLEMENTED."""
        raise NotImplementedError(
            "Moreau backward pass not yet implemented. "
            "Gradient support will be added in a future release."
        )


if torch is not None:
    class _CvxpyLayer(torch.autograd.Function):
        @staticmethod
        def forward(
            P_eval: torch.Tensor | None,
            q_eval: torch.Tensor,
            A_eval: torch.Tensor,
            cl_ctx: "pa.LayersContext",
            solver_args: dict[str, Any],
        ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
            ctx = cl_ctx.solver_ctx

            batch_size, originally_unbatched = _detect_batch_size(A_eval)

            if originally_unbatched:
                A_eval = A_eval.unsqueeze(1)
                q_eval = q_eval.unsqueeze(1)
                P_eval = P_eval.unsqueeze(1) if P_eval is not None else None

            # Extract P values in CSR order
            if ctx.P_idx is not None and P_eval is not None:
                P_idx_tensor = torch.tensor(ctx.P_idx, dtype=torch.long, device=P_eval.device)
                P_values = P_eval[P_idx_tensor, :]
            else:
                device = A_eval.device
                P_values = torch.zeros((0, batch_size), dtype=torch.float64, device=device)

            # Extract A values in CSR order
            A_idx_tensor = torch.tensor(ctx.A_idx, dtype=torch.long, device=A_eval.device)
            A_values = -A_eval[A_idx_tensor, :]

            # Extract b vector
            b_idx_tensor = torch.tensor(ctx.b_idx, dtype=torch.long, device=A_eval.device)
            b_start = A_eval.shape[0] - ctx.b_idx.size
            b_raw = A_eval[b_start:, :]
            b = torch.zeros((ctx.A_shape[0], batch_size), dtype=torch.float64, device=A_eval.device)
            b[b_idx_tensor, :] = b_raw

            # Extract q (linear cost)
            q = q_eval[:-1, :]

            # Detect device and get solver
            device = A_eval.device
            is_cuda = device.type == "cuda"

            # Transpose to (batch, dim) format for Moreau
            P_values = P_values.T.contiguous().to(device=device, dtype=torch.float64)
            A_values = A_values.T.contiguous().to(device=device, dtype=torch.float64)
            q = q.T.contiguous().to(device=device, dtype=torch.float64)
            b = b.T.contiguous().to(device=device, dtype=torch.float64)

            solver = ctx.get_torch_solver('cuda' if is_cuda else 'cpu')

            # Solve
            x, z, s, status, obj_val = solver.solve(P_values, A_values, q, b)

            return x, z, None, None

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
            pass

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, dprimal: torch.Tensor, ddual: torch.Tensor, _: Any, __: Any
        ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None, None]:
            raise NotImplementedError(
                "Moreau backward pass not yet implemented. "
                "Gradient support will be added in a future release."
            )
