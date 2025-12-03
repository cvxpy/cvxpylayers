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
    TensorLike = torch.Tensor | jnp.ndarray | np.ndarray
else:
    TensorLike = Any


def _detect_batch_size(con_values: TensorLike) -> tuple[int, bool]:
    """Detect batch size and whether input was originally unbatched.

    Args:
        con_values: Constraint values (torch.Tensor or jnp.ndarray)

    Returns:
        Tuple of (batch_size, originally_unbatched)
    """
    ndim = con_values.dim() if hasattr(con_values, "dim") else con_values.ndim

    if ndim == 1:
        return 1, True
    else:
        return con_values.shape[1], False


def _cvxpy_dims_to_moreau_cones(dims: dict):
    """Convert CVXPYLayers cone dimensions to Moreau Cones object.

    Args:
        dims: Dictionary with keys 'z', 'l', 'q', 'ep', 'p', etc.

    Returns:
        moreau.Cones object
    """
    if moreau is None:
        raise ImportError(
            "Moreau solver requires 'moreau' package. "
            "Install with: pip install moreau"
        )

    cones = moreau.Cones()

    # Zero cone (equality constraints)
    cones.num_zero_cones = dims.get("z", 0)

    # Nonnegative cone (inequality constraints)
    cones.num_nonneg_cones = dims.get("l", 0)

    # Second-order cones - preserve actual SOC dimensions
    cones.soc_dims = list(dims.get("q", []))

    # Exponential cones
    cones.num_exp_cones = dims.get("ep", 0)

    # Power cones
    cones.power_alphas = list(dims.get("p", []))

    return cones


class MOREAU_ctx:
    """Context class for Moreau solver.

    Stores problem structure (CSR format) and creates solvers with lazy batch init.
    Batch size is inferred from inputs at solve time (moreau handles auto-reset).
    """

    P_idx: np.ndarray | None
    P_col_indices: np.ndarray
    P_row_offsets: np.ndarray
    P_shape: tuple[int, int]

    A_idx: np.ndarray
    A_col_indices: np.ndarray
    A_row_offsets: np.ndarray
    A_shape: tuple[int, int]
    b_idx: np.ndarray

    cones: Any  # moreau.Cones (unified for NumPy and PyTorch)
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

        # Store CSR structure
        # Note: convert_csc_structure_to_csr_structure returns (col_indices, row_offsets)
        self.P_idx = P_shuffle
        self.P_col_indices = P_structure[0].astype(np.int64)
        self.P_row_offsets = P_structure[1].astype(np.int64)
        self.P_shape = P_shape

        self.A_idx = A_shuffle
        self.A_col_indices = A_structure[0].astype(np.int64)
        self.A_row_offsets = A_structure[1].astype(np.int64)
        self.A_shape = A_shape
        self.b_idx = b_idx

        # Store dimensions
        self.dims = dims
        self.options = options or {}

        # Create cones and solver lazily
        self._cones = None
        self._torch_solver_cuda = None  # CUDA solver (moreau.torch.Solver) with lazy init
        self._torch_solver_cpu = None  # CPU solver (moreau.torch.Solver) with lazy init

    @property
    def cones(self):
        """Get moreau.Cones (unified for NumPy and PyTorch paths)."""
        if self._cones is None:
            self._cones = _cvxpy_dims_to_moreau_cones(dims_to_solver_dict(self.dims))
        return self._cones

    def _get_settings(self):
        """Get moreau.Settings configured from self.options.

        Accepts any moreau.Settings field names directly (e.g., max_iter,
        tol_gap_abs, verbose, etc.).
        """
        settings = moreau.Settings()

        # Set any field that exists on moreau.Settings
        for key, value in self.options.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        return settings

    def get_torch_solver(self, device: str):
        """Get moreau.torch.Solver for the specified device (lazy init).

        Args:
            device: 'cuda' or 'cpu'

        Returns:
            moreau.torch.Solver configured for the specified device
        """
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

    def torch_to_data(
        self, quad_obj_values, lin_obj_values, con_values
    ) -> "MOREAU_data":
        """Prepare data for torch solve.

        Device-aware: uses GPU solver for CUDA tensors, CPU solver for CPU tensors.
        - CUDA: Zero-copy GPU tensor operations with moreau.TorchSolver
        - CPU: Uses moreau.Solver (numpy interface) with zero-copy torch<->numpy
        """
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        batch_size, originally_unbatched = _detect_batch_size(con_values)

        # Add batch dimension for uniform handling
        if originally_unbatched:
            con_values = con_values.unsqueeze(1)
            lin_obj_values = lin_obj_values.unsqueeze(1)
            quad_obj_values = (
                quad_obj_values.unsqueeze(1) if quad_obj_values is not None else None
            )

        # Extract values using torch indexing (stays on GPU if input is on GPU)
        # con_values shape: (num_con_entries, batch)
        # lin_obj_values shape: (n+1, batch) - last entry is constant term

        # Extract P values in CSR order
        if self.P_idx is not None and quad_obj_values is not None:
            P_idx_tensor = torch.tensor(self.P_idx, dtype=torch.long, device=quad_obj_values.device)
            P_values = quad_obj_values[P_idx_tensor, :]  # (nnzP, batch)
        else:
            # Empty P matrix
            device = con_values.device
            P_values = torch.zeros((0, batch_size), dtype=torch.float64, device=device)

        # Extract A values in CSR order
        A_idx_tensor = torch.tensor(self.A_idx, dtype=torch.long, device=con_values.device)
        A_values = -con_values[A_idx_tensor, :]  # (nnzA, batch), negated for Ax + s = b form

        # Extract b vector
        b_idx_tensor = torch.tensor(self.b_idx, dtype=torch.long, device=con_values.device)
        # b is in the last b_idx.size entries of con_values
        b_start = con_values.shape[0] - self.b_idx.size
        b_raw = con_values[b_start:, :]  # (m, batch) but may need reordering
        # Create full b tensor and fill in at correct indices
        b = torch.zeros((self.A_shape[0], batch_size), dtype=torch.float64, device=con_values.device)
        b[b_idx_tensor, :] = b_raw

        # Extract q (linear cost)
        q = lin_obj_values[:-1, :]  # (n, batch), exclude constant term

        # Detect device from input tensors
        device = con_values.device
        is_cuda = device.type == "cuda"

        # Transpose to (batch, dim) format for Moreau
        # Use .to() which is a no-op if already on correct device/dtype (zero-copy for CUDA)
        P_values = P_values.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, nnzP)
        A_values = A_values.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, nnzA)
        q = q.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, n)
        b = b.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, m)

        # Select solver based on device
        solver = self.get_torch_solver('cuda' if is_cuda else 'cpu')

        return MOREAU_data(
            P_values=P_values,
            A_values=A_values,
            q=q,
            b=b,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
            solver=solver,
            n=self.P_shape[0],
            m=self.A_shape[0],
            is_cuda=is_cuda,
        )

    def jax_to_data(
        self, quad_obj_values, lin_obj_values, con_values
    ) -> "MOREAU_data_jax":
        """Prepare data for JAX solve."""
        if jnp is None:
            raise ImportError(
                "JAX interface requires 'jax' package. Install with: pip install jax"
            )

        batch_size, originally_unbatched = _detect_batch_size(con_values)

        # Add batch dimension for uniform handling
        if originally_unbatched:
            con_values = jnp.expand_dims(con_values, 1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, 1)
            quad_obj_values = (
                jnp.expand_dims(quad_obj_values, 1) if quad_obj_values is not None else None
            )

        # Build matrices (JAX path still uses numpy for now)
        P_vals_list, q_list, A_vals_list, b_list = [], [], [], []

        for i in range(batch_size):
            con_vals_i = np.array(con_values[:, i])
            lin_vals_i = np.array(lin_obj_values[:-1, i])
            quad_vals_i = (
                np.array(quad_obj_values[:, i]) if quad_obj_values is not None else None
            )

            # Build P matrix values in CSR order
            if self.P_idx is not None and quad_vals_i is not None:
                P_vals = quad_vals_i[self.P_idx]
            else:
                P_vals = np.array([], dtype=np.float64)

            # Build A matrix values in CSR order
            A_vals = con_vals_i[self.A_idx]

            # Build b vector
            b = np.zeros(self.A_shape[0], dtype=np.float64)
            b[self.b_idx] = con_vals_i[-self.b_idx.size:]

            P_vals_list.append(P_vals)
            A_vals_list.append(-A_vals)  # Negate for Ax + s = b form
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
class MOREAU_data:
    """Data class for PyTorch Moreau solver.

    Supports both CPU and CUDA tensors via moreau.torch.Solver:
    - CUDA: Uses moreau.torch.Solver(device='cuda') for GPU operations
    - CPU: Uses moreau.torch.Solver(device='cpu') with efficient batch solving
    """

    P_values: Any  # torch.Tensor (batch, nnzP)
    A_values: Any  # torch.Tensor (batch, nnzA)
    q: Any  # torch.Tensor (batch, n)
    b: Any  # torch.Tensor (batch, m)
    batch_size: int
    originally_unbatched: bool
    solver: Any  # moreau.torch.Solver (CPU or CUDA)
    n: int
    m: int
    is_cuda: bool = True  # Whether tensors are on CUDA

    def torch_solve(self, solver_args=None):
        """Solve using moreau.torch.Solver."""
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        # moreau.torch.Solver provides unified API for both CPU and CUDA
        x, z, s, status, obj_val = self.solver.solve(
            self.P_values, self.A_values, self.q, self.b
        )

        primal = x
        dual = z

        # No backward info - gradients not yet implemented
        backwards_info = None

        return primal, dual, backwards_info

    def torch_derivative(self, dprimal, ddual, backwards_info):
        """Compute gradients. NOT YET IMPLEMENTED."""
        raise NotImplementedError(
            "Moreau backward pass not yet implemented. "
            "Gradient support will be added in a future release."
        )


@dataclass
class MOREAU_data_jax:
    """Data class for JAX Moreau solver."""

    P_vals_list: list
    q_list: list
    A_vals_list: list
    b_list: list
    cones: Any  # moreau.Cones
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

        # Create solver (batch_size inferred from inputs via lazy init)
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

        # Stack values for batched solve in (batch, dim) format
        if self.batch_size == 1:
            # Single batch: add batch dimension
            P_values = self.P_vals_list[0][np.newaxis, :]  # (1, nnzP)
            A_values = self.A_vals_list[0][np.newaxis, :]  # (1, nnzA)
            q = self.q_list[0][np.newaxis, :]  # (1, n)
            b = self.b_list[0][np.newaxis, :]  # (1, m)
        else:
            # Stack into (batch, dim) shaped arrays
            if self.P_vals_list[0].size > 0:
                P_values = np.stack(self.P_vals_list)  # (batch, nnzP)
            else:
                P_values = np.zeros((self.batch_size, 0), dtype=np.float64)
            A_values = np.stack(self.A_vals_list)  # (batch, nnzA)
            q = np.stack(self.q_list)  # (batch, n)
            b = np.stack(self.b_list)  # (batch, m)

        # Solve - moreau returns (batch, dim) shaped outputs
        result = solver.solve(P_values, A_values, q, b)

        # Extract solutions - already in (batch, dim) format
        x = result["x"]  # (batch, n)
        z = result["z"]  # (batch, m)

        # Convert to JAX arrays
        primal = jnp.array(x)
        dual = jnp.array(z)

        # No backward info
        backwards_info = None

        return primal, dual, backwards_info

    def jax_derivative(self, dprimal, ddual, backwards_info):
        """Compute gradients. NOT YET IMPLEMENTED."""
        raise NotImplementedError(
            "Moreau backward pass not yet implemented. "
            "Gradient support will be added in a future release."
        )
