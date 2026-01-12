"""Moreau solver interface for CVXPYLayers.

Moreau is a conic optimization solver that solves problems of the form:
    minimize    (1/2)x'Px + q'x
    subject to  Ax + s = b
                s in K
                l <= x <= u

where K is a product of cones.

Uses Moreau's native PyTorch and JAX solvers with built-in automatic differentiation:
- PyTorch: moreau.torch.Solver with three-step API and autograd support
- JAX: moreau.jax.Solver with custom_vjp for gradients
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

from cvxpylayers.utils.solver_utils import convert_csc_structure_to_csr_structure

# Moreau unified package (provides both NumPy and PyTorch interfaces)
try:
    import moreau
    import moreau.jax as moreau_jax
    import moreau.torch as moreau_torch
except ImportError:
    moreau = None  # type: ignore[assignment]
    moreau_torch = None  # type: ignore[assignment]
    moreau_jax = None  # type: ignore[assignment]

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None  # type: ignore[assignment]
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


if torch is not None:

    class _CvxpyLayer:
        """PyTorch layer using Moreau's native autograd.

        Unlike other solvers that implement torch.autograd.Function with custom
        backward, Moreau provides built-in autograd support. We just need to:
        1. Convert cvxpylayers format to Moreau format (preserving autograd)
        2. Call Moreau's solve (which returns tensors with grad_fn attached)
        3. Return the results directly (no custom backward needed)
        """

        @staticmethod
        def apply(
            P_eval: torch.Tensor | None,
            q_eval: torch.Tensor,
            A_eval: torch.Tensor,
            cl_ctx: "pa.LayersContext",
            solver_args: dict[str, Any],
        ) -> tuple[torch.Tensor, torch.Tensor, None, Any]:
            """Forward pass using Moreau's native solve with autograd.

            Returns (primal, dual, None, data) matching the interface expected
            by CvxpyLayer. The primal and dual tensors have Moreau's grad_fn
            attached and will automatically compute gradients during backward.
            """
            data = cl_ctx.solver_ctx.torch_to_data(P_eval, q_eval, A_eval)
            primal, dual, _ = data.torch_solve(solver_args)
            # Return primal/dual with Moreau's grad_fn intact
            # Third element (backwards_info) is None - not used since Moreau handles backward
            return primal, dual, None, data


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
            "Moreau solver requires 'moreau' package. Install with: pip install moreau"
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

    # Variable bounds (fixed at construction)
    lower: np.ndarray  # Lower bounds, shape (n,)
    upper: np.ndarray  # Upper bounds, shape (n,)

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

        # Store bounds (fixed at construction)
        n = P_shape[0]
        self.lower = lower_bounds if lower_bounds is not None else np.full(n, -np.inf)
        self.upper = upper_bounds if upper_bounds is not None else np.full(n, np.inf)

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

    def _get_settings(self, enable_grad: bool = True):
        """Get moreau.Settings configured from self.options.

        Args:
            enable_grad: Whether to enable gradient computation in Moreau.

        Accepts any moreau.Settings field names directly (e.g., max_iter,
        tol_gap_abs, verbose, etc.).
        """
        settings = moreau.Settings(enable_grad=enable_grad)

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
            moreau.torch.Solver configured for the specified device with enable_grad=True
        """
        if device == "cuda":
            if self._torch_solver_cuda is None:
                if moreau_torch is None or not moreau.device_available("cuda"):
                    raise ImportError(
                        "Moreau CUDA backend requires 'moreau' package with CUDA support. "
                        "Install with: pip install moreau[cuda]"
                    )
                settings = self._get_settings(enable_grad=True)
                settings.device = "cuda"
                self._torch_solver_cuda = moreau_torch.Solver(
                    n=self.P_shape[0],
                    m=self.A_shape[0],
                    P_row_offsets=torch.tensor(self.P_row_offsets, dtype=torch.int64),
                    P_col_indices=torch.tensor(self.P_col_indices, dtype=torch.int64),
                    A_row_offsets=torch.tensor(self.A_row_offsets, dtype=torch.int64),
                    A_col_indices=torch.tensor(self.A_col_indices, dtype=torch.int64),
                    cones=self.cones,
                    settings=settings,
                )
            return self._torch_solver_cuda
        else:
            if self._torch_solver_cpu is None:
                if moreau_torch is None:
                    raise ImportError(
                        "Moreau solver requires 'moreau' package. Install with: pip install moreau"
                    )
                settings = self._get_settings(enable_grad=True)
                settings.device = "cpu"
                self._torch_solver_cpu = moreau_torch.Solver(
                    n=self.P_shape[0],
                    m=self.A_shape[0],
                    P_row_offsets=torch.tensor(self.P_row_offsets, dtype=torch.int64),
                    P_col_indices=torch.tensor(self.P_col_indices, dtype=torch.int64),
                    A_row_offsets=torch.tensor(self.A_row_offsets, dtype=torch.int64),
                    A_col_indices=torch.tensor(self.A_col_indices, dtype=torch.int64),
                    cones=self.cones,
                    settings=settings,
                )
            return self._torch_solver_cpu

    def torch_to_data(self, quad_obj_values, lin_obj_values, con_values) -> "MOREAU_data":
        """Prepare data for torch solve.

        Device-aware: uses GPU solver for CUDA tensors, CPU solver for CPU tensors.
        - CUDA: Uses moreau.torch.Solver(device='cuda') for GPU operations
        - CPU: Uses moreau.torch.Solver(device='cpu') with efficient batch solving
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
            quad_obj_values = quad_obj_values.unsqueeze(1) if quad_obj_values is not None else None

        # Detect device from input tensors
        device = con_values.device
        is_cuda = device.type == "cuda"

        # Extract values using torch indexing (stays on GPU if input is on GPU)
        # con_values shape: (num_con_entries, batch)
        # lin_obj_values shape: (n+1, batch) - last entry is constant term

        # Extract P values in CSR order
        if self.P_idx is not None and quad_obj_values is not None:
            P_idx_tensor = torch.tensor(self.P_idx, dtype=torch.long, device=device)
            P_values = quad_obj_values[P_idx_tensor, :]  # (nnzP, batch)
        else:
            # Empty P matrix
            P_values = torch.zeros((0, batch_size), dtype=torch.float64, device=device)
            P_idx_tensor = torch.tensor([], dtype=torch.long, device=device)

        # Extract A values in CSR order
        A_idx_tensor = torch.tensor(self.A_idx, dtype=torch.long, device=device)
        A_values = -con_values[A_idx_tensor, :]  # (nnzA, batch), negated for Ax + s = b form

        # Extract b vector
        b_idx_tensor = torch.tensor(self.b_idx, dtype=torch.long, device=device)
        # b is in the last b_idx.size entries of con_values
        b_start = con_values.shape[0] - self.b_idx.size
        b_raw = con_values[b_start:, :]  # (m, batch) but may need reordering
        # Create full b tensor and fill in at correct indices
        b = torch.zeros((self.A_shape[0], batch_size), dtype=torch.float64, device=device)
        b[b_idx_tensor, :] = b_raw

        # Extract q (linear cost)
        q = lin_obj_values[:-1, :]  # (n, batch), exclude constant term

        # Transpose to (batch, dim) format for Moreau
        # Use .to() which is a no-op if already on correct device/dtype (zero-copy for CUDA)
        P_values = P_values.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, nnzP)
        A_values = A_values.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, nnzA)
        q = q.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, n)
        b = b.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, m)

        # Convert bounds to torch and broadcast to batch
        lower = torch.tensor(self.lower, dtype=torch.float64, device=device)  # (n,)
        upper = torch.tensor(self.upper, dtype=torch.float64, device=device)  # (n,)
        lower = lower.unsqueeze(0).expand(batch_size, -1).contiguous()  # (batch, n)
        upper = upper.unsqueeze(0).expand(batch_size, -1).contiguous()  # (batch, n)

        # Select solver based on device
        solver = self.get_torch_solver("cuda" if is_cuda else "cpu")

        return MOREAU_data(
            P_values=P_values,
            A_values=A_values,
            q=q,
            b=b,
            lower=lower,
            upper=upper,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
            solver=solver,
            n=self.P_shape[0],
            m=self.A_shape[0],
            is_cuda=is_cuda,
            # Store indices for gradient mapping
            P_idx_tensor=P_idx_tensor,
            A_idx_tensor=A_idx_tensor,
            b_idx_tensor=b_idx_tensor,
            # Store shapes for gradient scatter
            P_eval_size=quad_obj_values.shape[0] if quad_obj_values is not None else 0,
            q_eval_size=lin_obj_values.shape[0],
            A_eval_size=con_values.shape[0],
        )

    def jax_to_data(self, quad_obj_values, lin_obj_values, con_values) -> "MOREAU_data_jax":
        """Prepare data for JAX solve using moreau.jax.Solver."""
        if jnp is None:
            raise ImportError("JAX interface requires 'jax' package. Install with: pip install jax")

        batch_size, originally_unbatched = _detect_batch_size(con_values)

        # Add batch dimension for uniform handling
        if originally_unbatched:
            con_values = jnp.expand_dims(con_values, 1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, 1)
            quad_obj_values = (
                jnp.expand_dims(quad_obj_values, 1) if quad_obj_values is not None else None
            )

        # Extract P values in CSR order
        if self.P_idx is not None and quad_obj_values is not None:
            P_values = quad_obj_values[self.P_idx, :]  # (nnzP, batch)
        else:
            P_values = jnp.zeros((0, batch_size), dtype=jnp.float64)

        # Extract A values in CSR order
        A_values = -con_values[self.A_idx, :]  # (nnzA, batch), negated for Ax + s = b form

        # Extract b vector
        b_start = con_values.shape[0] - self.b_idx.size
        b_raw = con_values[b_start:, :]  # (m, batch)
        # Create full b and scatter at correct indices
        b = jnp.zeros((self.A_shape[0], batch_size), dtype=jnp.float64)
        b = b.at[self.b_idx, :].set(b_raw)

        # Extract q (linear cost)
        q = lin_obj_values[:-1, :]  # (n, batch)

        # Transpose to (batch, dim) format
        P_values = P_values.T  # (batch, nnzP)
        A_values = A_values.T  # (batch, nnzA)
        q = q.T  # (batch, n)
        b = b.T  # (batch, m)

        # Broadcast bounds to batch
        lower = jnp.full((batch_size, self.P_shape[0]), self.lower, dtype=jnp.float64)
        upper = jnp.full((batch_size, self.P_shape[0]), self.upper, dtype=jnp.float64)

        return MOREAU_data_jax(
            P_values=P_values,
            A_values=A_values,
            q=q,
            b=b,
            lower=lower,
            upper=upper,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
            # Structure for solver creation
            P_col_indices=self.P_col_indices,
            P_row_offsets=self.P_row_offsets,
            A_col_indices=self.A_col_indices,
            A_row_offsets=self.A_row_offsets,
            cones=self.cones,
            options=self.options,
            n=self.P_shape[0],
            m=self.A_shape[0],
            # Indices for gradient mapping
            P_idx=self.P_idx,
            A_idx=self.A_idx,
            b_idx=self.b_idx,
            # Sizes for gradient scatter
            P_eval_size=quad_obj_values.shape[0] if quad_obj_values is not None else 0,
            q_eval_size=lin_obj_values.shape[0],
            A_eval_size=con_values.shape[0],
        )


@dataclass
class MOREAU_data:
    """Data class for PyTorch Moreau solver.

    Uses moreau.torch.Solver with three-step API and built-in autograd support.
    """

    P_values: Any  # torch.Tensor (batch, nnzP)
    A_values: Any  # torch.Tensor (batch, nnzA)
    q: Any  # torch.Tensor (batch, n)
    b: Any  # torch.Tensor (batch, m)
    lower: Any  # torch.Tensor (batch, n) - lower bounds
    upper: Any  # torch.Tensor (batch, n) - upper bounds
    batch_size: int
    originally_unbatched: bool
    solver: Any  # moreau.torch.Solver (CPU or CUDA)
    n: int
    m: int
    is_cuda: bool  # Whether tensors are on CUDA

    # Indices for gradient mapping
    P_idx_tensor: Any  # torch.Tensor for scattering P gradients
    A_idx_tensor: Any  # torch.Tensor for scattering A gradients
    b_idx_tensor: Any  # torch.Tensor for scattering b gradients

    # Sizes for gradient scatter
    P_eval_size: int  # Size of P_eval tensor
    q_eval_size: int  # Size of q_eval tensor (n+1)
    A_eval_size: int  # Size of A_eval tensor

    def torch_solve(self, solver_args=None):
        """Solve using moreau.torch.Solver with autograd support.

        Returns (primal, dual, None) and stores backwards_info in self for
        gradient computation. We store it internally rather than returning it
        because returning it as an output would cause PyTorch to update the
        grad_fn of the tensors inside, breaking the autograd chain.
        """
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        # Enable gradients on inputs for Moreau's autograd
        P_values = self.P_values.requires_grad_(True)
        A_values = self.A_values.requires_grad_(True)
        q = self.q.requires_grad_(True)
        b = self.b.requires_grad_(True)

        # Three-step API: setup then solve
        self.solver.setup(P_values, A_values)
        solution = self.solver.solve(q, b, self.lower, self.upper)

        # Extract primal and dual - keep connected to autograd graph (don't detach!)
        primal = solution.x  # (batch, n)
        dual = solution.z  # (batch, m)

        # Store for backward - NOT returned as output to avoid grad_fn corruption
        self._backwards_info = {
            "P_values": P_values,
            "A_values": A_values,
            "q": q,
            "b": b,
            "primal": primal,
            "dual": dual,
        }

        return primal, dual, None

    def torch_derivative(self, dprimal, ddual, _backwards_info_unused):
        """Compute gradients using torch.autograd.grad through Moreau's autograd graph.

        Maps Moreau's gradients (dP_values, dA_values, dq, db) back to
        cvxpylayers format (dP_eval, dq_eval, dA_eval).

        Note: dprimal and ddual already have the batch dimension (1, n) even for
        originally unbatched inputs, because torch_solve() adds the batch dim
        and the autograd graph preserves it.

        The backwards_info parameter is unused - we use self._backwards_info which
        preserves the correct autograd graph (not corrupted by _CvxpyLayer).
        """
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        # Use the internally stored backwards_info (not the parameter which is None)
        backwards_info = self._backwards_info

        # Use torch.autograd.grad to backprop through Moreau's autograd graph
        grads = torch.autograd.grad(
            outputs=[backwards_info["primal"], backwards_info["dual"]],
            inputs=[
                backwards_info["P_values"],
                backwards_info["q"],
                backwards_info["A_values"],
                backwards_info["b"],
            ],
            grad_outputs=[dprimal, ddual],
            allow_unused=True,
        )
        dP_values, dq_moreau, dA_values, db_moreau = grads

        device = dprimal.device
        dtype = torch.float64

        # Scatter P gradients back to P_eval format
        if self.P_eval_size > 0 and dP_values is not None:
            # dP_values is (batch, nnzP), need to scatter to (P_eval_size, batch)
            dP_eval = torch.zeros((self.P_eval_size, self.batch_size), dtype=dtype, device=device)
            dP_eval[self.P_idx_tensor, :] = dP_values.T
        else:
            dP_eval = None

        # Scatter q gradients to q_eval format (q_eval has constant term appended)
        # dq_moreau is (batch, n), need to make (n+1, batch)
        if dq_moreau is not None:
            dq_eval = torch.zeros((self.q_eval_size, self.batch_size), dtype=dtype, device=device)
            dq_eval[:-1, :] = dq_moreau.T
        else:
            dq_eval = torch.zeros((self.q_eval_size, self.batch_size), dtype=dtype, device=device)

        # Scatter A and b gradients to A_eval format
        # A_eval contains: A values (negated), then b values
        dA_eval = torch.zeros((self.A_eval_size, self.batch_size), dtype=dtype, device=device)

        if dA_values is not None:
            # A was negated when extracting, so negate gradient back
            dA_eval[self.A_idx_tensor, :] = -dA_values.T

        if db_moreau is not None:
            # b gradients go to the last portion of A_eval
            b_start = self.A_eval_size - self.b_idx_tensor.shape[0]
            b_section = torch.zeros(
                (self.b_idx_tensor.shape[0], self.batch_size), dtype=dtype, device=device
            )
            b_section[self.b_idx_tensor, :] = db_moreau.T
            dA_eval[b_start:, :] = b_section

        # Remove batch dimension if originally unbatched
        if self.originally_unbatched:
            if dP_eval is not None:
                dP_eval = dP_eval.squeeze(1)
            dq_eval = dq_eval.squeeze(1)
            dA_eval = dA_eval.squeeze(1)

        return dP_eval, dq_eval, dA_eval


@dataclass
class MOREAU_data_jax:
    """Data class for JAX Moreau solver using moreau.jax.Solver."""

    P_values: Any  # jnp.ndarray (batch, nnzP)
    A_values: Any  # jnp.ndarray (batch, nnzA)
    q: Any  # jnp.ndarray (batch, n)
    b: Any  # jnp.ndarray (batch, m)
    lower: Any  # jnp.ndarray (batch, n) - lower bounds
    upper: Any  # jnp.ndarray (batch, n) - upper bounds
    batch_size: int
    originally_unbatched: bool

    # Structure for solver creation
    P_col_indices: np.ndarray
    P_row_offsets: np.ndarray
    A_col_indices: np.ndarray
    A_row_offsets: np.ndarray
    cones: Any  # moreau.Cones
    options: dict
    n: int
    m: int

    # Indices for gradient mapping
    P_idx: np.ndarray | None
    A_idx: np.ndarray
    b_idx: np.ndarray

    # Sizes for gradient scatter
    P_eval_size: int
    q_eval_size: int
    A_eval_size: int

    def jax_solve(self, solver_args=None):
        """Solve using moreau.jax.Solver with native custom_vjp gradients."""
        if jnp is None:
            raise ImportError("JAX interface requires 'jax' package. Install with: pip install jax")
        if moreau_jax is None:
            raise ImportError("Moreau JAX interface requires 'moreau' package with JAX support.")

        # Create settings
        settings = moreau.Settings(enable_grad=True)
        for key, value in self.options.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        # Create JAX solver (stateless, so we create fresh each time)
        solver = moreau_jax.Solver(
            n=self.n,
            m=self.m,
            P_row_offsets=self.P_row_offsets,
            P_col_indices=self.P_col_indices,
            A_row_offsets=self.A_row_offsets,
            A_col_indices=self.A_col_indices,
            cones=self.cones,
            settings=settings,
        )

        # Solve - moreau.jax.Solver returns JaxSolution with x, z, s, etc.
        solution = solver.solve(
            self.P_values, self.A_values, self.q, self.b, self.lower, self.upper
        )

        primal = solution.x  # (batch, n)
        dual = solution.z  # (batch, m)

        # Store info needed for gradient mapping
        backwards_info = {
            "solver": solver,
            "P_values": self.P_values,
            "A_values": self.A_values,
            "q": self.q,
            "b": self.b,
            "lower": self.lower,
            "upper": self.upper,
        }

        return primal, dual, backwards_info

    def jax_derivative(self, dprimal, ddual, backwards_info):
        """Compute gradients using JAX autodiff on Moreau's solve.

        Maps Moreau's gradients back to cvxpylayers format.
        """
        if jnp is None:
            raise ImportError("JAX interface requires 'jax' package. Install with: pip install jax")

        # Handle batch dimension
        if self.originally_unbatched:
            dprimal = jnp.expand_dims(dprimal, 0)
            ddual = jnp.expand_dims(ddual, 0)

        # Define loss function that extracts x and z from solution
        def solve_fn(P_vals, A_vals, q, b):
            solver = backwards_info["solver"]
            solution = solver.solve(
                P_vals, A_vals, q, b, backwards_info["lower"], backwards_info["upper"]
            )
            return solution.x, solution.z

        # Compute vector-Jacobian products using JAX
        _, vjp_fn = jax.vjp(
            solve_fn,
            backwards_info["P_values"],
            backwards_info["A_values"],
            backwards_info["q"],
            backwards_info["b"],
        )

        dP_values, dA_values, dq_moreau, db_moreau = vjp_fn((dprimal, ddual))

        # Scatter P gradients back to P_eval format
        if self.P_eval_size > 0 and self.P_idx is not None:
            dP_eval = jnp.zeros((self.P_eval_size, self.batch_size), dtype=jnp.float64)
            dP_eval = dP_eval.at[self.P_idx, :].set(dP_values.T)
        else:
            dP_eval = None

        # Scatter q gradients to q_eval format
        dq_eval = jnp.zeros((self.q_eval_size, self.batch_size), dtype=jnp.float64)
        dq_eval = dq_eval.at[:-1, :].set(dq_moreau.T)

        # Scatter A and b gradients to A_eval format
        dA_eval = jnp.zeros((self.A_eval_size, self.batch_size), dtype=jnp.float64)
        dA_eval = dA_eval.at[self.A_idx, :].set(-dA_values.T)  # Negate back

        b_start = self.A_eval_size - self.b_idx.size
        b_section = jnp.zeros((self.b_idx.size, self.batch_size), dtype=jnp.float64)
        b_section = b_section.at[self.b_idx, :].set(db_moreau.T)
        dA_eval = dA_eval.at[b_start:, :].set(b_section)

        # Remove batch dimension if originally unbatched
        if self.originally_unbatched:
            if dP_eval is not None:
                dP_eval = jnp.squeeze(dP_eval, 1)
            dq_eval = jnp.squeeze(dq_eval, 1)
            dA_eval = jnp.squeeze(dA_eval, 1)

        return dP_eval, dq_eval, dA_eval
