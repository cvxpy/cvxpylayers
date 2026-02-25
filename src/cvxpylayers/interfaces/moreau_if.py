"""Moreau solver interface for CVXPYLayers.

Moreau is a conic optimization solver that solves problems of the form:
    minimize    (1/2)x'Px + q'x
    subject to  Ax + s = b
                s in K

where K is a product of cones.

Uses Moreau's native PyTorch and JAX solvers with built-in automatic differentiation:
- PyTorch: moreau.torch.Solver with two-step API (setup + solve) and autograd support
- JAX: moreau.jax.Solver with custom_vjp for gradients

Limitations:
- Only 3D second-order cones are supported (Moreau assumes all SOCs have dimension 3).
- SOC dual variables are not supported; requesting them raises ValueError.
- Not thread-safe: solver instances are lazily initialized and cached on MOREAU_ctx.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

# Optional dependencies — each may be absent independently
try:
    import moreau
    import moreau.jax as moreau_jax
    import moreau.torch as moreau_torch
except ImportError:
    moreau = moreau_torch = moreau_jax = None  # type: ignore[assignment]

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = jnp = None  # type: ignore[assignment]

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
            needs_grad: bool = True,
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

    Raises:
        ValueError: If any second-order cone has dimension != 3. Moreau assumes
            all SOCs are 3-dimensional (the ``num_so_cones`` API only stores
            a count, not per-cone sizes).
    """
    if moreau is None:
        raise ImportError(
            "Moreau solver requires 'moreau' package. Install with: pip install moreau"
        )

    # Second-order cones: moreau now uses num_so_cones (count) instead of soc_dims (list)
    # Each SOC in moreau is assumed to be dimension 3
    soc_dims = dims.get("q", [])
    if soc_dims:
        # Verify all SOCs are dimension 3 (moreau's assumption)
        for i, dim in enumerate(soc_dims):
            if dim != 3:
                raise ValueError(
                    f"Moreau only supports 3D second-order cones, but SOC {i} has dimension {dim}. "
                    "Consider using a different solver for problems with non-3D SOCs."
                )

    cones = moreau.Cones(
        num_zero_cones=dims.get("z", 0),
        num_nonneg_cones=dims.get("l", 0),
        num_so_cones=len(soc_dims),
        num_exp_cones=dims.get("ep", 0),
        power_alphas=list(dims.get("p", [])),
    )

    return cones


class MOREAU_ctx:
    """Context class for Moreau solver.

    Stores problem structure (CSR format) and creates solvers with lazy batch init.
    Batch size is inferred from inputs at solve time (moreau handles auto-reset).

    CSR conversion and parametrization matrix row permutation are performed
    externally by ``get_solver_ctx`` before this class is constructed.
    """

    P_col_indices: np.ndarray
    P_row_offsets: np.ndarray
    P_shape: tuple[int, int]

    A_col_indices: np.ndarray
    A_row_offsets: np.ndarray
    A_shape: tuple[int, int]
    b_idx: np.ndarray

    cones: Any  # moreau.Cones (unified for NumPy and PyTorch)
    dims: dict

    def __init__(
        self,
        csr,
        dims,
        options=None,
        *,
        reduced_P_mat=None,
        reduced_A_mat=None,
    ):
        """Initialize Moreau solver context.

        Args:
            csr: ``CsrProblemData`` with pre-computed CSR structure.  The
                caller (``get_solver_ctx``) has already permuted the
                parametrization matrix rows so ``matrix @ params`` produces
                values in CSR order.
            dims: Cone dimensions dictionary.
            options: Solver options forwarded to ``moreau.Settings``.
            reduced_P_mat: Already-permuted sparse parametrization matrix for P.
                When ``reduced_P_mat[:, :-1].nnz == 0`` (and same for A), P and A
                are constant across parameter values, enabling a one-time
                ``setup()`` call (the ``PA_is_constant`` optimisation).
            reduced_A_mat: Already-permuted sparse parametrization matrix for A.
        """
        # Store CSR structure (handle None P for LP problems)
        if csr.P_csr_structure is not None:
            self.P_col_indices = csr.P_csr_structure[0].astype(np.int64)
            self.P_row_offsets = csr.P_csr_structure[1].astype(np.int64)
        else:
            n = csr.P_shape[0]
            self.P_col_indices = np.array([], dtype=np.int64)
            self.P_row_offsets = np.zeros(n + 1, dtype=np.int64)
        self.P_shape = csr.P_shape

        self.A_col_indices = csr.A_csr_structure[0].astype(np.int64)
        self.A_row_offsets = csr.A_csr_structure[1].astype(np.int64)
        self.A_shape = csr.A_shape
        self.b_idx = csr.b_idx

        # Number of non-zeros in P and A (used for slicing in forward/backward)
        self.nnz_P = csr.nnz_P
        self.nnz_A = csr.nnz_A

        # Store dimensions
        self.dims = dims
        self.options = options or {}

        # Create cones and solver lazily
        self._cones = None
        self._torch_solver_cuda = None  # CUDA solver (moreau.torch.Solver) with lazy init
        self._torch_solver_cpu = None  # CPU solver (moreau.torch.Solver) with lazy init
        self._jax_solver = None  # JAX solver (moreau.jax.Solver) with lazy init

        # Detect if P and A are constant (don't depend on any parameters)
        # Matrices are parametrized: each column corresponds to a parameter, last column is constant
        # If all non-constant columns are zero, the matrix values are fixed
        self.PA_is_constant = (
            (reduced_P_mat is None or reduced_P_mat[:, :-1].nnz == 0)
            and reduced_A_mat is not None
            and reduced_A_mat[:, :-1].nnz == 0
        )

        if self.PA_is_constant:
            # Pre-extract constant values (last column only, already in CSR order
            # thanks to the row permutation applied by get_solver_ctx)
            if reduced_P_mat is not None:
                P_csr = reduced_P_mat[:, -1].tocsr()
                self._P_const_values = P_csr.data
            else:
                self._P_const_values = np.array([], dtype=np.float64)
            A_csr = reduced_A_mat[:, -1].tocsr()
            # Only take A values (first nnz_A), not b values; negate for Ax + s = b form
            self._A_const_values = -A_csr.data[: self.nnz_A]
        else:
            self._P_const_values = None
            self._A_const_values = None

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

    def _create_torch_solver(self, device: str):
        """Create a moreau.torch.Solver for the specified device.

        Called lazily on first use. If P/A are constant, also calls setup().
        """
        if moreau_torch is None:
            raise ImportError(
                "Moreau solver requires 'moreau' package. Install with: pip install moreau"
            )
        if device == "cuda" and not moreau.device_available("cuda"):
            raise ImportError(
                "Moreau CUDA backend requires 'moreau' package with CUDA support. "
                "Install with: pip install moreau[cuda]"
            )

        settings = self._get_settings(enable_grad=True)
        settings.device = device
        solver = moreau_torch.Solver(
            n=self.P_shape[0],
            m=self.A_shape[0],
            P_row_offsets=torch.tensor(self.P_row_offsets, dtype=torch.int64),
            P_col_indices=torch.tensor(self.P_col_indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(self.A_row_offsets, dtype=torch.int64),
            A_col_indices=torch.tensor(self.A_col_indices, dtype=torch.int64),
            cones=self.cones,
            settings=settings,
        )

        # If P and A are constant, do setup once now (expensive factorization)
        if self.PA_is_constant:
            P_values = torch.tensor(
                self._P_const_values, dtype=torch.float64, device=device
            ).unsqueeze(0)
            A_values = torch.tensor(
                self._A_const_values, dtype=torch.float64, device=device
            ).unsqueeze(0)
            solver.setup(P_values, A_values)

        return solver

    def get_torch_solver(self, device: str):
        """Get moreau.torch.Solver for the specified device (lazy init).

        Args:
            device: 'cuda' or 'cpu'

        Returns:
            moreau.torch.Solver configured for the specified device with enable_grad=True
        """
        if device == "cuda":
            if self._torch_solver_cuda is None:
                self._torch_solver_cuda = self._create_torch_solver("cuda")
            return self._torch_solver_cuda
        else:
            if self._torch_solver_cpu is None:
                self._torch_solver_cpu = self._create_torch_solver("cpu")
            return self._torch_solver_cpu

    def _create_jax_solver(self):
        """Create a moreau.jax.Solver. Called lazily on first use."""
        if moreau_jax is None:
            raise ImportError("Moreau JAX interface requires 'moreau' package with JAX support.")

        settings = self._get_settings(enable_grad=True)
        solver = moreau_jax.Solver(
            n=self.P_shape[0],
            m=self.A_shape[0],
            P_row_offsets=self.P_row_offsets,
            P_col_indices=self.P_col_indices,
            A_row_offsets=self.A_row_offsets,
            A_col_indices=self.A_col_indices,
            cones=self.cones,
            settings=settings,
        )

        # For simplicity, we use the 4-arg jax solve in all cases.

        return solver

    def get_jax_solver(self):
        """Get moreau.jax.Solver (lazy init)."""
        if self._jax_solver is None:
            self._jax_solver = self._create_jax_solver()
        return self._jax_solver

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

        # Extract values using slicing (stays on GPU if input is on GPU).
        # Parametrization matrices have been pre-permuted so that matrix
        # multiplication produces values directly in CSR order.
        # con_values shape: (nnz_A + nb, batch) — first nnz_A are A in CSR order
        # lin_obj_values shape: (n+1, batch) - last entry is constant term

        # P values are already in CSR order from pre-permuted matrix multiply
        if quad_obj_values is not None:
            P_values = quad_obj_values  # (nnzP, batch), already CSR-ordered
        else:
            # Empty P matrix (LP problem)
            P_values = torch.zeros((0, batch_size), dtype=torch.float64, device=device)

        # A values: first nnz_A entries are in CSR order, negated for Ax + s = b form
        A_values = -con_values[: self.nnz_A, :]  # (nnzA, batch)

        # b vector: entries after the A values
        b_idx_tensor = torch.tensor(self.b_idx, dtype=torch.long, device=device)
        b_raw = con_values[self.nnz_A :, :]  # (nb, batch)
        # Scatter into full b tensor (use non-in-place scatter to preserve autograd)
        b_idx_expanded = b_idx_tensor.unsqueeze(1).expand(-1, batch_size)
        b = torch.zeros(
            (self.A_shape[0], batch_size), dtype=torch.float64, device=device
        ).scatter(0, b_idx_expanded, b_raw.to(dtype=torch.float64))

        # Extract q (linear cost)
        q = lin_obj_values[:-1, :]  # (n, batch), exclude constant term

        # Transpose to (batch, dim) format for Moreau
        # Use .to() which is a no-op if already on correct device/dtype (zero-copy for CUDA)
        P_values = P_values.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, nnzP)
        A_values = A_values.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, nnzA)
        q = q.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, n)
        b = b.T.contiguous().to(device=device, dtype=torch.float64)  # (batch, m)

        # Select solver based on device
        solver = self.get_torch_solver("cuda" if is_cuda else "cpu")

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
            # Store b indices for gradient mapping (P/A no longer need shuffle)
            b_idx_tensor=b_idx_tensor,
            # Store shapes for gradient scatter
            P_eval_size=quad_obj_values.shape[0] if quad_obj_values is not None else 0,
            q_eval_size=lin_obj_values.shape[0],
            A_eval_size=con_values.shape[0],
            # Whether setup() was already called (P/A constant optimization)
            setup_cached=self.PA_is_constant,
        )

@dataclass
class MOREAU_data:
    """Data class for PyTorch Moreau solver.

    Uses moreau.torch.Solver with two-step API (setup + solve) and built-in autograd support.
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
    is_cuda: bool  # Whether tensors are on CUDA

    # Index for b gradient mapping (P/A no longer need shuffle indices)
    b_idx_tensor: Any  # torch.Tensor for scattering b gradients

    # Sizes for gradient scatter
    P_eval_size: int  # Size of P_eval tensor (= nnz_P)
    q_eval_size: int  # Size of q_eval tensor (n+1)
    A_eval_size: int  # Size of A_eval tensor (= nnz_A + nb)

    # Whether setup() was already called in get_torch_solver (P/A constant case)
    setup_cached: bool = False

    def torch_solve(self, solver_args=None):
        """Solve using moreau.torch.Solver with autograd support.

        Returns (primal, dual, None) and stores backwards_info in self for
        gradient computation. We store it internally rather than returning it
        because returning it as an output would cause PyTorch to update the
        grad_fn of the tensors inside, breaking the autograd chain.

        Args:
            solver_args: Optional dict of solver settings to override for this call.
                Supports 'verbose', 'max_iter', 'time_limit', and other moreau.Settings fields.
        """
        if torch is None:
            raise ImportError(
                "PyTorch interface requires 'torch' package. Install with: pip install torch"
            )

        # Apply per-call solver_args by updating the solver's internal settings
        if solver_args:
            settings = self.solver._impl._settings
            for key, value in solver_args.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

        # Enable gradients on inputs for Moreau's autograd
        q = self.q.requires_grad_(True)
        b = self.b.requires_grad_(True)

        # Setup is expensive - skip if already done with constant P/A
        if self.setup_cached:
            # P and A are constant; setup was called once during solver creation
            P_values = self.P_values
            A_values = self.A_values
        else:
            # P and/or A depend on parameters; need setup each call
            P_values = self.P_values.requires_grad_(True)
            A_values = self.A_values.requires_grad_(True)
            self.solver.setup(P_values, A_values)

        solution = self.solver.solve(q, b)

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
        self._backwards_info = None  # Free memory after extraction

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
        dP_values, dq_raw, dA_values, db_raw = grads

        device = dprimal.device
        dtype = torch.float64

        # P gradients: directly transpose (no scatter needed, matrices are pre-permuted)
        if self.P_eval_size > 0 and dP_values is not None:
            # dP_values is (batch, nnzP), transpose to (nnzP, batch) = (P_eval_size, batch)
            dP_eval = dP_values.T
        else:
            dP_eval = None

        # Scatter q gradients to q_eval format (q_eval has constant term appended)
        # dq_raw is (batch, n), need to make (n+1, batch)
        if dq_raw is not None:
            dq_eval = torch.zeros((self.q_eval_size, self.batch_size), dtype=dtype, device=device)
            dq_eval[:-1, :] = dq_raw.T
        else:
            dq_eval = torch.zeros((self.q_eval_size, self.batch_size), dtype=dtype, device=device)

        # A and b gradients to A_eval format
        # A_eval layout (after pre-permutation): [A values in CSR order, b values]
        nnz_A = self.A_eval_size - self.b_idx_tensor.shape[0]
        dA_eval = torch.zeros((self.A_eval_size, self.batch_size), dtype=dtype, device=device)

        if dA_values is not None:
            # A was negated when extracting, so negate gradient back
            dA_eval[:nnz_A, :] = -dA_values.T

        if db_raw is not None:
            # b gradients go to the last portion of A_eval
            # Gather gradients from positions b_idx (backward of scatter)
            b_section = db_raw[:, self.b_idx_tensor].T  # (b_idx.size, batch)
            dA_eval[nnz_A:, :] = b_section

        # Remove batch dimension if originally unbatched
        if self.originally_unbatched:
            if dP_eval is not None:
                dP_eval = dP_eval.squeeze(1)
            dq_eval = dq_eval.squeeze(1)
            dA_eval = dA_eval.squeeze(1)

        return dP_eval, dq_eval, dA_eval
