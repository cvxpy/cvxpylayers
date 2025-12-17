from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

try:
    import jax
    import jax.experimental.sparse
    import jax.numpy as jnp
    import mpax
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    mpax = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import cvxpylayers.utils.parse_args as pa

    # Type alias for multi-framework tensor types
    TensorLike = torch.Tensor | jnp.ndarray | np.ndarray | mx.array
else:
    TensorLike = Any


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
            solver_ctx = cl_ctx.solver_ctx

            # Convert torch to jax
            quad_obj_values = jnp.array(P_eval) if P_eval is not None else None
            lin_obj_values = jnp.array(q_eval)
            con_values = jnp.array(A_eval)

            # Detect batch size
            if con_values.ndim == 1:
                originally_unbatched = True
                batch_size = 1
                con_values = jnp.expand_dims(con_values, axis=1)
                lin_obj_values = jnp.expand_dims(lin_obj_values, axis=1)
                quad_obj_values = jnp.expand_dims(quad_obj_values, axis=1) if quad_obj_values is not None else None
            else:
                originally_unbatched = False
                batch_size = con_values.shape[1]

            if solver_args is None:
                solver_args = {}

            initial_primal = solver_args.get("initial_primal_solution", None)
            initial_dual = solver_args.get("initial_dual_solution", None)

            def solve_single_batch(quad_obj_vals_i, lin_obj_vals_i, con_vals_i):
                return _build_and_solve_qp(
                    quad_obj_vals_i,
                    lin_obj_vals_i,
                    con_vals_i,
                    solver_ctx,
                    initial_primal,
                    initial_dual,
                )

            solve_batched = jax.vmap(solve_single_batch, in_axes=(1, 1, 1))

            def batched_solver(quad_vals, lin_vals, con_vals):
                return solve_batched(quad_vals, lin_vals, con_vals)

            (primal, dual), vjp_fun = jax.vjp(
                batched_solver,
                quad_obj_values,
                lin_obj_values,
                con_values,
            )

            primal_torch = torch.utils.dlpack.from_dlpack(primal)
            dual_torch = torch.utils.dlpack.from_dlpack(dual)

            return primal_torch, dual_torch, vjp_fun, None

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
            pass

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, dprimal: torch.Tensor, ddual: torch.Tensor, _vjp: Any, _: Any
        ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None, None]:
            raise NotImplementedError(
                "Backward pass is not implemented for MPAX solver. "
                "Use solver='DIFFCP' for differentiable optimization layers."
            )


def _parse_objective_structure(
    objective_structure: tuple,
) -> tuple[slice, np.ndarray, tuple[np.ndarray, np.ndarray],
           tuple[int, int], int]:
    """Parse objective structure to extract quadratic (Q) matrix components.

    Converts CVXPY's canonical objective structure
    into sparse matrix components
    for the quadratic cost matrix Q in the QP formulation.

    Args:
        objective_structure: Tuple of (indices, indptr, (n, n)) from CVXPY

    Returns:
        Tuple of (c_slice, Q_idxs, Q_structure, Q_shape, n) where:
            - c_slice: Slice for linear cost vector c
            - Q_idxs: Data indices for Q matrix values
            - Q_structure: (indices, indptr) for Q sparse structure
            - Q_shape: Shape (n, n) of Q matrix
            - n: Number of primal variables
    """
    obj_indices, obj_ptr, (n, _) = objective_structure
    c_slice = slice(0, n)

    # Convert to CSR format for efficient row access
    obj_csr = sp.csc_array(
        (np.arange(obj_indices.size), obj_indices, obj_ptr),
        shape=(n, n),
    ).tocsr()

    Q_idxs = obj_csr.data
    Q_structure = obj_csr.indices, obj_csr.indptr
    Q_shape = (n, n)

    return c_slice, Q_idxs, Q_structure, Q_shape, n


def _initialize_solver(options:
                       dict[str, Any] | None) -> tuple[Callable, bool]:
    """Initialize MPAX solver based on options.

    Args:
        options: Solver options dictionary containing:
            - warm_start: Whether to use warm
            starting (currently must be False)
            - algorithm: "raPDHG" or "r2HPDHG"
            - Additional solver-specific options

    Returns:
        Tuple of (jitted_solver_fn, warm_start_flag)

    Raises:
        ValueError: If algorithm is not "raPDHG" or "r2HPDHG"
    """
    if options is None:
        options = {}

    warm_start = options.pop("warm_start", False)
    assert warm_start is False

    algorithm = options.pop("algorithm", "raPDHG")

    if algorithm == "raPDHG":
        alg = mpax.raPDHG
    elif algorithm == "r2HPDHG":
        alg = mpax.r2HPDHG
    else:
        raise ValueError("Invalid MPAX algorithm")

    solver = alg(warm_start=warm_start, unroll=True, **options)
    return jax.jit(solver.optimize), warm_start


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

    lower: jnp.ndarray
    upper: jnp.ndarray

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
        if mpax is None or jax is None:
            raise ImportError(
                "MPAX solver requires 'mpax' and 'jax' "
                "packages to be installed. "
                "Install with: pip install mpax jax"
            )

        # Parse objective structure
        self.c_slice, self.Q_idxs, self.Q_structure, self.Q_shape, n = (
            _parse_objective_structure(
                objective_structure)
        )

        # Parse constraint structure - splits into equality
        # (A) and inequality (G) matrices
        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1

        # Extract indices for the last column
        # (which contains b and h RHS values)
        # Use indices instead of slices because
        # sparse matrices may have reduced out
        # explicit zeros, so we need to reconstruct the full dense vectors
        self.last_col_start = con_ptr[-2]
        self.last_col_end = con_ptr[-1]
        self.last_col_indices = con_indices[
            self.last_col_start:self.last_col_end
        ]
        self.m = m  # Total number of constraint rows

        # Convert to CSR format for row-based splitting
        con_csr = sp.csc_array(
            (np.arange(con_indices.size), con_indices, con_ptr[:-1]),
            shape=(m, n),
        ).tocsr()
        # Split point between equality and inequality

        split = con_csr.indptr[dims.zero]
        # Extract equality constraints (A)
        self.A_idxs = con_csr.data[:split]
        self.A_structure = con_csr.indices[:split], \
            con_csr.indptr[: dims.zero + 1]
        self.A_shape = (dims.zero, n)

        # Extract inequality constraints (G)
        self.G_idxs = con_csr.data[split:]
        self.G_structure = con_csr.indices[split:], \
            con_csr.indptr[dims.zero:] - split
        self.G_shape = (m - dims.zero, n)

        # Precompute split_at to avoid binary search on every solve
        self.split_at = int(jnp.searchsorted(self.last_col_indices, dims.zero))

        # Set bounds
        self.lower = (
            lower_bounds
            if lower_bounds is not None
            else -jnp.inf * jnp.ones(n)
        )
        self.upper = (
            upper_bounds
            if upper_bounds is not None
            else jnp.inf * jnp.ones(n)
        )

        # Initialize solver
        self.solver, self.warm_start = _initialize_solver(options)

    def jax_to_data(
        self,
        quad_obj_values,
        lin_obj_values,
        con_values,
    ):
        # Detect batch size and whether input was originally unbatched
        if con_values.ndim == 1:
            originally_unbatched = True
            batch_size = 1
            # Add batch dimension for uniform handling
            con_values = jnp.expand_dims(con_values, axis=1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, axis=1)
            quad_obj_values = jnp.expand_dims(quad_obj_values, axis=1)
        else:
            originally_unbatched = False
            batch_size = con_values.shape[1]

        return MPAX_data(
            ctx=self,
            quad_obj_values=quad_obj_values,
            lin_obj_values=lin_obj_values,
            con_values=con_values,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )

    def mlx_to_data(
        self,
        quad_obj_values,
        lin_obj_values,
        con_values,
    ) -> "MPAX_data":

        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        # Detect batch size and whether input was originally unbatched
        if con_values.ndim == 1:
            originally_unbatched = True
            batch_size = 1
            # Add batch dimension for uniform handling
            con_values = mx.expand_dims(con_values, axis=1)
            lin_obj_values = mx.expand_dims(lin_obj_values, axis=1)
            quad_obj_values = mx.expand_dims(quad_obj_values, axis=1)
        else:
            originally_unbatched = False
            batch_size = con_values.shape[1]

        # Convert to JAX arrays for MPAX (MPAX uses JAX internally)
        # MLX arrays can be converted to numpy first, then to JAX
        import jax.numpy as jnp
        quad_obj_values_jax = jnp.array(np.array(quad_obj_values))
        lin_obj_values_jax = jnp.array(np.array(lin_obj_values))
        con_values_jax = jnp.array(np.array(con_values))

        return MPAX_data(
            ctx=self,
            quad_obj_values=quad_obj_values_jax,
            lin_obj_values=lin_obj_values_jax,
            con_values=con_values_jax,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )


def _extract_rhs_vectors(
    con_vals_i: jnp.ndarray, ctx: "MPAX_ctx"
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract and reconstruct b and h right-hand-side vectors from constraint values.

    CVXPY stores RHS values sparsely in the last column. This reconstructs
    dense b (equality) and h (inequality) vectors from the sparse representation.

    Args:
        con_vals_i: Constraint coefficient values for single batch element
        ctx: MPAX context with structure information

    Returns:
        Tuple of (b_vals, h_vals) where:
            - b_vals: Dense equality constraint RHS vector
            - h_vals: Dense inequality constraint RHS vector
    """
    # Extract sparse RHS values from last column
    rhs_sparse_values = con_vals_i[ctx.last_col_start: ctx.last_col_end]
    rhs_row_indices = ctx.last_col_indices

    num_eq_constraints = ctx.A_shape[0]
    num_ineq_constraints = ctx.G_shape[0]

    # Split sparse values between equality (b) and inequality (h) constraints
    split_at = ctx.split_at  # Precomputed split index

    b_row_indices = rhs_row_indices[:split_at]
    b_sparse_values = rhs_sparse_values[:split_at]

    h_row_indices = rhs_row_indices[split_at:] - num_eq_constraints
    h_sparse_values = rhs_sparse_values[split_at:]

    # Reconstruct dense vectors from sparse representation
    b_vals = jnp.zeros(num_eq_constraints)
    h_vals = jnp.zeros(num_ineq_constraints)

    # Note: Negation matches MPAX's sign convention
    b_vals = b_vals.at[b_row_indices].set(-b_sparse_values)
    h_vals = h_vals.at[h_row_indices].set(-h_sparse_values)

    return b_vals, h_vals


def _build_and_solve_qp(
    quad_obj_vals_i: jnp.ndarray,
    lin_obj_vals_i: jnp.ndarray,
    con_vals_i: jnp.ndarray,
    ctx: "MPAX_ctx",
    initial_primal: jnp.ndarray | None,
    initial_dual: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build and solve a quadratic program for a single batch element.

    Constructs an MPAX QP model from parameter values and solves it using
    the precompiled solver.

    Args:
        quad_obj_vals_i: Quadratic objective coefficient values
        lin_obj_vals_i: Linear objective coefficient values
        con_vals_i: Constraint coefficient values
        ctx: MPAX context with problem structure
        initial_primal: Optional warm-start primal solution
        initial_dual: Optional warm-start dual solution

    Returns:
        Tuple of (primal_solution, dual_solution)
    """
    # Extract RHS values and reconstruct b and h vectors
    b_vals, h_vals = _extract_rhs_vectors(con_vals_i, ctx)

    # Build QP model: minimize (1/2)x^T Q x + c^T x subject to Ax = b, Gx <= h, l <= x <= u
    model = mpax.create_qp(
        jax.experimental.sparse.BCSR(
            (quad_obj_vals_i[ctx.Q_idxs], *ctx.Q_structure),
            shape=ctx.Q_shape,
        ),
        lin_obj_vals_i[ctx.c_slice],
        jax.experimental.sparse.BCSR(
            (con_vals_i[ctx.A_idxs], *ctx.A_structure),
            shape=ctx.A_shape,
        ),
        b_vals,
        jax.experimental.sparse.BCSR(
            (con_vals_i[ctx.G_idxs], *ctx.G_structure),
            shape=ctx.G_shape,
        ),
        h_vals,
        ctx.lower,
        ctx.upper,
    )

    # Solve with optional warm start
    solution = ctx.solver(
        model,
        initial_primal_solution=initial_primal,
        initial_dual_solution=initial_dual,
    )
    return solution.primal_solution, solution.dual_solution


@dataclass
class MPAX_data:
    ctx: "MPAX_ctx"  # Reference to context with structure info
    quad_obj_values: jnp.ndarray  # Shape: (n_Q,) or (n_Q, batch_size)
    lin_obj_values: jnp.ndarray  # Shape: (n,) or (n, batch_size)
    con_values: jnp.ndarray  # Shape: (n_con,) or (n_con, batch_size)
    batch_size: int
    originally_unbatched: bool

    def jax_solve(self, solver_args=None):
        if solver_args is None:
            solver_args = {}

        # Extract warm start options if provided
        initial_primal = solver_args.get("initial_primal_solution", None)
        initial_dual = solver_args.get("initial_dual_solution", None)

        def solve_single_batch(quad_obj_vals_i, lin_obj_vals_i, con_vals_i):
            """Build model and solve for a single batch element."""
            return _build_and_solve_qp(
                quad_obj_vals_i,
                lin_obj_vals_i,
                con_vals_i,
                self.ctx,
                initial_primal,
                initial_dual,
            )

        # Vectorize over batch dimension (axis 1 of parameter arrays)
        solve_batched = jax.vmap(solve_single_batch, in_axes=(1, 1, 1))

        def batched_solver(quad_vals, lin_vals, con_vals):
            return solve_batched(quad_vals, lin_vals, con_vals)

        # Compute forward pass and VJP function
        (primal, dual), vjp_fun = jax.vjp(
            batched_solver,
            self.quad_obj_values,
            self.lin_obj_values,
            self.con_values,
        )

        return primal, dual, vjp_fun

    def mlx_solve(self, solver_args=None):
        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        # Use JAX solve (MPAX uses JAX internally)
        primal_jax, dual_jax, vjp_fun = self.jax_solve(solver_args)

        # Convert JAX arrays to MLX arrays
        primal = mx.array(np.array(primal_jax), dtype=mx.float64)
        dual = mx.array(np.array(dual_jax), dtype=mx.float64)

        return primal, dual, vjp_fun

    def jax_derivative(self, dprimal, ddual, vjp_fun):
        # vjp_fun takes output cotangents (dprimal, ddual) and returns
        # input cotangents (d_quad_obj, d_lin_obj, d_con)
        dQ, dq, dA = vjp_fun((dprimal, ddual))

        if self.originally_unbatched:
            dq = jnp.squeeze(dq, 1)
            dA = jnp.squeeze(dA, 1)
            dQ = jnp.squeeze(dQ, 1) if dQ is not None else None

        return dQ, dq, dA

    def mlx_derivative(self, primal, dual, adj_batch):

        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        # Convert MLX arrays to JAX arrays for derivative computation
        import jax.numpy as jnp
        primal_jax = jnp.array(np.array(primal))
        dual_jax = jnp.array(np.array(dual))

        # Compute derivatives using JAX
        quad, lin, con = self.jax_derivative(primal_jax, dual_jax, adj_batch)

        # Convert back to MLX
        quad_mlx = mx.array(np.array(quad), dtype=mx.float64) if quad is not None else None
        lin_mlx = mx.array(np.array(lin), dtype=mx.float64)
        con_mlx = mx.array(np.array(con), dtype=mx.float64)

        return quad_mlx, lin_mlx, con_mlx
