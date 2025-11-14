from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:
    import jax
    import jax.experimental.sparse
    import jax.numpy as jnp
    import mpax
except ImportError:
    pass


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
        obj_indices, obj_ptr, (n, _) = objective_structure
        self.c_slice = slice(0, n)
        obj_csr = sp.csc_array(
            (np.arange(obj_indices.size), obj_indices, obj_ptr),
            shape=(n, n),
        ).tocsr()
        self.Q_idxs = obj_csr.data
        self.Q_structure = obj_csr.indices, obj_csr.indptr
        self.Q_shape = (n, n)

        con_indices, con_ptr, (m, np1) = constraint_structure
        assert np1 == n + 1

        # Extract indices for the last column (which contains b and h values)
        # Use indices instead of slices because sparse matrices may have reduced out
        # explicit zeros, so we need to reconstruct the full dense vectors
        self.last_col_start = con_ptr[-2]
        self.last_col_end = con_ptr[-1]
        self.last_col_indices = con_indices[self.last_col_start : self.last_col_end]
        self.m = m  # Total number of constraint rows

        con_csr = sp.csc_array(
            (np.arange(con_indices.size), con_indices, con_ptr[:-1]),
            shape=(m, n),
        ).tocsr()
        split = con_csr.indptr[dims.zero]

        self.A_idxs = con_csr.data[:split]
        self.A_structure = con_csr.indices[:split], con_csr.indptr[: dims.zero + 1]
        self.A_shape = (dims.zero, n)

        self.G_idxs = con_csr.data[split:]
        self.G_structure = con_csr.indices[split:], con_csr.indptr[dims.zero :] - split
        self.G_shape = (m - dims.zero, n)

        self.lower = lower_bounds if lower_bounds is not None else -jnp.inf * jnp.ones(n)
        self.upper = upper_bounds if upper_bounds is not None else jnp.inf * jnp.ones(n)

        # Precompute split_at to avoid binary search on every solve
        self.split_at = int(jnp.searchsorted(self.last_col_indices, dims.zero))

        if options is None:
            options = {}
        self.warm_start = options.pop("warm_start", False)
        assert self.warm_start is False
        algorithm = options.pop("algorithm", "raPDHG")

        if algorithm == "raPDHG":
            alg = mpax.raPDHG
        elif algorithm == "r2HPDHG":
            alg = mpax.r2HPDHG
        else:
            raise ValueError("Invalid MPAX algorithm")
        solver = alg(warm_start=self.warm_start, **options)
        self.solver = jax.jit(solver.optimize)

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

    def torch_to_data(
        self,
        quad_obj_values,
        lin_obj_values,
        con_values,
    ) -> "MPAX_data":
        return self.jax_to_data(
            jnp.array(quad_obj_values),
            jnp.array(lin_obj_values),
            jnp.array(con_values),
        )


@dataclass
class MPAX_data:
    ctx: "MPAX_ctx"  # Reference to context with structure info
    quad_obj_values: jnp.ndarray  # Shape: (n_Q,) or (n_Q, batch_size)
    lin_obj_values: jnp.ndarray  # Shape: (n,) or (n, batch_size)
    con_values: jnp.ndarray  # Shape: (n_con,) or (n_con, batch_size)
    batch_size: int
    originally_unbatched: bool

    def jax_solve(self):
        def solve_single_batch(quad_obj_vals_i, lin_obj_vals_i, con_vals_i):
            """Build model and solve for a single batch element."""
            # Extract RHS values and reconstruct b and h vectors
            # (same logic as old jax_to_data, but for single batch element)
            rhs_sparse_values = con_vals_i[self.ctx.last_col_start : self.ctx.last_col_end]
            rhs_row_indices = self.ctx.last_col_indices

            num_eq_constraints = self.ctx.A_shape[0]
            num_ineq_constraints = self.ctx.G_shape[0]

            # Use precomputed split_at from context
            split_at = self.ctx.split_at

            b_row_indices = rhs_row_indices[:split_at]
            b_sparse_values = rhs_sparse_values[:split_at]

            h_row_indices = rhs_row_indices[split_at:] - num_eq_constraints
            h_sparse_values = rhs_sparse_values[split_at:]

            b_vals = jnp.zeros(num_eq_constraints)
            h_vals = jnp.zeros(num_ineq_constraints)

            b_vals = b_vals.at[b_row_indices].set(-b_sparse_values)
            h_vals = h_vals.at[h_row_indices].set(-h_sparse_values)

            # Build QP model
            model = mpax.create_qp(
                jax.experimental.sparse.BCSR(
                    (quad_obj_vals_i[self.ctx.Q_idxs], *self.ctx.Q_structure),
                    shape=self.ctx.Q_shape,
                ),
                lin_obj_vals_i[self.ctx.c_slice],
                jax.experimental.sparse.BCSR(
                    (con_vals_i[self.ctx.A_idxs], *self.ctx.A_structure),
                    shape=self.ctx.A_shape,
                ),
                b_vals,
                jax.experimental.sparse.BCSR(
                    (con_vals_i[self.ctx.G_idxs], *self.ctx.G_structure),
                    shape=self.ctx.G_shape,
                ),
                h_vals,
                self.ctx.lower,
                self.ctx.upper,
            )

            # Solve
            solution = self.ctx.solver(model)
            return solution.primal_solution, solution.dual_solution

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

    def torch_solve(self, solver_args=None):
        import torch

        primal, dual, vjp_fun = self.jax_solve()
        # Convert JAX arrays to PyTorch tensors
        # jax_solve returns shapes: (batch_size, n) and (batch_size, m)
        primal_torch = torch.utils.dlpack.from_dlpack(primal)
        dual_torch = torch.utils.dlpack.from_dlpack(dual)
        return (
            primal_torch,
            dual_torch,
            vjp_fun,
        )

    def jax_derivative(self, primal, dual, fun):
        raise NotImplementedError(
            "Backward pass is not implemented for MPAX solver. "
            "Use solver='DIFFCP' for differentiable optimization layers."
        )

    def torch_derivative(self, primal, dual, adj_batch):
        import torch

        # Squeeze batch dimension (MPAX doesn't support batching, always has batch_size=1)
        primal_unbatched = primal.squeeze(0) if primal.dim() > 1 else primal
        dual_unbatched = dual.squeeze(0) if dual.dim() > 1 else dual

        quad, lin, con = self.jax_derivative(
            jnp.array(primal_unbatched), jnp.array(dual_unbatched), adj_batch
        )
        # Use DLpack for JAX to PyTorch conversion to avoid read-only flag errors
        return (
            torch.utils.dlpack.from_dlpack(quad),
            torch.utils.dlpack.from_dlpack(lin),
            torch.utils.dlpack.from_dlpack(con),
        )
