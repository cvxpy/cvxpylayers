from typing import Any, cast

import cvxpy as cp
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import scipy.sparse

import cvxpylayers.utils.parse_args as pa


class GpuCvxpyLayer:
    def __init__(
        self,
        problem: cp.Problem,
        parameters: list[cp.Parameter],
        variables: list[cp.Variable],
        solver: str | None = None,
        gp: bool = False,
        verbose: bool = False,
        canon_backend: str | None = None,
        solver_args: dict[str, Any] | None = None,
    ) -> None:
        assert gp is False
        if solver_args is None:
            solver_args = {}
        self.ctx = pa.parse_args(
            problem,
            variables,
            parameters,
            solver,
            gp=gp,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_args=solver_args,
        )
        if self.ctx.reduced_P.reduced_mat is not None:  # type: ignore[attr-defined]
            self.P = scipy_csr_to_jax_bcsr(self.ctx.reduced_P.reduced_mat)  # type: ignore[attr-defined]
        else:
            self.P = None
        self.q: jax.experimental.sparse.BCSR = scipy_csr_to_jax_bcsr(self.ctx.q.tocsr())  # type: ignore[assignment]
        self.A: jax.experimental.sparse.BCSR = scipy_csr_to_jax_bcsr(self.ctx.reduced_A.reduced_mat)  # type: ignore[attr-defined,assignment]

    def __call__(
        self, *params: jnp.ndarray, solver_args: dict[str, Any] | None = None
    ) -> tuple[jnp.ndarray, ...]:
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))
        flattened_params: list[jnp.ndarray | None] = [None] * (len(params) + 1)
        for i, param in enumerate(params):
            # Check if this parameter is batched or needs broadcasting
            if self.ctx.batch_sizes[i] == 0 and batch:
                # Unbatched parameter - expand to match batch size
                # Add batch dimension by repeating
                param_expanded = jnp.expand_dims(param, 0)
                param_expanded = jnp.broadcast_to(param_expanded, batch + param.shape)
                flattened_params[self.ctx.user_order_to_col_order[i]] = jnp.reshape(
                    param_expanded,
                    batch + (-1,),
                    order="F",
                )
            else:
                # Already batched or no batch dimension needed
                flattened_params[self.ctx.user_order_to_col_order[i]] = jnp.reshape(
                    param,
                    batch + (-1,),
                    order="F",
                )
        flattened_params[-1] = jnp.ones(batch + (1,), dtype=params[0].dtype)
        # Assert all parameters have been assigned (no Nones remain)
        assert all(p is not None for p in flattened_params), "All parameters must be assigned"
        p_stack = jnp.concatenate(cast(list[jnp.ndarray], flattened_params), -1)
        # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
        if batch:
            p_stack = p_stack.T
        P_eval = self.P @ p_stack if self.P is not None else None
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack

        # Store data and adjoint in closure for backward pass
        # This avoids JAX trying to trace through DIFFCP's Python-based solver
        data_container = {}

        @jax.custom_vjp
        def solve_problem(P_eval, q_eval, A_eval):
            # Forward pass: solve the optimization problem
            data = self.ctx.solver_ctx.jax_to_data(P_eval, q_eval, A_eval)  # type: ignore[attr-defined]
            primal, dual, adj_batch = data.jax_solve(solver_args)  # type: ignore[attr-defined]
            # Store for backward pass (outside JAX tracing)
            data_container["data"] = data
            data_container["adj_batch"] = adj_batch
            return primal, dual

        def solve_problem_fwd(P_eval, q_eval, A_eval):
            # Call forward to execute and populate container
            primal, dual = solve_problem(P_eval, q_eval, A_eval)
            # Return empty residuals (data is in closure)
            return (primal, dual), ()

        def solve_problem_bwd(res, g):
            # Backward pass using adjoint method
            dprimal, ddual = g
            data = data_container["data"]
            adj_batch = data_container["adj_batch"]
            dP, dq, dA = data.jax_derivative(dprimal, ddual, adj_batch)
            return dP, dq, dA

        solve_problem.defvjp(solve_problem_fwd, solve_problem_bwd)
        primal, dual = solve_problem(P_eval, q_eval, A_eval)
        results = tuple(var.recover(primal, dual) for var in self.ctx.var_recover)

        # Squeeze batch dimension for unbatched inputs
        if not batch:
            results = tuple(jnp.squeeze(r, 0) for r in results)

        return results


def scipy_csr_to_jax_bcsr(
    scipy_csr: scipy.sparse.csr_array | None,
) -> jax.experimental.sparse.BCSR | None:
    if scipy_csr is None:
        return None
    # Use cast to help type checker understand scipy_csr is not None
    scipy_csr = cast(scipy.sparse.csr_array, scipy_csr)
    # Get the CSR format components
    values = scipy_csr.data
    col_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr
    num_rows, num_cols = scipy_csr.shape  # type: ignore[misc]

    # Create the JAX BCSR tensor
    jax_bcsr = jax.experimental.sparse.BCSR(
        (jnp.array(values), jnp.array(col_indices), jnp.array(row_ptr)),
        shape=(num_rows, num_cols),
    )

    return jax_bcsr


CvxpyLayer = GpuCvxpyLayer
