from typing import Any, cast

import cvxpy as cp
import jax
import jax.experimental.sparse
import jax.numpy as jnp
import numpy as np
import scipy.sparse

try:
    from moreau._types import BatchedWarmStart, WarmStart
except ImportError:
    WarmStart = None  # type: ignore[assignment,misc]
    BatchedWarmStart = None  # type: ignore[assignment,misc]

import cvxpylayers.utils.parse_args as pa


def _reshape_fortran(array: jnp.ndarray, shape: tuple) -> jnp.ndarray:
    """Reshape array using Fortran (column-major) order.

    Args:
        array: Input array to reshape
        shape: Target shape tuple

    Returns:
        Reshaped array in Fortran order
    """
    return jnp.reshape(array, shape, order="F")


def _apply_gp_log_transform(
    params: tuple[jnp.ndarray, ...],
    ctx: pa.LayersContext,
) -> tuple[jnp.ndarray, ...]:
    """Apply log transformation to geometric program (GP) parameters.

    Geometric programs are solved in log-space after conversion to DCP.
    This function applies log transformation to the appropriate parameters.

    Args:
        params: Tuple of parameter arrays in original GP space
        ctx: Layer context containing GP parameter mapping info

    Returns:
        Tuple of transformed parameters (log-space for GP params, unchanged otherwise)
    """
    if not ctx.gp or ctx.gp_log_mask is None:
        return params

    # Use pre-computed mask for JIT compatibility (no dict lookups)
    return tuple(
        jnp.log(p) if needs_log else p
        for p, needs_log in zip(params, ctx.gp_log_mask)
    )


def _flatten_and_batch_params(
    params: tuple[jnp.ndarray, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> jnp.ndarray:
    """Flatten and batch parameters into a single stacked array.

    Converts a tuple of parameter arrays (potentially with mixed batched/unbatched)
    into a single concatenated array suitable for matrix multiplication with the
    parametrized problem matrices.

    Args:
        params: Tuple of parameter arrays
        ctx: Layer context with batch info and ordering
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Concatenated parameter array with shape (num_params, batch_size) or (num_params,)
    """
    flattened_params: list[jnp.ndarray | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0 and batch:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            param_expanded = jnp.broadcast_to(jnp.expand_dims(param, 0), batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded,
                batch + (-1,),
            )
        else:
            # Already batched or no batch dimension needed
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param,
                batch + (-1,),
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = jnp.ones(batch + (1,), dtype=params[0].dtype)

    assert all(p is not None for p in flattened_params), "All parameters must be assigned"
    p_stack = jnp.concatenate(cast(list[jnp.ndarray], flattened_params), -1)
    # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
    if batch:
        p_stack = p_stack.T
    return p_stack


def _svec_to_symmetric(
    svec: jnp.ndarray,
    n: int,
    batch: tuple,
    rows: np.ndarray,
    cols: np.ndarray,
    scale: np.ndarray | None = None,
) -> jnp.ndarray:
    """Convert vectorized form to full symmetric matrix.

    Args:
        svec: Vectorized form, shape (*batch, n*(n+1)/2)
        n: Matrix dimension
        batch: Batch dimensions
        rows: Row indices for unpacking
        cols: Column indices for unpacking
        scale: Optional scaling factors (for svec format with sqrt(2) scaling)

    Returns:
        Full symmetric matrix, shape (*batch, n, n)
    """
    rows_arr = jnp.array(rows)
    cols_arr = jnp.array(cols)
    data = svec * jnp.array(scale) if scale is not None else svec
    out_shape = batch + (n, n)
    result = jnp.zeros(out_shape, dtype=svec.dtype)
    result = result.at[..., rows_arr, cols_arr].set(data)
    result = result.at[..., cols_arr, rows_arr].set(data)
    return result


def _unpack_primal_svec(svec: jnp.ndarray, n: int, batch: tuple) -> jnp.ndarray:
    """Unpack symmetric primal variable from vectorized form.

    CVXPY stores symmetric variables in upper triangular row-major order:
    [X[0,0], X[0,1], ..., X[0,n-1], X[1,1], X[1,2], ..., X[n-1,n-1]]

    Args:
        svec: Vectorized symmetric variable
        n: Matrix dimension
        batch: Batch dimensions

    Returns:
        Full symmetric matrix
    """
    rows, cols = np.triu_indices(n)
    return _svec_to_symmetric(svec, n, batch, rows, cols)


def _unpack_svec(svec: jnp.ndarray, n: int, batch: tuple) -> jnp.ndarray:
    """Unpack scaled vectorized (svec) form to full symmetric matrix.

    The svec format stores a symmetric n x n matrix as a vector of length n*(n+1)/2,
    with off-diagonal elements scaled by sqrt(2). Uses column-major lower triangular
    ordering: (0,0), (1,0), (1,1), (2,0), ...

    Args:
        svec: Scaled vectorized form
        n: Matrix dimension
        batch: Batch dimensions

    Returns:
        Full symmetric matrix with scaling removed
    """
    rows_rm, cols_rm = np.tril_indices(n)
    sort_idx = np.lexsort((rows_rm, cols_rm))
    rows = rows_rm[sort_idx]
    cols = cols_rm[sort_idx]
    # Scale: 1.0 for diagonal, 1/sqrt(2) for off-diagonal
    scale = np.where(rows == cols, 1.0, 1.0 / np.sqrt(2.0))
    return _svec_to_symmetric(svec, n, batch, rows, cols, scale)


def _recover_results(
    primal: jnp.ndarray,
    dual: jnp.ndarray,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[jnp.ndarray, ...]:
    """Recover variable values from primal/dual solutions.

    Extracts the requested variables from the solver's primal and dual
    solutions, unpacks symmetric matrices if needed, applies inverse GP
    transformation, and removes batch dimension for unbatched inputs.

    Args:
        primal: Primal solution from solver
        dual: Dual solution from solver
        ctx: Layer context with variable recovery info
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Tuple of recovered variable values
    """
    results = []
    batch_shape = tuple(primal.shape[:-1])

    for var in ctx.var_recover:
        # Use pre-computed source field to select data (JIT-compatible)
        if var.source == "primal":
            data = primal[..., var.primal]
        else:  # var.source == "dual"
            data = dual[..., var.dual]

        # Use pre-computed unpack_fn field (JIT-compatible)
        if var.unpack_fn == "svec_primal":
            results.append(_unpack_primal_svec(data, var.shape[0], batch_shape))
        elif var.unpack_fn == "svec_dual":
            results.append(_unpack_svec(data, var.shape[0], batch_shape))
        elif var.unpack_fn == "reshape":
            results.append(_reshape_fortran(data, batch_shape + var.shape))
        else:
            raise ValueError(f"Unknown variable recovery type: {var.unpack_fn}")

    # Apply exp transformation to recover primal variables from log-space for GP
    # (dual variables stay in original space - no transformation needed)
    # Uses pre-computed source field (JIT-compatible)
    if ctx.gp:
        results = [
            jnp.exp(r) if var.source == "primal" else r
            for r, var in zip(results, ctx.var_recover)
        ]

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = [jnp.squeeze(r, 0) for r in results]

    return tuple(results)


class CvxpyLayer:
    """A differentiable convex optimization layer for JAX.

    This layer wraps a parametrized CVXPY problem, solving it in the forward pass
    and computing gradients via implicit differentiation. Compatible with
    ``jax.grad``, ``jax.jit``, and ``jax.vmap``.

    JIT/vmap Compatibility:
        When using solver="MOREAU", this layer is fully compatible with
        jax.jit, jax.vmap, and jax.pmap. The Moreau solver provides native
        JAX autodiff support via custom_vjp with pure_callback, enabling
        JIT compilation of the entire solve-differentiate pipeline.

        Other solvers (DIFFCP) use Python-based solving and are not
        JIT-compatible due to closure-based gradient handling.

    Example:
        >>> import cvxpy as cp
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from cvxpylayers.jax import CvxpyLayer
        >>>
        >>> # Define a simple QP
        >>> x = cp.Variable(2)
        >>> A = cp.Parameter((3, 2))
        >>> b = cp.Parameter(3)
        >>> problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
        >>>
        >>> # Create the layer
        >>> layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        >>>
        >>> # Solve and compute gradients
        >>> A_jax = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
        >>> b_jax = jax.random.normal(jax.random.PRNGKey(1), (3,))
        >>> (solution,) = layer(A_jax, b_jax)
        >>>
        >>> # Gradient computation
        >>> def loss_fn(A, b):
        ...     (x,) = layer(A, b)
        ...     return jnp.sum(x)
        >>> grads = jax.grad(loss_fn, argnums=[0, 1])(A_jax, b_jax)
    """

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
        """Initialize the differentiable optimization layer.

        Args:
            problem: A CVXPY Problem. Must be DPP-compliant (``problem.is_dpp()``
                must return True).
            parameters: List of CVXPY Parameters that will be filled with values
                at runtime. Order must match the order of arrays passed to __call__().
            variables: List of CVXPY Variables whose optimal values will be returned
                by __call__(). Order determines the order of returned arrays.
            solver: CVXPY solver to use (e.g., ``cp.CLARABEL``, ``cp.SCS``).
                If None, uses the default diffcp solver.
            gp: If True, problem is a geometric program. Parameters will be
                log-transformed before solving.
            verbose: If True, print solver output.
            canon_backend: Backend for canonicalization. Options are 'diffcp',
                'cuclarabel', or None (auto-select).
            solver_args: Default keyword arguments passed to the solver.
                Can be overridden per-call in __call__().

        Raises:
            AssertionError: If problem is not DPP-compliant.
            ValueError: If parameters or variables are not part of the problem.
        """
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

        # Cache the Moreau solve function for JIT compatibility.
        # Must be captured as closure, not looked up dynamically during tracing.
        if self.ctx.solver == "MOREAU":
            self._moreau_solve_fn = self.ctx.solver_ctx.get_jax_solver()._impl.solve
        self._warm_start_cache = None

    def __call__(
        self,
        *params: jnp.ndarray,
        solver_args: dict[str, Any] | None = None,
        warm_start: bool = False,
    ) -> tuple[jnp.ndarray, ...]:
        """Solve the optimization problem and return optimal variable values.

        Args:
            *params: Array values for each CVXPY Parameter, in the same order
                as the ``parameters`` argument to __init__(). Each array shape must
                match the corresponding Parameter shape, optionally with a batch
                dimension prepended. Batched and unbatched parameters can be mixed;
                unbatched parameters are broadcast.
            solver_args: Keyword arguments passed to the solver, overriding any
                defaults set in __init__().
            warm_start: If True, use the cached solution from the previous solve
                as a warm start for the solver. Only supported with solver="MOREAU".

        Returns:
            Tuple of arrays containing optimal values for each CVXPY Variable
            specified in the ``variables`` argument to __init__(). If inputs are
            batched, outputs will have matching batch dimensions.

        Raises:
            RuntimeError: If the solver fails to find a solution.
            ValueError: If warm_start=True with a non-Moreau solver.

        Example:
            >>> # Single problem
            >>> (x_opt,) = layer(A_array, b_array)
            >>>
            >>> # Batched: solve 10 problems in parallel
            >>> A_batch = jax.random.normal(key, (10, 3, 2))
            >>> b_batch = jax.random.normal(key, (10, 3))
            >>> (x_batch,) = layer(A_batch, b_batch)  # x_batch.shape = (10, 2)
        """
        if solver_args is None:
            solver_args = {}
        if warm_start and self.ctx.solver != "MOREAU":
            raise ValueError(
                "warm_start=True is only supported with solver='MOREAU'. "
                f"Current solver is '{self.ctx.solver}'."
            )
        batch = self.ctx.validate_params(list(params))

        # Apply log transformation to GP parameters
        params = _apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = _flatten_and_batch_params(params, self.ctx, batch)

        # Evaluate parametrized matrices
        P_eval = self.P @ p_stack if self.P is not None else None
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack

        # Check if solver has native JAX autodiff (Moreau)
        # If so, bypass the closure-based custom_vjp wrapper and use Moreau's native autodiff
        if self.ctx.solver == "MOREAU":
            ws = self._warm_start_cache if warm_start else None
            return self._solve_moreau(P_eval, q_eval, A_eval, batch, solver_args,
                                      warm_start=ws)

        # Non-Moreau: use existing custom_vjp wrapper (not JIT-compatible)
        return self._solve_with_custom_vjp(P_eval, q_eval, A_eval, batch, solver_args)

    def _solve_moreau(
        self,
        P_eval: jnp.ndarray | None,
        q_eval: jnp.ndarray,
        A_eval: jnp.ndarray,
        batch: tuple,
        solver_args: dict[str, Any],
        warm_start: Any = None,
    ) -> tuple[jnp.ndarray, ...]:
        """Direct call to Moreau solver - uses its native custom_vjp (JIT-compatible).

        Moreau's JAX solver (moreau.jax.Solver) implements custom_vjp with
        pure_callback and vmap_method="broadcast_all", making it fully compatible
        with jax.jit, jax.vmap, and jax.pmap.

        This method uses jax.vmap for batched cases to ensure each problem is
        solved individually, which is required for JIT compatibility.

        Args:
            warm_start: Optional WarmStart or BatchedWarmStart from a previous
                solve. Set on the solver's _impl before solving.
        """
        solver_ctx = self.ctx.solver_ctx  # type: ignore[attr-defined]
        jax_solver = solver_ctx.get_jax_solver()

        # Apply per-call solver_args to solver settings
        if solver_args:
            settings = jax_solver._impl._settings
            for key, value in solver_args.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

        # Set warm start on the solver impl (mimics moreau.jax.Solver.solve() internals)
        if warm_start is not None:
            jax_solver._impl._pending_warm_start = {
                "warm_x": np.asarray(warm_start.x, dtype=np.float64),
                "warm_z": np.asarray(warm_start.z, dtype=np.float64),
                "warm_s": np.asarray(warm_start.s, dtype=np.float64),
            }

        # Use cached solve function (captured as closure, not dynamic lookup)
        solve_fn = self._moreau_solve_fn

        # Cache solver_ctx attributes for closure capture.
        # Parametrization matrices are pre-permuted so matrix multiplication
        # directly produces values in CSR order — no shuffle indexing needed.
        b_idx = solver_ctx.b_idx
        nnz_A = solver_ctx.nnz_A
        m = solver_ctx.A_shape[0]

        def extract_and_solve(
            P_eval_single: jnp.ndarray | None,
            q_eval_single: jnp.ndarray,
            A_eval_single: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Extract problem data and solve a single (unbatched) problem."""
            # P values are already in CSR order from pre-permuted matrix multiply
            if P_eval_single is not None:
                P_values = P_eval_single  # (nnzP,)
            else:
                P_values = jnp.zeros(0, dtype=jnp.float64)

            # A values: first nnz_A entries are in CSR order, negated for Ax + s = b form
            A_values = -A_eval_single[:nnz_A]  # (nnzA,)

            # b vector: entries after the A values
            b_raw = A_eval_single[nnz_A:]  # (nb,)

            # Scatter b_raw into full b vector at correct indices
            b = jnp.zeros(m, dtype=jnp.float64)
            b = b.at[b_idx].set(b_raw)

            # Extract q (linear cost) - exclude the constant term at the end
            q = q_eval_single[:-1]  # (n,)

            # Call moreau's solve with unbatched inputs
            solution, _info = solve_fn(P_values, A_values, q, b)

            return solution.x, solution.z, solution.s  # (n,), (m,), (m,)

        if batch:
            # Batched case: inputs have shape (dim, batch)
            # Transpose to (batch, dim) for vmap
            P_eval_batched = P_eval.T if P_eval is not None else None
            q_eval_batched = q_eval.T  # (batch, dim)
            A_eval_batched = A_eval.T  # (batch, dim)

            # Use vmap to solve each problem individually
            vmapped_solve = jax.vmap(extract_and_solve, in_axes=(0, 0, 0))
            primal, dual, slack = vmapped_solve(
                P_eval_batched, q_eval_batched, A_eval_batched
            )
            # primal: (batch, n), dual: (batch, m), slack: (batch, m)

        else:
            # Unbatched case: inputs have shape (dim,)
            primal, dual, slack = extract_and_solve(P_eval, q_eval, A_eval)
            # primal: (n,), dual: (m,), slack: (m,)

            # Add batch dimension for _recover_results (which expects it)
            primal = jnp.expand_dims(primal, 0)  # (1, n)
            dual = jnp.expand_dims(dual, 0)  # (1, m)

        # Clear pending warm start after solve
        jax_solver._impl._pending_warm_start = None

        # Always update warm start cache (negligible cost)
        if batch:
            self._warm_start_cache = BatchedWarmStart(
                x=np.asarray(primal, dtype=np.float64),
                z=np.asarray(dual, dtype=np.float64),
                s=np.asarray(slack, dtype=np.float64),
            )
        else:
            self._warm_start_cache = WarmStart(
                x=np.asarray(primal.squeeze(0), dtype=np.float64),
                z=np.asarray(dual.squeeze(0), dtype=np.float64),
                s=np.asarray(slack, dtype=np.float64),
            )

        return _recover_results(primal, dual, self.ctx, batch)

    def _solve_with_custom_vjp(
        self,
        P_eval: jnp.ndarray | None,
        q_eval: jnp.ndarray,
        A_eval: jnp.ndarray,
        batch: tuple,
        solver_args: dict[str, Any],
    ) -> tuple[jnp.ndarray, ...]:
        """Solve using closure-based custom_vjp wrapper (not JIT-compatible).

        This is used for non-Moreau solvers (e.g., DIFFCP) that don't have
        native JAX autodiff support. The closure-based approach stores data
        in a Python dict for the backward pass, which breaks JIT tracing.
        """
        # Store data and adjoint in closure for backward pass
        # This avoids JAX trying to trace through DIFFCP's Python-based solver
        data_container: dict[str, Any] = {}

        @jax.custom_vjp
        def solve_problem(P_eval, q_eval, A_eval):
            # Function body: runs when NOT differentiating
            # Use solve_only to skip computing the adjoint operator
            data = self.ctx.solver_ctx.jax_to_data(P_eval, q_eval, A_eval)  # type: ignore[attr-defined]
            primal, dual = data.jax_solve_only(solver_args)  # type: ignore[attr-defined]
            return primal, dual

        def solve_problem_fwd(P_eval, q_eval, A_eval):
            # Runs when differentiating: compute derivatives for backward pass
            data = self.ctx.solver_ctx.jax_to_data(P_eval, q_eval, A_eval)  # type: ignore[attr-defined]
            primal, dual, adj_batch = data.jax_solve(solver_args)  # type: ignore[attr-defined]
            # Store for backward pass (outside JAX tracing)
            data_container["data"] = data
            data_container["adj_batch"] = adj_batch
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

        # Recover results and apply GP inverse transform if needed
        return _recover_results(primal, dual, self.ctx, batch)


def scipy_csr_to_jax_bcsr(
    scipy_csr: scipy.sparse.csr_array | None,
) -> jax.experimental.sparse.BCSR | None:
    if scipy_csr is None:
        return None
    # Use cast to help type checker understand scipy_csr is not None
    scipy_csr = cast(scipy.sparse.csr_array, scipy_csr)
    num_rows, num_cols = scipy_csr.shape  # type: ignore[misc]

    # JAX BCSR doesn't handle empty matrices (0 rows) properly.
    # Create a minimal valid BCSR with a single zero element instead.
    if num_rows == 0:
        # Create a (1, num_cols) matrix with a single zero at position (0, 0)
        # This will produce a (1, ...) result when multiplied, which we'll slice to (0, ...)
        return _EmptyBCSRWrapper(num_cols)

    # Get the CSR format components
    values = scipy_csr.data
    col_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr

    # Create the JAX BCSR tensor
    jax_bcsr = jax.experimental.sparse.BCSR(
        (jnp.array(values), jnp.array(col_indices), jnp.array(row_ptr)),
        shape=(num_rows, num_cols),
    )

    return jax_bcsr


class _EmptyBCSRWrapper:
    """Wrapper for empty (0-row) sparse matrices that JAX BCSR can't handle.

    When multiplied with a vector/matrix, returns an empty array with the correct shape.
    """

    def __init__(self, num_cols: int):
        self.num_cols = num_cols
        self.shape = (0, num_cols)

    def __matmul__(self, other: jnp.ndarray) -> jnp.ndarray:
        # other shape: (num_cols,) or (num_cols, batch)
        if other.ndim == 1:
            return jnp.zeros((0,), dtype=other.dtype)
        else:
            batch_size = other.shape[1]
            return jnp.zeros((0, batch_size), dtype=other.dtype)
