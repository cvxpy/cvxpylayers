import warnings
from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse
import torch

import cvxpylayers.utils.parse_args as pa


def _reshape_fortran(array: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Reshape array using Fortran (column-major) order.

    PyTorch doesn't support order='F' directly, so we use permutation.

    Args:
        array: Input tensor to reshape
        shape: Target shape tuple

    Returns:
        Reshaped tensor in Fortran order
    """
    if len(array.shape) == 0:
        return array.reshape(shape)
    x = array.permute(*reversed(range(len(array.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def _apply_gp_log_transform(
    params: tuple[torch.Tensor, ...],
    ctx: pa.LayersContext,
) -> tuple[torch.Tensor, ...]:
    """Apply log transformation to geometric program (GP) parameters.

    Geometric programs are solved in log-space after conversion to DCP.
    This function applies log transformation to the appropriate parameters.

    Args:
        params: Tuple of parameter tensors in original GP space
        ctx: Layer context containing GP parameter mapping info

    Returns:
        Tuple of transformed parameters (log-space for GP params, unchanged otherwise)
    """
    if not ctx.gp or ctx.gp_log_mask is None:
        return params

    # Use pre-computed mask for JIT compatibility (no dict lookups)
    return tuple(
        torch.log(p) if needs_log else p
        for p, needs_log in zip(params, ctx.gp_log_mask)
    )


def _flatten_and_batch_params(
    params: tuple[torch.Tensor, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> torch.Tensor:
    """Flatten and batch parameters into a single stacked tensor.

    Converts a tuple of parameter tensors (potentially with mixed batched/unbatched)
    into a single concatenated tensor suitable for matrix multiplication with the
    parametrized problem matrices.

    Args:
        params: Tuple of parameter tensors
        ctx: Layer context with batch info and ordering
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Concatenated parameter tensor with shape (num_params, batch_size) or (num_params,)
    """
    flattened_params: list[torch.Tensor | None] = [None] * (len(params) + 1)

    # Always work with batch_size >= 1 for uniform processing
    # batch is () for unbatched, (B,) for batched
    effective_batch = batch if batch else (1,)

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            param_expanded = param.unsqueeze(0).expand(effective_batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded,
                effective_batch + (-1,),
            )
        else:
            # Already batched
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param,
                effective_batch + (-1,),
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = torch.ones(
        effective_batch + (1,),
        dtype=params[0].dtype,
        device=params[0].device,
    )

    assert all(p is not None for p in flattened_params), "All parameters must be assigned"
    p_stack = torch.cat(flattened_params, -1)  # type: ignore[arg-type]
    # p_stack is always (batch_size, num_params), transpose to (num_params, batch_size)
    # Use permute which works uniformly for 2D tensors
    p_stack = p_stack.T
    # For unbatched case, squeeze the trailing dimension to get (num_params,)
    # len(batch) is 0 for unbatched, 1 for batched - use this as a shape selector
    # squeeze(-1) removes the last dimension if it's 1, otherwise no-op
    # But we need conditional-free: use reshape with computed output shape
    output_shape = (p_stack.shape[0],) + batch  # (num_params,) or (num_params, B)
    return p_stack.reshape(output_shape)


def _svec_to_symmetric(
    svec: torch.Tensor,
    n: int,
    batch: tuple,
    rows: np.ndarray,
    cols: np.ndarray,
    scale: np.ndarray | None = None,
) -> torch.Tensor:
    """Convert vectorized form to full symmetric matrix.

    Args:
        svec: Vectorized form, shape (*batch, n*(n+1)/2)
        n: Matrix dimension
        batch: Batch dimensions (from tensor shape, not boolean)
        rows: Row indices for unpacking
        cols: Column indices for unpacking
        scale: Optional scaling factors (for svec format with sqrt(2) scaling)

    Returns:
        Full symmetric matrix, shape (*batch, n, n)
    """
    rows_t = torch.tensor(rows, dtype=torch.long, device=svec.device)
    cols_t = torch.tensor(cols, dtype=torch.long, device=svec.device)
    if scale is not None:
        scale_t = torch.tensor(scale, dtype=svec.dtype, device=svec.device)
        data = svec * scale_t
    else:
        data = svec

    # Output shape: (*batch, n, n) - works for () or (B,) batch
    out_shape = batch + (n, n)
    result = torch.zeros(out_shape, dtype=svec.dtype, device=svec.device)

    # Use ellipsis indexing - works uniformly for any number of batch dimensions
    result[..., rows_t, cols_t] = data
    result[..., cols_t, rows_t] = data
    return result


def _unpack_primal_svec(svec: torch.Tensor, n: int, batch: tuple) -> torch.Tensor:
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


def _unpack_svec(svec: torch.Tensor, n: int, batch: tuple) -> torch.Tensor:
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
    primal: torch.Tensor,
    dual: torch.Tensor,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[torch.Tensor, ...]:
    """Recover variable values from primal/dual solutions.

    Extracts the requested variables from the solver's primal and dual
    solutions, unpacks symmetric matrices if needed, applies inverse GP
    transformation, and removes batch dimension for unbatched inputs.

    Args:
        primal: Primal solution from solver
        dual: Dual solution from solver
        ctx: Layer context with variable recovery info
        batch: Batch dimensions tuple (empty () for unbatched, (B,) for batched)

    Returns:
        Tuple of recovered variable values
    """
    results = []
    # Solver always returns 2D tensors (batch_size, num_vars), even for unbatched (batch_size=1)
    # Use internal_batch for intermediate processing, output_batch for final reshape
    internal_batch = tuple(primal.shape[:-1])  # (1,) for unbatched, (B,) for batched
    output_batch = batch  # () for unbatched, (B,) for batched

    for var in ctx.var_recover:
        # Use pre-computed source field to select data (JIT-compatible)
        if var.source == "primal":
            data = primal[..., var.primal]
        else:  # var.source == "dual"
            data = dual[..., var.dual]

        # Use pre-computed unpack_fn field (JIT-compatible)
        if var.unpack_fn == "svec_primal":
            result = _unpack_primal_svec(data, var.shape[0], internal_batch)
        elif var.unpack_fn == "svec_dual":
            result = _unpack_svec(data, var.shape[0], internal_batch)
        else:  # var.unpack_fn == "reshape"
            result = _reshape_fortran(data, internal_batch + var.shape)

        # Reshape to output batch shape (removes dummy batch dim for unbatched)
        # reshape from (1,) + var.shape to () + var.shape works because total elements match
        results.append(result.reshape(output_batch + var.shape))

    # Apply exp transformation to recover primal variables from log-space for GP
    # (dual variables stay in original space - no transformation needed)
    # Uses pre-computed source field (JIT-compatible)
    if ctx.gp:
        results = [
            torch.exp(r) if var.source == "primal" else r
            for r, var in zip(results, ctx.var_recover)
        ]

    return tuple(results)


class CvxpyLayer(torch.nn.Module):
    """A differentiable convex optimization layer for PyTorch.

    This layer wraps a parametrized CVXPY problem, solving it in the forward pass
    and computing gradients via implicit differentiation in the backward pass.

    Example:
        >>> import cvxpy as cp
        >>> import torch
        >>> from cvxpylayers.torch import CvxpyLayer
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
        >>> # Solve with gradients
        >>> A_t = torch.randn(3, 2, requires_grad=True)
        >>> b_t = torch.randn(3, requires_grad=True)
        >>> (solution,) = layer(A_t, b_t)
        >>> solution.sum().backward()
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
                at runtime. Order must match the order of tensors passed to forward().
            variables: List of CVXPY Variables whose optimal values will be returned
                by forward(). Order determines the order of returned tensors.
            solver: CVXPY solver to use (e.g., ``cp.CLARABEL``, ``cp.SCS``).
                If None, uses the default diffcp solver.
            gp: If True, problem is a geometric program. Parameters will be
                log-transformed before solving.
            verbose: If True, print solver output.
            canon_backend: Backend for canonicalization. Options are 'diffcp',
                'cuclarabel', or None (auto-select).
            solver_args: Default keyword arguments passed to the solver.
                Can be overridden per-call in forward().

        Raises:
            AssertionError: If problem is not DPP-compliant.
            ValueError: If parameters or variables are not part of the problem.
        """
        super().__init__()
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
            self.register_buffer(
                "P",
                scipy_csr_to_torch_csr(self.ctx.reduced_P.reduced_mat),  # type: ignore[attr-defined]
            )
        else:
            self.P = None
        self.register_buffer("q", scipy_csr_to_torch_csr(self.ctx.q.tocsr()))
        self.register_buffer(
            "A",
            scipy_csr_to_torch_csr(self.ctx.reduced_A.reduced_mat),  # type: ignore[attr-defined]
        )

    def forward(
        self, *params: torch.Tensor, solver_args: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, ...]:
        """Solve the optimization problem and return optimal variable values.

        Args:
            *params: Tensor values for each CVXPY Parameter, in the same order
                as the ``parameters`` argument to __init__. Each tensor shape must
                match the corresponding Parameter shape, optionally with a batch
                dimension prepended. Batched and unbatched parameters can be mixed;
                unbatched parameters are broadcast.
            solver_args: Keyword arguments passed to the solver, overriding any
                defaults set in __init__.

        Returns:
            Tuple of tensors containing optimal values for each CVXPY Variable
            specified in the ``variables`` argument to __init__. If inputs are
            batched, outputs will have matching batch dimensions.

        Raises:
            RuntimeError: If the solver fails to find a solution.

        Example:
            >>> # Single problem
            >>> (x_opt,) = layer(A_tensor, b_tensor)
            >>>
            >>> # Batched: solve 10 problems in parallel
            >>> A_batch = torch.randn(10, 3, 2)
            >>> b_batch = torch.randn(10, 3)
            >>> (x_batch,) = layer(A_batch, b_batch)  # x_batch.shape = (10, 2)
        """
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))

        # Apply log transformation to GP parameters
        params = _apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = _flatten_and_batch_params(params, self.ctx, batch)

        # Get dtype and device from input parameters to ensure type/device matching
        param_dtype = p_stack.dtype
        param_device = p_stack.device

        # Evaluate parametrized matrices (convert sparse matrices to match input dtype and device)
        P_eval = (
            (self.P.to(dtype=param_dtype, device=param_device) @ p_stack)
            if self.P is not None
            else None
        )
        q_eval = self.q.to(dtype=param_dtype, device=param_device) @ p_stack  # type: ignore[operator]
        A_eval = self.A.to(dtype=param_dtype, device=param_device) @ p_stack  # type: ignore[operator]

        # Get the solver-specific _CvxpyLayer class
        from cvxpylayers.interfaces import get_torch_cvxpylayer

        _CvxpyLayer = get_torch_cvxpylayer(self.ctx.solver)

        # Solve optimization problem
        primal, dual, _, _ = _CvxpyLayer.apply(  # type: ignore[misc]
            P_eval,
            q_eval,
            A_eval,
            self.ctx,
            solver_args,
        )

        # Recover results and apply GP inverse transform if needed
        return _recover_results(primal, dual, self.ctx, batch)


def scipy_csr_to_torch_csr(
    scipy_csr: scipy.sparse.csr_array | None,
) -> torch.Tensor | None:
    if scipy_csr is None:
        return None
    # Use cast to help type checker understand scipy_csr is not None
    scipy_csr = cast(scipy.sparse.csr_array, scipy_csr)
    # Get the CSR format components
    values = scipy_csr.data
    col_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr
    num_rows, num_cols = scipy_csr.shape  # type: ignore[misc]

    # Create the torch sparse csr_tensor
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state",
            category=UserWarning,
        )
        torch_csr = torch.sparse_csr_tensor(
            crow_indices=torch.tensor(row_ptr, dtype=torch.int64),
            col_indices=torch.tensor(col_indices, dtype=torch.int64),
            values=torch.tensor(values, dtype=torch.float64),
            size=(num_rows, num_cols),
        )

    return torch_csr
