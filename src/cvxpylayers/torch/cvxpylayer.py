from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse
import torch

import cvxpylayers.utils.parse_args as pa


def _reshape_fortran(array: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Reshape array using Fortran (column-major) order."""
    if len(array.shape) == 0:
        return array.reshape(shape)
    x = array.permute(*reversed(range(len(array.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def _apply_gp_log_transform(
    params: tuple[torch.Tensor, ...],
    ctx: pa.LayersContext,
) -> tuple[torch.Tensor, ...]:
    """Apply log transformation to geometric program (GP) parameters."""
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params
    return tuple(
        torch.log(p) if ctx.parameters[i] in ctx.gp_param_to_log_param else p
        for i, p in enumerate(params)
    )


def _flatten_and_batch_params(
    params: tuple[torch.Tensor, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> torch.Tensor:
    """Flatten and batch parameters into a single stacked tensor."""
    flattened_params: list[torch.Tensor | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        if ctx.batch_sizes[i] == 0 and batch:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            param_expanded = param.unsqueeze(0).expand(batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded, batch + (-1,)
            )
        else:
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param, batch + (-1,)
            )

    # Add constant 1.0 column for offset terms
    flattened_params[-1] = torch.ones(batch + (1,), dtype=params[0].dtype, device=params[0].device)

    p_stack = torch.cat([p for p in flattened_params if p is not None], -1)
    if batch:
        p_stack = p_stack.T
    return p_stack


def _svec_to_symmetric(
    svec: torch.Tensor,
    n: int,
    batch: tuple,
    rows: np.ndarray,
    cols: np.ndarray,
    scale: np.ndarray | None = None,
) -> torch.Tensor:
    """Convert vectorized form to full symmetric matrix."""
    rows_t = torch.tensor(rows, dtype=torch.long, device=svec.device)
    cols_t = torch.tensor(cols, dtype=torch.long, device=svec.device)
    if scale is not None:
        scale_t = torch.tensor(scale, dtype=svec.dtype, device=svec.device)
        data = svec * scale_t
    else:
        data = svec

    out_shape = batch + (n, n)
    if batch:
        batch_size = int(np.prod(batch))
        data_flat = data.reshape(batch_size, -1)
        result = torch.zeros(batch_size, n, n, dtype=svec.dtype, device=svec.device)
        result[:, rows_t, cols_t] = data_flat
        result[:, cols_t, rows_t] = data_flat
        return result.reshape(out_shape)
    else:
        result = torch.zeros(n, n, dtype=svec.dtype, device=svec.device)
        result[rows_t, cols_t] = data
        result[cols_t, rows_t] = data
        return result


def _unpack_primal_svec(svec: torch.Tensor, n: int, batch: tuple) -> torch.Tensor:
    """Unpack symmetric primal variable from vectorized form (upper tri row-major)."""
    rows, cols = np.triu_indices(n)
    return _svec_to_symmetric(svec, n, batch, rows, cols)


def _unpack_svec(svec: torch.Tensor, n: int, batch: tuple) -> torch.Tensor:
    """Unpack scaled vectorized (svec) form to full symmetric matrix."""
    rows_rm, cols_rm = np.tril_indices(n)
    sort_idx = np.lexsort((rows_rm, cols_rm))
    rows = rows_rm[sort_idx]
    cols = cols_rm[sort_idx]
    scale = np.where(rows == cols, 1.0, 1.0 / np.sqrt(2.0))
    return _svec_to_symmetric(svec, n, batch, rows, cols, scale)


def _recover_results(
    primal: torch.Tensor,
    dual: torch.Tensor,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[torch.Tensor, ...]:
    """Recover variable values from primal/dual solutions."""
    results = []
    for var in ctx.var_recover:
        batch_shape = tuple(primal.shape[:-1])
        if var.primal is not None:
            data = primal[..., var.primal]
            if var.is_symmetric:
                results.append(_unpack_primal_svec(data, var.shape[0], batch_shape))
            else:
                results.append(_reshape_fortran(data, batch_shape + var.shape))
        elif var.dual is not None:
            data = dual[..., var.dual]
            if var.is_psd_dual:
                results.append(_unpack_svec(data, var.shape[0], batch_shape))
            else:
                results.append(_reshape_fortran(data, batch_shape + var.shape))
        else:
            raise RuntimeError("Invalid VariableRecovery")

    # Apply exp transformation for GP primal variables
    if ctx.gp:
        results = [
            torch.exp(r) if var.primal is not None else r
            for r, var in zip(results, ctx.var_recover)
        ]

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = [r.squeeze(0) for r in results]

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

        # Get dtype from input parameters to ensure type matching
        param_dtype = p_stack.dtype

        # Evaluate parametrized matrices (convert sparse matrices to match input dtype)
        P_eval = (self.P.to(dtype=param_dtype) @ p_stack) if self.P is not None else None
        q_eval = self.q.to(dtype=param_dtype) @ p_stack  # type: ignore[operator]
        A_eval = self.A.to(dtype=param_dtype) @ p_stack  # type: ignore[operator]

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
    torch_csr = torch.sparse_csr_tensor(
        crow_indices=torch.tensor(row_ptr, dtype=torch.int64),
        col_indices=torch.tensor(col_indices, dtype=torch.int64),
        values=torch.tensor(values, dtype=torch.float64),
        size=(num_rows, num_cols),
    )

    return torch_csr
