from typing import Any, cast

import cvxpy as cp
import scipy.sparse
import torch

import cvxpylayers.utils.parse_args as pa
from cvxpylayers.utils.parse_args import (
    apply_gp_log_transform,
    flatten_and_batch_params,
    recover_results,
)


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
        params = apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = flatten_and_batch_params(params, self.ctx, batch)

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
        return recover_results(primal, dual, self.ctx, batch)


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
