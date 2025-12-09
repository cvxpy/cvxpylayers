from typing import Any, cast

import cvxpy as cp
import scipy.sparse
import torch

import cvxpylayers.utils.parse_args as pa


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
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params

    params_transformed = []
    for i, param in enumerate(params):
        cvxpy_param = ctx.parameters[i]
        if cvxpy_param in ctx.gp_param_to_log_param:
            # This parameter needs log transformation for GP
            params_transformed.append(torch.log(param))
        else:
            params_transformed.append(param)
    return tuple(params_transformed)


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

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0 and batch:
            # Unbatched parameter - expand to match batch size
            param_expanded = param.unsqueeze(0).expand(batch + param.shape)
            flattened_params[ctx.user_order_to_col_order[i]] = reshape_fortran(
                param_expanded,
                batch + (-1,),
            )
        else:
            # Already batched or no batch dimension needed
            flattened_params[ctx.user_order_to_col_order[i]] = reshape_fortran(
                param,
                batch + (-1,),
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = torch.ones(
        batch + (1,),
        dtype=params[0].dtype,
        device=params[0].device,
    )
    assert all(p is not None for p in flattened_params), "All parameters must be assigned"

    p_stack = torch.cat(cast(list[torch.Tensor], flattened_params), -1)
    # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
    if batch:
        p_stack = p_stack.T
    return p_stack


def _recover_results(
    primal: torch.Tensor,
    dual: torch.Tensor,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[torch.Tensor, ...]:
    """Recover variable values from primal/dual solutions.

    Extracts the requested variables from the solver's primal and dual
    solutions, applies inverse GP transformation if needed, and removes
    batch dimension for unbatched inputs.

    Args:
        primal: Primal solution from solver
        dual: Dual solution from solver
        ctx: Layer context with variable recovery info
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Tuple of recovered variable values
    """
    # Extract each variable using its slice and reshape
    results = tuple(
        var.recover(primal, dual, reshape_fortran)
        for var in ctx.var_recover
    )

    # Apply exp transformation to recover from log-space for GP
    if ctx.gp:
        results = tuple(torch.exp(r) for r in results)

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = tuple(r.squeeze(0) for r in results)

    return results


class CvxpyLayer(torch.nn.Module):
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
                "P", scipy_csr_to_torch_csr(self.ctx.reduced_P.reduced_mat)  # type: ignore[attr-defined]
            )
        else:
            self.P = None
        self.register_buffer("q", scipy_csr_to_torch_csr(self.ctx.q.tocsr()))
        self.register_buffer(
            "A", scipy_csr_to_torch_csr(self.ctx.reduced_A.reduced_mat)  # type: ignore[attr-defined]
        )

    def forward(
        self, *params: torch.Tensor, solver_args: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, ...]:
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
        q_eval = self.q.to(dtype=param_dtype) @ p_stack
        A_eval = self.A.to(dtype=param_dtype) @ p_stack

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


class _CvxpyLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        P_eval: torch.Tensor | None,
        q_eval: torch.Tensor,
        A_eval: torch.Tensor,
        cl_ctx: pa.LayersContext,
        solver_args: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
        data = cl_ctx.solver_ctx.torch_to_data(P_eval, q_eval, A_eval)
        return *data.torch_solve(solver_args), data

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
        _, _, backwards, data = outputs
        ctx.backwards = backwards
        ctx.data = data
        ctx.device = inputs[1].device
        ctx.dtype = inputs[1].dtype

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, primal: torch.Tensor, dual: torch.Tensor, backwards: Any, data: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None, None]:
        (
            dP,
            dq,
            dA,
        ) = ctx.data.torch_derivative(primal, dual, ctx.backwards)
        if dP is not None:
            dP = torch.as_tensor(dP, device=ctx.device, dtype=ctx.dtype)
            dP = None if len(dP) == 0 else dP
        dA = torch.as_tensor(dA, device=ctx.device, dtype=ctx.dtype)
        dq = torch.as_tensor(dq, device=ctx.device, dtype=ctx.dtype)
        return dP, dq, dA, None, None


def reshape_fortran(x: torch.Tensor, shape: tuple) -> torch.Tensor:
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


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

