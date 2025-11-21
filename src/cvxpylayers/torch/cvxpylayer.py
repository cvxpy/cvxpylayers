from typing import Any, cast

import cvxpy as cp
import scipy.sparse
import torch

import cvxpylayers.utils.parse_args as pa


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
            self.P = torch.nn.Buffer(  # type: ignore[arg-type]
                scipy_csr_to_torch_csr(self.ctx.reduced_P.reduced_mat),  # type: ignore[attr-defined]
            )
        else:
            self.P = None
        self.q = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.q.tocsr()))  # type: ignore[arg-type]
        self.A = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.reduced_A.reduced_mat))  # type: ignore[arg-type, attr-defined]

    def forward(
        self, *params: torch.Tensor, solver_args: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, ...]:
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))
        flattened_params: list[torch.Tensor | None] = [None] * (len(params) + 1)
        for i, param in enumerate(params):
            # Check if this parameter is batched or needs broadcasting
            if self.ctx.batch_sizes[i] == 0 and batch:
                # Unbatched parameter - expand to match batch size
                # Add batch dimension by repeating
                param_expanded = param.unsqueeze(0).expand(batch + param.shape)
                flattened_params[self.ctx.user_order_to_col_order[i]] = reshape_fortran(
                    param_expanded,
                    batch + (-1,),
                )
            else:
                # Already batched or no batch dimension needed
                flattened_params[self.ctx.user_order_to_col_order[i]] = reshape_fortran(
                    param,
                    batch + (-1,),
                )
        flattened_params[-1] = torch.ones(
            batch + (1,),
            dtype=params[0].dtype,
            device=params[0].device,
        )
        # Assert all parameters have been assigned (no Nones remain)
        assert all(p is not None for p in flattened_params), "All parameters must be assigned"
        p_stack = torch.cat(cast(list[torch.Tensor], flattened_params), -1)
        # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
        if batch:
            p_stack = p_stack.T
        P_eval = self.P @ p_stack if self.P is not None else None
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack
        primal, dual, _, _ = _CvxpyLayer.apply(  # type: ignore[misc]
            P_eval,
            q_eval,
            A_eval,
            self.ctx,
            solver_args,
        )
        results = tuple(var.recover(primal, dual) for var in self.ctx.var_recover)

        # Squeeze batch dimension for unbatched inputs
        if not batch:
            results = tuple(r.squeeze(0) for r in results)

        return results


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

