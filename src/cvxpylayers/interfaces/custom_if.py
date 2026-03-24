"""Torch autograd adapter for custom SolverInterface implementations.

``_CvxpyLayer`` is the ``torch.autograd.Function`` that bridges
``CvxpyLayer.forward()`` (which always calls ``_CvxpyLayer.apply``) and the
user's ``SolverInterface`` subclass (which exposes ``solve_torch_batch`` /
``derivative_torch_batch`` as its torch entry points).

Input arrays arrive with batch as the **last** dimension (shape ``(n, B)``)
or 1-D for unbatched inputs.  This module normalises them to batch-first
``(B, n)`` before forwarding to the solver, and reverses the transform on the
gradient outputs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cvxpylayers.utils.parse_args as pa


try:
    import torch as _torch

    class _CvxpyLayer(_torch.autograd.Function):
        @staticmethod
        def forward(
            P_eval: _torch.Tensor | None,
            q_eval: _torch.Tensor,
            A_eval: _torch.Tensor,
            cl_ctx: pa.LayersContext,
            solver_args: dict[str, Any],
            needs_grad: bool = True,
            warm_start: Any = None,
        ) -> tuple[_torch.Tensor, _torch.Tensor, Any, bool]:
            """Solve via the custom solver and return batched primal/dual.

            Normalises inputs from batch-last ``(n, B)`` / 1-D to
            batch-first ``(B, n)``, calls ``solve_torch_batch``, and returns
            the results together with an ``originally_unbatched`` flag so
            that the backward pass can squeeze the dummy batch dimension back
            out of the gradients.

            Returns
            -------
            primal : Tensor, shape ``(B, n_primal)``
            dual   : Tensor, shape ``(B, n_dual)``
            adjoint_data : Any  (opaque; passed straight to derivative)
            originally_unbatched : bool  (non-tensor; gradient is None)
            """
            originally_unbatched = q_eval.dim() == 1

            # Normalise: (n,) → (1, n)  or  (n, B) → (B, n)
            def _to_bf(x: _torch.Tensor | None) -> _torch.Tensor | None:
                if x is None:
                    return None
                return x.unsqueeze(0) if x.dim() == 1 else x.T

            P_bf = _to_bf(P_eval)
            q_bf = _to_bf(q_eval)
            A_bf = _to_bf(A_eval)

            primal, dual, adjoint_data = cl_ctx.solver.solve_torch_batch(  # type: ignore[union-attr]
                P_bf, q_bf, A_bf,
                cl_ctx.cone_dims,
                {**solver_args},
                needs_grad,
            )
            return primal, dual, adjoint_data, originally_unbatched

        @staticmethod
        def setup_context(
            ctx: Any,
            inputs: tuple,
            outputs: tuple,
        ) -> None:
            _, _, _, cl_ctx, _, _, _ = inputs
            _, _, adjoint_data, originally_unbatched = outputs
            ctx.custom_solver = cl_ctx.solver
            ctx.adjoint_data = adjoint_data
            ctx.originally_unbatched = originally_unbatched

        @staticmethod
        @_torch.autograd.function.once_differentiable
        def backward(
            ctx: Any,
            dprimal: _torch.Tensor,
            ddual: _torch.Tensor,
            _adj: Any,
            _ub: Any,
        ) -> tuple[
            _torch.Tensor | None,
            _torch.Tensor,
            _torch.Tensor,
            None, None, None, None,
        ]:
            """Propagate gradients via ``derivative_torch_batch``.

            ``dprimal``/``ddual`` are ``(B, n)`` — batch first, matching the
            layout of the primal/dual tensors returned by ``forward``.

            Returns gradients in batch-**last** ``(n, B)`` format (or 1-D for
            originally-unbatched inputs) to match what
            ``_ScipySparseMatmul.backward`` expects.  One ``None`` per
            non-tensor input to ``forward``.
            """
            dP_bf, dq_bf, dA_bf = ctx.custom_solver.derivative_torch_batch(
                dprimal, ddual, ctx.adjoint_data,
            )

            # Transpose (B, n) → (n, B); squeeze trailing dim for unbatched.
            def _to_bl(x: _torch.Tensor | None) -> _torch.Tensor | None:
                if x is None:
                    return None
                t = x.T
                return t.squeeze(1) if ctx.originally_unbatched else t

            return (
                _to_bl(dP_bf),   # dP_eval
                _to_bl(dq_bf),   # dq_eval  (never None)
                _to_bl(dA_bf),   # dA_eval  (never None)
                None,  # cl_ctx
                None,  # solver_args
                None,  # needs_grad
                None,  # warm_start
            )

except ImportError:
    _CvxpyLayer = None  # type: ignore[assignment,misc]
