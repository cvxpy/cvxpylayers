"""Torch autograd adapters for custom SolverInterface implementations.

Two autograd functions live here:

* ``_CvxpyLayer`` — canonical-matrix-space adapter (receives ``P/q/A`` evals,
  delegates to ``solve_torch_batch`` / ``derivative_torch_batch``).
* ``_ParametricLayer`` — parameter-space adapter for ``is_parametric`` solvers
  (e.g. CVXPYgen).  Takes the raw parameter tensors as inputs, calls the three
  ``_cpg_*`` functions on ``cl_ctx.problem``, and propagates gradients directly
  through ``param.gradient`` — no pseudoinverse required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import cvxpylayers.utils.parse_args as pa


# ---------------------------------------------------------------------------
# Helper — pack CVXPY variable values into (1, n) primal / dual numpy arrays
# ---------------------------------------------------------------------------

def _pack_primal_dual(cl_ctx: pa.LayersContext) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(1, n_primal)`` and ``(1, n_dual)`` arrays from ``var.value``.

    After a parametric solve, each ``var.value`` is set to the solution.  This
    function packs those values into the flat primal/dual vectors that
    :func:`~cvxpylayers.torch.cvxpylayer._recover_results` expects.
    """
    assert cl_ctx.variables is not None
    primal_size = max(
        (v.primal.stop for v in cl_ctx.var_recover if v.primal is not None),
        default=0,
    )
    dual_size = max(
        (v.dual.stop for v in cl_ctx.var_recover if v.dual is not None),
        default=0,
    )
    primal = np.zeros((1, primal_size))
    dual   = np.zeros((1, dual_size))
    for var_info, cvxpy_var in zip(cl_ctx.var_recover, cl_ctx.variables):
        val = np.asarray(cvxpy_var.value)
        if var_info.source == "primal" and var_info.primal is not None:
            primal[0, var_info.primal] = val.flatten(order="F")
        elif var_info.source == "dual" and var_info.dual is not None:
            dual[0, var_info.dual] = val.flatten(order="F")
    return primal, dual


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

    # -----------------------------------------------------------------------
    # _ParametricLayer — autograd Function for is_parametric SolverInterface
    # -----------------------------------------------------------------------

    class _ParametricLayer(_torch.autograd.Function):
        """Autograd Function for parameter-space solvers (e.g. CVXPYgen).

        Takes the original parameter tensors (not canonical matrices) as inputs
        so that :meth:`backward` can return per-parameter gradients directly,
        without a pseudoinverse.

        Call signature::

            _ParametricLayer.apply(*params, cl_ctx, solver_args, needs_grad)

        where ``*params`` are the user-facing torch tensors in the same order
        as ``CvxpyLayer``'s ``parameters`` argument.  ``cl_ctx.problem`` must
        already have ``param.value`` set before ``apply`` is called.
        """

        @staticmethod
        def forward(*args: Any) -> tuple[Any, Any, Any, bool]:
            *param_tensors, cl_ctx, solver_args, needs_grad = args
            solver = cl_ctx.solver
            problem = cl_ctx.problem  # param.value already set by CvxpyLayer

            if needs_grad and solver._cpg_solve_and_gradient is not None:
                _, cpg_grad_primal, cpg_grad_dual = solver._cpg_solve_and_gradient(
                    problem
                )
                grad_info = (cpg_grad_primal, cpg_grad_dual)
            else:
                solver._cpg_solve(problem)
                grad_info = None

            primal_np, dual_np = _pack_primal_dual(cl_ctx)
            dtype  = param_tensors[0].dtype  if param_tensors else _torch.float64
            device = param_tensors[0].device if param_tensors else _torch.device("cpu")
            primal = _torch.tensor(primal_np, dtype=dtype, device=device)
            dual   = _torch.tensor(dual_np,   dtype=dtype, device=device)
            return primal, dual, grad_info, False

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
            *param_tensors, cl_ctx, solver_args, needs_grad = inputs
            _primal, _dual, grad_info, _ub = outputs
            ctx.cl_ctx        = cl_ctx
            ctx.grad_info     = grad_info
            ctx.param_shapes  = [p.shape  for p in param_tensors]
            ctx.param_dtypes  = [p.dtype  for p in param_tensors]
            ctx.param_devices = [p.device for p in param_tensors]

        @staticmethod
        @_torch.autograd.function.once_differentiable
        def backward(
            ctx: Any,
            dprimal: _torch.Tensor,
            ddual:   _torch.Tensor,
            _grad_info: Any,
            _ub: Any,
        ) -> tuple:
            cl_ctx = ctx.cl_ctx
            solver = cl_ctx.solver
            assert cl_ctx.variables is not None

            # Unpack dprimal/ddual into CVXPY variable .gradient attributes.
            for var_info, cvxpy_var in zip(cl_ctx.var_recover, cl_ctx.variables):
                if var_info.source == "primal" and var_info.primal is not None:
                    g = dprimal[0, var_info.primal].detach().cpu().numpy()
                    cvxpy_var.gradient = g.reshape(cvxpy_var.shape, order="F")
                elif var_info.source == "dual" and var_info.dual is not None:
                    g = ddual[0, var_info.dual].detach().cpu().numpy()
                    cvxpy_var.gradient = g.reshape(cvxpy_var.shape, order="F")

            # Run the solver's backward pass; sets param.gradient for each param.
            cpg_grad_primal, cpg_grad_dual = (
                ctx.grad_info if ctx.grad_info is not None else (None, None)
            )
            solver._cpg_gradient(cl_ctx.problem, cpg_grad_primal, cpg_grad_dual)

            # Collect param.gradient → per-input-tensor gradients.
            param_grads: list[_torch.Tensor] = []
            for param_obj, shape, dtype, device in zip(
                cl_ctx.parameters,
                ctx.param_shapes,
                ctx.param_dtypes,
                ctx.param_devices,
            ):
                g_np = np.asarray(param_obj.gradient)
                param_grads.append(
                    _torch.tensor(g_np.reshape(shape), dtype=dtype, device=device)
                )

            # One gradient per input to apply(): *params + cl_ctx/args/ng
            return (*param_grads, None, None, None)

except ImportError:
    _CvxpyLayer = None       # type: ignore[assignment,misc]
    _ParametricLayer = None  # type: ignore[assignment,misc]
