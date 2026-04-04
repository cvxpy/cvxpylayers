"""Torch autograd adapter for custom SolverInterface implementations.

A single ``_CvxpyLayer`` autograd Function handles both solver kinds:

* **Canonical-matrix solvers** (``is_parametric=False``): called as
  ``_CvxpyLayer.apply(P_eval, q_eval, A_eval, cl_ctx, solver_args, needs_grad,
  warm_start)``.  Receives ``P/q/A`` evals, delegates to
  ``solve_torch_batch`` / ``derivative_torch_batch``.

* **Parameter-space solvers** (``is_parametric=True``, e.g. CVXPYgen): called as
  ``_CvxpyLayer.apply(None, None, None, cl_ctx, solver_args, needs_grad, None,
  *params)``.  Takes the raw parameter tensors as trailing variadic inputs,
  calls ``_cpg_solve`` / ``_cpg_solve_and_gradient`` / ``_cpg_gradient``, and
  propagates gradients directly through ``param.gradient`` — no pseudoinverse
  required.  Because ``*params`` are extra inputs, ``backward`` returns
  ``(None,) * 7 + tuple(param_grads)`` so autograd matches one gradient per
  input.
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
        """Unified autograd Function for custom SolverInterface solvers.

        Call signatures
        ---------------
        Canonical-matrix path (``is_parametric=False``)::

            _CvxpyLayer.apply(P_eval, q_eval, A_eval, cl_ctx,
                              solver_args, needs_grad, warm_start)

        Parameter-space path (``is_parametric=True``)::

            _CvxpyLayer.apply(None, None, None, cl_ctx,
                              solver_args, needs_grad, None, *params)

        The trailing ``*params`` are the user-facing torch tensors.  Adding
        them as extra inputs lets ``backward`` return per-param gradients
        directly without changing the fixed-argument interface.
        """

        @staticmethod
        def forward(
            P_eval: _torch.Tensor | None,
            q_eval: _torch.Tensor | None,
            A_eval: _torch.Tensor | None,
            cl_ctx: pa.LayersContext,
            solver_args: dict[str, Any],
            needs_grad: bool = True,
            warm_start: Any = None,
            *params: _torch.Tensor,
        ) -> tuple[_torch.Tensor, _torch.Tensor, Any, bool]:
            solver = cl_ctx.solver

            if getattr(solver, "is_parametric", False):
                # ----------------------------------------------------------
                # Parameter-space path (e.g. CVXPYgen)
                # ----------------------------------------------------------
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
                dtype  = params[0].dtype  if params else _torch.float64
                device = params[0].device if params else _torch.device("cpu")
                primal = _torch.tensor(primal_np, dtype=dtype, device=device)
                dual   = _torch.tensor(dual_np,   dtype=dtype, device=device)
                return primal, dual, grad_info, False

            # ------------------------------------------------------------------
            # Canonical-matrix path
            # ------------------------------------------------------------------
            originally_unbatched = q_eval.dim() == 1  # type: ignore[union-attr]

            # Normalise: (n,) → (1, n)  or  (n, B) → (B, n)
            def _to_bf(x: _torch.Tensor | None) -> _torch.Tensor | None:
                if x is None:
                    return None
                return x.unsqueeze(0) if x.dim() == 1 else x.T

            P_bf = _to_bf(P_eval)
            q_bf = _to_bf(q_eval)
            A_bf = _to_bf(A_eval)

            primal, dual, saved_state = solver.solve_torch_batch(  # type: ignore[union-attr]
                P_bf, q_bf, A_bf,
                cl_ctx.cone_dims,
                {**solver_args},
                needs_grad,
            )
            return primal, dual, saved_state, originally_unbatched

        @staticmethod
        def setup_context(
            ctx: Any,
            inputs: tuple,
            outputs: tuple,
        ) -> None:
            P_eval, q_eval, A_eval, cl_ctx, _, _, _, *params = inputs
            _, _, saved_state_or_grad_info, originally_unbatched = outputs

            solver = cl_ctx.solver
            ctx.is_parametric = getattr(solver, "is_parametric", False)

            if ctx.is_parametric:
                ctx.cl_ctx        = cl_ctx
                ctx.grad_info     = saved_state_or_grad_info
                ctx.param_shapes  = [p.shape  for p in params]
                ctx.param_dtypes  = [p.dtype  for p in params]
                ctx.param_devices = [p.device for p in params]
            else:
                ctx.custom_solver        = solver
                ctx.saved_state          = saved_state_or_grad_info
                ctx.originally_unbatched = originally_unbatched

        @staticmethod
        @_torch.autograd.function.once_differentiable
        def backward(
            ctx: Any,
            dprimal: _torch.Tensor,
            ddual: _torch.Tensor,
            _adj: Any,
            _ub: Any,
        ) -> tuple:
            if ctx.is_parametric:
                # --------------------------------------------------------------
                # Parameter-space backward
                # --------------------------------------------------------------
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

                # 7 Nones for the fixed inputs (P_eval, q_eval, A_eval,
                # cl_ctx, solver_args, needs_grad, warm_start), then one
                # gradient per trailing param tensor.
                return (None, None, None, None, None, None, None, *param_grads)

            # ------------------------------------------------------------------
            # Canonical-matrix backward
            # ------------------------------------------------------------------
            dP_bf, dq_bf, dA_bf = ctx.custom_solver.derivative_torch_batch(
                dprimal, ddual, ctx.saved_state,
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
                # no trailing param gradients for canonical-matrix path
            )

except ImportError:
    _CvxpyLayer = None  # type: ignore[assignment,misc]
