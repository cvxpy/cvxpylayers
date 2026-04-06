"""NVIDIA cuOpt backend for cvxpylayers (forward-only, LP).

Exposes cuOpt as a GPU LP forward solver. Differentiation is not implemented;
for differentiable GPU LPs use ``solver='MOREAU'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (
    dims_to_solver_dict as scs_dims_to_solver_dict,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


if TYPE_CHECKING:
    import cvxpylayers.utils.parse_args as pa


_SUPPORTED_CONES = {"f", "l", "z"}

_DIFF_HINT = (
    "The CUOPT backend is forward-only. For differentiable GPU LPs use "
    "solver='MOREAU'; for differentiable CPU LPs use solver='DIFFCP'."
)


def _validate_cones(cone_dims: dict) -> tuple[int, int]:
    """Ensure cone is a product of zero + nonneg only; return (n_eq, n_ineq)."""
    scs_cones = scs_dims_to_solver_dict(cone_dims)
    n_eq = int(scs_cones.get("f", 0)) + int(scs_cones.get("z", 0))
    n_ineq = int(scs_cones.get("l", 0))
    bad = []
    for k, v in scs_cones.items():
        if k in _SUPPORTED_CONES:
            continue
        if isinstance(v, (list, tuple)):
            if len(v) > 0:
                bad.append(k)
        elif isinstance(v, (int, np.integer)):
            if int(v) > 0:
                bad.append(k)
    if bad:
        raise ValueError(
            "CUOPT backend supports only LP problems (zero + nonneg cones). "
            f"Got unsupported cones: {sorted(set(bad))}."
        )
    return n_eq, n_ineq


if torch is not None:

    class _CvxpyLayer(torch.autograd.Function):
        @staticmethod
        def forward(
            P_eval: Any | None,
            q_eval: Any,
            A_eval: Any,
            cl_ctx: "pa.LayersContext",
            solver_args: dict[str, Any] | None,
        ) -> tuple[Any, Any, Any, Any]:
            solver_ctx: CUOPT_ctx = cl_ctx.solver_ctx  # type: ignore[assignment]
            data = solver_ctx.torch_to_data(
                quad_obj_values=None, lin_obj_values=q_eval, con_values=A_eval
            )
            primal, dual = data.torch_solve(solver_args or {})
            in_dev = q_eval.device
            return primal.to(in_dev), dual.to(in_dev), None, None

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
            pass

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, *grad_outputs: Any) -> tuple:
            raise NotImplementedError(_DIFF_HINT)


def _solve_cuopt_batch(
    qs: list[np.ndarray],
    As_csr: list[sp.csr_array],
    bs: list[np.ndarray],
    n_eq: int,
    n_ineq: int,
    lower_var: np.ndarray,
    upper_var: np.ndarray,
    solver_args: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Solve a batch of LPs using cuOpt.

    Rows [0, n_eq) encode ``Ax = b``; rows [n_eq, n_eq+n_ineq) encode ``Ax <= b``.
    """
    try:
        from cuopt.linear_programming.data_model import DataModel
        from cuopt.linear_programming.solver import Solve
        from cuopt.linear_programming.solver_settings import SolverSettings
    except ImportError as e:
        raise ImportError(
            "cuOpt is required for the CUOPT backend. Install via:\n"
            "  pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13'"
        ) from e

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for q, A_csr, b in zip(qs, As_csr, bs, strict=True):
        n = q.size
        m = A_csr.shape[0]
        assert m == n_eq + n_ineq

        con_lower = np.empty(m, dtype=np.float64)
        con_upper = np.empty(m, dtype=np.float64)
        con_lower[:n_eq] = b[:n_eq]
        con_upper[:n_eq] = b[:n_eq]
        con_lower[n_eq:] = -np.inf
        con_upper[n_eq:] = b[n_eq:]

        # cuOpt's PSLP presolver returns an empty primal vector when the
        # constraint matrix has no structural non-zeros. Augment with a
        # trivial row so cuOpt keeps the variable mapping.
        if A_csr.nnz == 0:
            dummy_row = sp.csr_array(([1e-30], [0], [0, 1]), shape=(1, n))
            A_cuopt = sp.csr_array(sp.vstack([A_csr, dummy_row], format="csr"))
            con_lower = np.concatenate([con_lower, [-np.inf]])
            con_upper = np.concatenate([con_upper, [np.inf]])
        else:
            A_cuopt = A_csr

        dm = DataModel()
        dm.set_csr_constraint_matrix(
            A_cuopt.data,
            A_cuopt.indices.astype(np.int32),
            A_cuopt.indptr.astype(np.int32),
        )
        dm.set_constraint_lower_bounds(con_lower)
        dm.set_constraint_upper_bounds(con_upper)
        dm.set_variable_lower_bounds(lower_var)
        dm.set_variable_upper_bounds(upper_var)
        dm.set_objective_coefficients(q)
        dm.set_maximize(False)

        settings = SolverSettings()
        try:
            settings.set_parameter("presolve", 0)
        except Exception:
            pass
        for k, v in solver_args.items():
            setter = getattr(settings, f"set_{k}", None)
            if callable(setter):
                setter(v)

        sol = Solve(dm, settings)
        x = np.asarray(sol.get_primal_solution(), dtype=np.float64)
        y_raw = np.asarray(sol.get_dual_solution(), dtype=np.float64)[:m]
        # Negate to match cvxpylayers' dual sign convention.
        xs.append(x)
        ys.append(-y_raw)

    return xs, ys


class CUOPT_ctx:
    """Solver context for the cuOpt LP backend (forward-only)."""

    def __init__(
        self,
        objective_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]] | None,
        constraint_structure: tuple[np.ndarray, np.ndarray, tuple[int, int]],
        cone_dims: dict,
        lower_bounds: np.ndarray | None,
        upper_bounds: np.ndarray | None,
        options: dict | None = None,
    ):
        self.dims = cone_dims
        self.options = options or {}
        self.n_eq, self.n_ineq = _validate_cones(cone_dims)

        con_indices, con_ptr, (m, np1) = constraint_structure
        n = np1 - 1
        self.A_shape = (m, n)
        self.b_idxs_np = con_indices[con_ptr[-2] : con_ptr[-1]]

        con_csr = sp.csc_array(
            (np.arange(con_indices.size), con_indices, con_ptr[:-1]),
            shape=(m, n),
        ).tocsr()
        self.A_csr_idxs_np = con_csr.data.astype(np.intp)
        self.A_csr_indices_i32 = con_csr.indices.astype(np.int32)
        self.A_csr_indptr_i32 = con_csr.indptr.astype(np.int32)

        self.lower = (
            np.asarray(lower_bounds, dtype=np.float64)
            if lower_bounds is not None
            else np.full(n, -np.inf, dtype=np.float64)
        )
        self.upper = (
            np.asarray(upper_bounds, dtype=np.float64)
            if upper_bounds is not None
            else np.full(n, np.inf, dtype=np.float64)
        )

    def torch_to_data(
        self,
        quad_obj_values,
        lin_obj_values,
        con_values,
    ) -> "CUOPT_data":
        if con_values.ndim == 1:
            originally_unbatched = True
            con_values_b = con_values.unsqueeze(1)
            lin_b = lin_obj_values.unsqueeze(1)
        else:
            originally_unbatched = False
            con_values_b = con_values
            lin_b = lin_obj_values

        batch_size = con_values_b.shape[1]

        qs: list[np.ndarray] = []
        As_csr: list[sp.csr_array] = []
        bs: list[np.ndarray] = []

        con_np = con_values_b.detach().cpu().numpy().astype(np.float64)
        lin_np = lin_b.detach().cpu().numpy().astype(np.float64)

        m, n = self.A_shape
        for i in range(batch_size):
            con_i = con_np[:, i]
            # cvxpylayers' parametric A has the opposite sign of cvxpy's static
            # get_problem_data A; negate to recover ``Ax ⋛ b`` as cuOpt expects.
            A_data = -con_i[self.A_csr_idxs_np]
            A_csr = sp.csr_array(
                (A_data, self.A_csr_indices_i32, self.A_csr_indptr_i32), shape=(m, n)
            )
            b_vec = np.zeros(m, dtype=np.float64)
            b_vec[self.b_idxs_np] = con_i[-self.b_idxs_np.size :]

            qs.append(lin_np[:-1, i])
            As_csr.append(A_csr)
            bs.append(b_vec)

        return CUOPT_data(
            ctx=self,
            qs=qs,
            As_csr=As_csr,
            bs=bs,
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
        )


@dataclass
class CUOPT_data:
    ctx: CUOPT_ctx
    qs: list
    As_csr: list
    bs: list
    batch_size: int
    originally_unbatched: bool

    def torch_solve(self, solver_args: dict[str, Any]):
        ctx = self.ctx
        xs_np, ys_np = _solve_cuopt_batch(
            qs=self.qs,
            As_csr=self.As_csr,
            bs=self.bs,
            n_eq=ctx.n_eq,
            n_ineq=ctx.n_ineq,
            lower_var=ctx.lower,
            upper_var=ctx.upper,
            solver_args=solver_args,
        )
        primal = torch.stack([torch.from_numpy(x) for x in xs_np])
        dual = torch.stack([torch.from_numpy(y) for y in ys_np])
        return primal, dual

    def torch_derivative(self, *args, **kwargs):
        raise NotImplementedError(_DIFF_HINT)
