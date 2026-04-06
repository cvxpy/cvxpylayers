"""NVIDIA cuOpt backend for cvxpylayers (forward-only, LP).

Exposes cuOpt as a GPU LP forward solver. Differentiation is not implemented;
for differentiable GPU LPs use ``solver='MOREAU'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (
    dims_to_solver_dict as scs_dims_to_solver_dict,
)

from cvxpylayers.utils.solver_utils import convert_csc_structure_to_csr_structure

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
            data = solver_ctx.torch_to_data(q_eval, A_eval)
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


def _build_solver_settings(
    solver_args: dict[str, Any],
) -> Any:
    """Build a cuOpt SolverSettings with presolve disabled and args applied."""
    from cuopt.linear_programming.solver_settings import SolverSettings

    settings = SolverSettings()
    # cuOpt's PSLP presolver can return an empty primal vector for LPs with
    # no structural constraints, breaking the dummy-row workaround below.
    settings.set_parameter("presolve", 0)
    for k, v in solver_args.items():
        setter = getattr(settings, f"set_{k}", None)
        if not callable(setter):
            raise ValueError(
                f"Unknown CUOPT solver_args key: {k!r}. Expected a SolverSettings.set_{k} method."
            )
        setter(v)
    return settings


def _solve_cuopt_batch(
    ctx: CUOPT_ctx,
    qs: list[np.ndarray],
    A_datas: list[np.ndarray],
    bs: list[np.ndarray],
    solver_args: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Solve a batch of LPs using cuOpt.

    Rows [0, n_eq) encode ``Ax = b``; rows [n_eq, n_eq+n_ineq) encode ``Ax <= b``.

    NOTE: cuOpt's Python API has evolved across releases. The imports and
    ``set_parameter`` / ``set_*`` calls below are against cuopt-cu13 25.x+.
    """
    try:
        from cuopt.linear_programming.data_model import DataModel
        from cuopt.linear_programming.solver import Solve
    except ImportError as e:
        raise ImportError(
            "cuOpt is required for the CUOPT backend. Install via:\n"
            "  pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13'"
        ) from e

    settings = _build_solver_settings(solver_args)
    m = ctx.n_eq + ctx.n_ineq
    A_indices = ctx.A_cuopt_indices
    A_indptr = ctx.A_cuopt_indptr
    con_lower_tmpl = ctx.con_lower_tmpl
    con_upper_tmpl = ctx.con_upper_tmpl

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for q, A_data, b in zip(qs, A_datas, bs, strict=True):
        con_lower = con_lower_tmpl.copy()
        con_upper = con_upper_tmpl.copy()
        con_lower[: ctx.n_eq] = b[: ctx.n_eq]
        con_upper[: ctx.n_eq] = b[: ctx.n_eq]
        con_upper[ctx.n_eq : m] = b[ctx.n_eq :]

        dm = DataModel()
        dm.set_csr_constraint_matrix(A_data, A_indices, A_indptr)
        dm.set_constraint_lower_bounds(con_lower)
        dm.set_constraint_upper_bounds(con_upper)
        dm.set_variable_lower_bounds(ctx.lower)
        dm.set_variable_upper_bounds(ctx.upper)
        dm.set_objective_coefficients(q)
        dm.set_maximize(False)

        sol = Solve(dm, settings)
        x = np.asarray(sol.get_primal_solution(), dtype=np.float64)
        y_raw = np.asarray(sol.get_dual_solution(), dtype=np.float64)[:m]
        # cuOpt returns the multiplier of `lhs <= Ax <= rhs`; cvxpylayers uses
        # the diffcp convention `Ax + s = b, s ∈ K` with opposite sign.
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
        self.n_eq, self.n_ineq = _validate_cones(cone_dims)

        A_csr_idxs, (A_indices, A_indptr), (m, n), b_idxs = convert_csc_structure_to_csr_structure(
            constraint_structure, extract_last_column=True
        )
        self.A_shape = (m, n)
        self.b_idxs_np = b_idxs
        self.A_csr_idxs_np = A_csr_idxs.astype(np.intp)
        A_indices_i32 = A_indices.astype(np.int32)
        A_indptr_i32 = A_indptr.astype(np.int32)

        # Precompute constraint-bound templates and the cuOpt CSR structure.
        # When the parametric A has no structural non-zeros, cuOpt's PSLP
        # presolver drops variables; augment with a single trivial row to
        # preserve the variable mapping. The dummy is invisible to the LP.
        if A_csr_idxs.size == 0:
            self.A_cuopt_indices = np.concatenate([A_indices_i32, [np.int32(0)]])
            self.A_cuopt_indptr = np.concatenate([A_indptr_i32, [A_indptr_i32[-1] + np.int32(1)]])
            self._A_data_pad = np.array([1e-30], dtype=np.float64)
            self.con_lower_tmpl = np.concatenate([np.full(m, -np.inf), [-np.inf]])
            self.con_upper_tmpl = np.concatenate([np.full(m, np.inf), [np.inf]])
        else:
            self.A_cuopt_indices = A_indices_i32
            self.A_cuopt_indptr = A_indptr_i32
            self._A_data_pad = None
            self.con_lower_tmpl = np.full(m, -np.inf)
            self.con_upper_tmpl = np.full(m, np.inf)

        self.lower = (
            np.ascontiguousarray(lower_bounds, dtype=np.float64)
            if lower_bounds is not None
            else np.full(n, -np.inf, dtype=np.float64)
        )
        self.upper = (
            np.ascontiguousarray(upper_bounds, dtype=np.float64)
            if upper_bounds is not None
            else np.full(n, np.inf, dtype=np.float64)
        )

    def torch_to_data(self, lin_obj_values, con_values) -> "CUOPT_data":
        if con_values.ndim == 1:
            con_values_b = con_values.unsqueeze(1)
            lin_b = lin_obj_values.unsqueeze(1)
        else:
            con_values_b = con_values
            lin_b = lin_obj_values
        batch_size = con_values_b.shape[1]

        con_np = np.ascontiguousarray(con_values_b.detach().cpu().numpy(), dtype=np.float64)
        lin_np = np.ascontiguousarray(lin_b.detach().cpu().numpy(), dtype=np.float64)

        qs: list[np.ndarray] = []
        A_datas: list[np.ndarray] = []
        bs: list[np.ndarray] = []
        m = self.A_shape[0]
        b_idxs = self.b_idxs_np

        for i in range(batch_size):
            con_i = con_np[:, i]
            # cvxpylayers' parametric A has the opposite sign of cvxpy's static
            # get_problem_data A; negate so cuOpt sees ``Ax ⋛ b``.
            A_data = -con_i[self.A_csr_idxs_np]
            if self._A_data_pad is not None:
                A_data = np.concatenate([A_data, self._A_data_pad])

            b_vec = np.zeros(m, dtype=np.float64)
            b_vec[b_idxs] = con_i[-b_idxs.size :]

            qs.append(lin_np[:-1, i])
            A_datas.append(A_data)
            bs.append(b_vec)

        return CUOPT_data(ctx=self, qs=qs, A_datas=A_datas, bs=bs)


@dataclass
class CUOPT_data:
    ctx: CUOPT_ctx
    qs: list
    A_datas: list
    bs: list

    def torch_solve(self, solver_args: dict[str, Any]):
        xs_np, ys_np = _solve_cuopt_batch(self.ctx, self.qs, self.A_datas, self.bs, solver_args)
        primal = torch.from_numpy(np.stack(xs_np))
        dual = torch.from_numpy(np.stack(ys_np))
        return primal, dual
