import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import cvxpy as cp
import numpy as np
import scipy.sparse
from cvxpy.reductions.canonical_maps import Exp, ParameterTransform
from cvxpy.reductions.solvers.canonical_export import build_canonical_maps
from cvxpy.utilities import scopes

import cvxpylayers.interfaces
from cvxpylayers.interfaces.base import SolverInterface

# Solvers whose cvxpylayers interface supports quadratic objectives directly.
# For these solvers, quad_form_dpp_scope is entered during canonicalization
# so that quad_form(x, P) with parametric P is accepted as DPP.
# DIFFCP decomposes quad_form to SOC and cannot handle parametric P.
SUPPORTS_QUAD_OBJ = frozenset({"MOREAU", "CUCLARABEL", "MPAX"})

# Solvers whose inputs come from CVXPY's canonical maps
# (``Problem.get_canonical_maps``). The others still read CVXPY's internal
# parameter tensors until their CVXPY interfaces export canonical maps.
CANONICAL_INPUT_SOLVERS = frozenset({"DIFFCP"})


if TYPE_CHECKING:
    import torch


class SolverData(Protocol):
    """Protocol for data objects returned by solver context."""

    def torch_solve(
        self, solver_args: dict[str, Any] | None = None
    ) -> tuple["torch.Tensor", "torch.Tensor", Any]:
        """Solve the problem using torch backend.

        Returns:
            tuple of (primal, dual, adj_batch) where primal and dual are torch tensors
            and adj_batch is a solver-specific adjoint/backward object.
        """
        ...

    def torch_derivative(
        self, primal: "torch.Tensor", dual: "torch.Tensor", adj_batch: Any
    ) -> tuple["torch.Tensor | None", "torch.Tensor", "torch.Tensor"]:
        """Compute derivatives using torch backend.

        Returns:
            tuple of (dP, dq, dA) gradients as torch tensors (dP can be None).
        """
        ...


class SolverContext(Protocol):
    """Protocol for solver context objects."""

    def torch_to_data(
        self,
        quad_obj_values: "torch.Tensor | None",
        lin_obj_values: "torch.Tensor",
        con_values: "torch.Tensor",
    ) -> SolverData:
        """Convert torch tensors to solver data format."""
        ...


@dataclass(frozen=True)
class OutputRecovery:
    """How to read one requested output from the solver's primal/dual vectors.

    The recovered entries are ``vec[indices]`` (a gather), or, when
    ``indices`` is None, ``out[rows] += vals * vec[cols]`` (a sparse linear
    map with ``size`` outputs). ``exp`` applies elementwise exponentiation
    (geometric programs). When ``scatter`` is given, the entries are the
    stored entries of a sparse or diagonal output, at the Fortran-order
    positions ``scatter`` of the dense output.
    """

    source: Literal["primal", "dual"]
    shape: tuple[int, ...]
    size: int
    indices: np.ndarray | None = None
    rows: np.ndarray | None = None
    cols: np.ndarray | None = None
    vals: np.ndarray | None = None
    exp: bool = False
    scatter: np.ndarray | None = None


@dataclass
class LayersContext:
    """Context holding canonicalized problem data and solver information.

    The forward pass has three stages, all derived from CVXPY's canonical
    maps:

    1. Parameter preparation: evaluate ``param_transforms`` on the user's
       parameter values (read through ``param_sources``) and pack the
       compiled parameters listed in ``param_layout``, followed by a 1.
    2. Solver inputs: ``P_mat``, ``q_mat`` and ``A_mat`` map that stacked
       vector to the values the solver context expects.
    3. Recovery: ``outputs`` read the requested values from the solution.
    """

    parameters: list[cp.Parameter]
    # Solver inputs: each matrix maps [param_vec; 1] to the solver's values.
    P_mat: scipy.sparse.csr_array | None
    q_mat: scipy.sparse.csr_array
    A_mat: scipy.sparse.csr_array
    cone_dims: Any
    solver_ctx: SolverContext | None  # None when a SolverInterface is used
    solver: str | SolverInterface  # SolverInterface instance for custom solvers
    # Parameter preparation.
    param_sources: tuple[np.ndarray | None, ...]
    param_transforms: tuple[ParameterTransform, ...]
    param_layout: tuple[tuple[int, int], ...]  # (compiled parameter ID, size)
    # Recovery of each requested variable.
    outputs: tuple[OutputRecovery, ...]
    batch_sizes: list[int] | None = (
        None  # Track which params are batched (0=unbatched, N=batch size)
    )
    # Original CVXPY problem and variables — used by parameter-space custom
    # solvers (e.g. CVXPYgen) that read var.value / set var.gradient directly.
    problem: cp.Problem | None = None
    variables: list[cp.Variable] | None = None
    gp: bool = False
    # TRANSITIONAL: CVXPY's internal parameter tensors, set only for solvers
    # that do not take canonical maps yet. Custom solvers receive values in
    # this format and may inspect its structure in ``SolverInterface.setup``.
    reduced_P: Any = None
    reduced_A: Any = None
    q: scipy.sparse.sparray | None = None

    def validate_params(self, values: list) -> tuple:
        if len(values) != len(self.parameters):
            raise ValueError(
                f"A tensor must be provided for each CVXPY parameter; "
                f"received {len(values)} tensors, expected {len(self.parameters)}",
            )

        # Determine batch size from all parameters
        batch_sizes = []
        for i, (value, param) in enumerate(zip(values, self.parameters, strict=False)):
            # Check if value has the right shape (with or without batch dimension)
            if len(value.shape) == len(param.shape):
                # No batch dimension for this parameter
                if value.shape != param.shape:
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected: {param.shape}, Got: {value.shape}",
                    )
                batch_sizes.append(0)
            elif len(value.shape) == len(param.shape) + 1:
                # Has batch dimension
                if value.shape[1:] != param.shape:
                    shape_str = ", ".join(map(str, param.shape))
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected batched shape: (batch_size, {shape_str}), "
                        f"Got: {value.shape}",
                    )
                batch_sizes.append(value.shape[0])
            else:
                raise ValueError(
                    f"Invalid parameter dimensionality for parameter {i}. "
                    f"Expected {len(param.shape)} or {len(param.shape) + 1} dimensions, "
                    f"Got: {len(value.shape)} dimensions",
                )

        # Check that all non-zero batch sizes are the same
        nonzero_batch_sizes = [b for b in batch_sizes if b > 0]
        if nonzero_batch_sizes:
            batch_size = nonzero_batch_sizes[0]
            if not all(b == batch_size for b in nonzero_batch_sizes):
                raise ValueError(
                    f"Inconsistent batch sizes. Expected all batched parameters to have "
                    f"the same batch size, but got: {batch_sizes}",
                )
            # Store batch_sizes for use in forward pass
            self.batch_sizes = batch_sizes
            return (batch_size,)
        self.batch_sizes = batch_sizes
        return ()

    @property
    def output_offsets(self) -> tuple[int, ...]:
        """Offsets of the requested variables in their concatenated values.

        Parameter-space solvers (e.g. CVXPYgen) return variable values
        directly; this is the layout used to pass them through autograd.
        """
        assert self.variables is not None
        return tuple(np.cumsum([0] + [v.size for v in self.variables]).tolist())


def _validate_problem(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    gp: bool,
    qp_solver: bool,
) -> None:
    """Validate that the problem is DPP-compliant and inputs are well-formed.

    Args:
        problem: CVXPY problem to validate
        variables: List of CVXPY variables to track (can include constraint dual variables)
        parameters: List of CVXPY parameters
        gp: Whether this is a geometric program (GP)
        qp_solver: Whether the solver handles quadratic objectives directly.
            When True, quad_form(x, P) with parametric P is allowed in the
            objective but rejected in constraints.

    Raises:
        ValueError: If problem is not DPP-compliant or inputs are invalid
    """
    if gp:
        if not problem.is_dpp('dgp'):
            raise ValueError("Problem must be DPP for geometric programming.")
    elif qp_solver:
        if not problem.is_dpp(quad_form_dpp='qp'):  # type: ignore[call-arg]
            raise ValueError(
                "Problem must be DPP. Note: quad_form(x, P) with parametric P "
                "is only supported in the objective, not in constraints."
            )
    else:
        if not problem.is_dpp():
            raise ValueError("Problem must be DPP.")

    # Validate parameters match problem definition
    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match problem.parameters")
    if not isinstance(parameters, list) and not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as a list or tuple")
    if not isinstance(variables, list) and not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as a list or tuple")

    # Validate variables: each must be either a primal variable or a constraint dual variable
    valid_ids = {v.id for v in problem.variables()} | {
        dv.id for con in problem.constraints for dv in con.dual_variables
    }
    for v in variables:
        if v.id not in valid_ids:
            raise ValueError(
                f"Variable {v} must be a subset of problem.variables or a constraint dual variable"
            )


def _output_recovery(source, rec, shape, sparse_idx) -> OutputRecovery:
    """Translate one CVXPY recovery entry into an ``OutputRecovery``."""
    exp = isinstance(rec, Exp)
    if exp:
        rec = rec.indices
    scatter = None
    if sparse_idx is not None:
        scatter = np.ravel_multi_index(sparse_idx, shape, order="F")
    if scipy.sparse.issparse(rec):
        coo = scipy.sparse.coo_array(rec)
        return OutputRecovery(source, shape, coo.shape[0], rows=coo.row, cols=coo.col,
                              vals=coo.data, exp=exp, scatter=scatter)
    return OutputRecovery(source, shape, len(rec), indices=np.asarray(rec), exp=exp,
                          scatter=scatter)


def _with_constant(M: scipy.sparse.sparray, v: np.ndarray) -> scipy.sparse.csr_array:
    """``[M | v]``: the affine map ``M @ p + v`` as a map on ``[p; 1]``."""
    return scipy.sparse.hstack([M, scipy.sparse.csr_array(v[:, None])], format="csr")


def parse_args(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    solver: str | None,
    gp: bool = False,
    verbose: bool = False,
    canon_backend: str | None = None,
    solver_args: dict[str, Any] | None = None,
    custom_solver: Any = None,
) -> LayersContext:
    """Parse and canonicalize a CVXPY problem for use in differentiable layers.

    Args:
        problem: CVXPY problem to canonicalize
        variables: List of variables to return from forward pass
        parameters: List of parameters that will be provided at runtime
        solver: Solver backend to use (DIFFCP, MOREAU, CUCLARABEL, MPAX), or
            the CVXPY canonicalization solver string when custom_solver is set.
        gp: Whether this is a geometric program
        verbose: Whether to print solver output
        canon_backend: Backend for canonicalization
        solver_args: Default solver arguments
        custom_solver: A SolverInterface subclass instance.  When provided,
            ``solver`` is used only for CVXPY canonicalization and
            LayersContext.solver is set to the SolverInterface object itself.

    Returns:
        LayersContext containing canonicalized problem data
    """
    if custom_solver is not None:
        # Use the custom solver's canon_solver as the CVXPY solver string,
        # falling back to whatever the caller passed in solver.
        solver = solver or custom_solver.canon_solver_name
        qp_solver = solver in SUPPORTS_QUAD_OBJ or custom_solver.supports_quad_obj
    else:
        solver = solver or "DIFFCP"
        qp_solver = solver in SUPPORTS_QUAD_OBJ

    _validate_problem(problem, variables, parameters, gp, qp_solver)

    compilation = dict(gp=gp, verbose=verbose, canon_backend=canon_backend,
                       solver_opts=solver_args)
    canonical_inputs = custom_solver is None and solver in CANONICAL_INPUT_SOLVERS
    # Enter quad_form_dpp_scope for QP-capable solvers so that CVXPY caches and
    # canonicalizes under the QP-aware key (parametric quad_form P allowed).
    with (scopes.quad_form_dpp_scope() if qp_solver else contextlib.nullcontext()):
        if canonical_inputs:
            canon = problem.get_canonical_maps(solver, list(variables), **compilation)
        else:
            # TRANSITIONAL: the CVXPY interfaces of these solvers do not export
            # input maps yet, so read CVXPY's internal parameter tensors and
            # take only parameter preparation and recovery from CVXPY.
            data, chain, inverse_data = problem.get_problem_data(
                solver, enforce_dpp=True, **compilation)
            param_prob = data[cp.settings.PARAM_PROB]  # type: ignore[attr-defined]
            canon = build_canonical_maps(problem, chain, param_prob, inverse_data,
                                         list(variables), solver_inputs=False)

    legacy: dict[str, Any] = {}
    if canonical_inputs:
        P_mat = None  # DIFFCP has no quadratic objective.
        q_mat = _with_constant(*canon.maps["c"])
        A_mat = scipy.sparse.vstack([_with_constant(*canon.maps["A"]),
                                     _with_constant(*canon.maps["b"])], format="csr")
        solver_ctx = cvxpylayers.interfaces.get_canonical_solver_ctx(
            solver, canon, solver_args, verbose=verbose)
        layer_solver: str | SolverInterface = solver
    else:
        if custom_solver is not None:
            solver_ctx = None
            layer_solver = custom_solver
        else:
            # May permute the rows of param_prob's tensors (CSR ordering).
            solver_ctx = cvxpylayers.interfaces.get_solver_ctx(
                solver, param_prob, canon.cone_dims, data, solver_args, verbose=verbose)
            layer_solver = solver
        reduced_P = param_prob.reduced_P.reduced_mat
        P_mat = None if reduced_P is None else reduced_P.tocsr()
        q_mat = param_prob.q.tocsr()
        A_mat = param_prob.reduced_A.reduced_mat.tocsr()
        legacy = dict(reduced_P=param_prob.reduced_P, reduced_A=param_prob.reduced_A,
                      q=param_prob.q)

    # Parameter preparation: sparse parameters enter the transforms as their
    # stored entries; the user passes dense values.
    param_sources = tuple(
        None if p.sparse_idx is None
        else np.ravel_multi_index(p.sparse_idx, p.shape, order="F")
        for p in parameters
    )
    layout = sorted(canon.param_id_to_col.items(), key=lambda item: item[1])
    param_layout = tuple((pid, canon.param_id_to_size[pid]) for pid, _ in layout)

    primal_info = {info.var_id: info for info in canon.primal_vars}
    dual_info = {info.dual_var_id: info for info in canon.dual_vars}
    outputs = []
    for v in variables:
        if v.id in primal_info:
            info = primal_info[v.id]
            outputs.append(_output_recovery(
                "primal", canon.primal_recovery[v.id], info.shape, info.sparse_idx))
        else:
            outputs.append(_output_recovery(
                "dual", canon.dual_recovery[v.id], dual_info[v.id].shape, None))

    return LayersContext(
        parameters,
        P_mat,
        q_mat,
        A_mat,
        canon.cone_dims,
        solver_ctx,  # type: ignore[arg-type]
        layer_solver,
        param_sources=param_sources,
        param_transforms=tuple(canon.param_transforms),
        param_layout=param_layout,
        outputs=tuple(outputs),
        problem=problem,
        variables=list(variables),
        gp=gp,
        **legacy,
    )
