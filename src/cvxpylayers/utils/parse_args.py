from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import cvxpy as cp
import scipy.sparse
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg

import cvxpylayers.interfaces
from cvxpylayers.utils.dgp_reduction import _Dgp2DcpNoValueCheck

if TYPE_CHECKING:
    import torch

T = TypeVar("T")


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


@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None

    def recover(self, primal_sol: T, dual_sol: T) -> T:
        if self.primal is not None:
            # Use ellipsis slicing to handle both batched and unbatched
            return primal_sol[..., self.primal]  # type: ignore[index]
        if self.dual is not None:
            return dual_sol[..., self.dual]  # type: ignore[index]
        raise RuntimeError(
            "Invalid VariableRecovery: both primal and dual slices are None. "
            "At least one must be set to recover variable values."
        )


@dataclass
class LayersContext:
    parameters: list[cp.Parameter]
    reduced_P: scipy.sparse.csr_array
    q: scipy.sparse.csr_array | None
    reduced_A: scipy.sparse.csr_array
    cone_dims: dict[str, int | list[int]]
    solver_ctx: SolverContext
    var_recover: list[VariableRecovery]
    user_order_to_col_order: dict[int, int]
    batch_sizes: list[int] | None = (
        None  # Track which params are batched (0=unbatched, N=batch size)
    )
    # GP (Geometric Programming) support
    gp: bool = False
    # Maps original GP parameters to their log-space DCP parameters
    # Used to determine which parameters need log transformation in forward pass
    gp_param_to_log_param: dict[cp.Parameter, cp.Parameter] | None = None

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


def _validate_problem(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    gp: bool,
) -> None:
    """Validate that the problem is DPP-compliant and inputs are well-formed.

    Args:
        problem: CVXPY problem to validate
        variables: List of CVXPY variables to track
        parameters: List of CVXPY parameters
        gp: Whether this is a geometric program (GP)

    Raises:
        ValueError: If problem is not DPP-compliant or inputs are invalid
    """
    # Check if problem follows disciplined parametrized programming (DPP) rules
    if gp:
        if not problem.is_dgp(dpp=True):  # type: ignore[call-arg]
            raise ValueError("Problem must be DPP for geometric programming.")
    else:
        if not problem.is_dcp(dpp=True):  # type: ignore[call-arg]
            raise ValueError("Problem must be DPP.")

    # Validate parameters match problem definition
    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match problem.parameters")
    if not set(variables).issubset(set(problem.variables())):
        raise ValueError("Argument variables must be a subset of problem.variables")
    if not isinstance(parameters, list) and not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as a list or tuple")
    if not isinstance(variables, list) and not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as a list or tuple")


def _build_user_order_mapping(
    parameters: list[cp.Parameter],
    param_prob: ParamConeProg,
    gp: bool,
    gp_param_to_log_param: dict[cp.Parameter, cp.Parameter] | None,
) -> dict[int, int]:
    """Build mapping from user parameter order to column order.

    CVXPY internally reorders parameters when canonicalizing problems. This
    creates a mapping from the user's parameter order to the internal column
    order used in the canonical form.

    Args:
        parameters: List of CVXPY parameters in user order
        param_prob: CVXPY's parametrized problem object
        gp: Whether this is a geometric program
        gp_param_to_log_param: Mapping from GP params to log-space DCP params

    Returns:
        Dictionary mapping user parameter index to column order index
    """
    # For GP problems, we need to use the log-space DCP parameter IDs
    if gp and gp_param_to_log_param:
        # Map user order index to column using log-space DCP parameters
        user_order_to_col = {
            i: param_prob.param_id_to_col[
                gp_param_to_log_param[p].id if p in gp_param_to_log_param else p.id
            ]
            for i, p in enumerate(parameters)
        }
    else:
        # Standard DCP problem - use original parameters
        user_order_to_col = {
            i: col
            for col, i in sorted(
                [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(parameters)],
            )
        }

    # Convert column indices to sequential order mapping
    user_order_to_col_order = {}
    for j, i in enumerate(user_order_to_col.keys()):
        user_order_to_col_order[i] = j

    return user_order_to_col_order


def parse_args(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    solver: str | None,
    gp: bool = False,
    verbose: bool = False,
    canon_backend: str | None = None,
    solver_args: dict[str, Any] | None = None,
) -> LayersContext:
    # Validate problem is DPP (disciplined parametrized programming)
    _validate_problem(problem, variables, parameters, gp)

    if solver is None:
        solver = "DIFFCP"

    # Handle GP problems using our custom reduction
    gp_param_to_log_param = None
    if gp:
        # Apply custom DGP→DCP reduction that doesn't require parameter values
        dgp2dcp = _Dgp2DcpNoValueCheck()
        dcp_problem, _ = dgp2dcp.apply(problem)

        # Extract parameter mapping from the reduction
        gp_param_to_log_param = dgp2dcp.canon_methods._parameters

        # Get problem data from the already-transformed DCP problem
        data, _, _ = dcp_problem.get_problem_data(
            solver=solver,
            gp=False,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_args,
        )
    else:
        # Standard DCP path
        data, _, _ = problem.get_problem_data(
            solver=solver if solver != 'DIFFQCP' else cp.CLARABEL,
            gp=False,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_args,
        )

    param_prob = data[cp.settings.PARAM_PROB]  # type: ignore[attr-defined]
    cone_dims = data["dims"]

    # Create solver context
    solver_ctx = cvxpylayers.interfaces.get_solver_ctx(
        solver,
        param_prob,
        cone_dims,
        data,
        solver_args,
    )

    # Build parameter ordering mapping
    user_order_to_col_order = _build_user_order_mapping(
        parameters, param_prob, gp, gp_param_to_log_param
    )

    q = getattr(param_prob, "q", getattr(param_prob, "c", None))

    return LayersContext(
        parameters,
        param_prob.reduced_P,
        q,
        param_prob.reduced_A,
        cone_dims,
        solver_ctx,
        var_recover=[
            VariableRecovery(
                slice(start := param_prob.var_id_to_col[v.id], start + v.size),
                None,
            )
            for v in variables
        ],
        user_order_to_col_order=user_order_to_col_order,
        gp=gp,
        gp_param_to_log_param=gp_param_to_log_param,
    )
