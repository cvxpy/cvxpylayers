from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import cvxpy as cp
import scipy.sparse

import cvxpylayers.interfaces

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


@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None

    def recover(self, primal_sol: Any, dual_sol: Any) -> Any:
        if self.primal is not None:
            # Use ellipsis slicing to handle both batched and unbatched
            return primal_sol[..., self.primal]
        if self.dual is not None:
            return dual_sol[..., self.dual]
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


def parse_args(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    solver: str | None,
    kwargs: dict[str, Any],
) -> LayersContext:
    if not problem.is_dcp(dpp=True):  # type: ignore[call-arg]
        raise ValueError("Problem must be DPP.")

    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match problem.parameters")
    if not set(variables).issubset(set(problem.variables())):
        raise ValueError("Argument variables must be a subset of problem.variables")
    if not isinstance(parameters, list) and not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as a list or tuple")
    if not isinstance(variables, list) and not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as a list or tuple")

    if solver is None:
        solver = "DIFFCP"
    data, _, _ = problem.get_problem_data(solver=solver, **kwargs)
    param_prob = data[cp.settings.PARAM_PROB]  # type: ignore[attr-defined]
    cone_dims = data["dims"]

    solver_ctx = cvxpylayers.interfaces.get_solver_ctx(
        solver,
        param_prob,
        cone_dims,
        data,
        kwargs,
    )
    user_order_to_col = {
        i: col
        for col, i in sorted(
            [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(parameters)],
        )
    }
    user_order_to_col_order = {}
    for j, i in enumerate(user_order_to_col.keys()):
        user_order_to_col_order[i] = j

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
    )
