from typing import Any
import cvxpy as cp
import cvxpylayers.interfaces
import scipy.sparse
from dataclasses import dataclass


@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None

    def recover(self, primal_sol, dual_sol):
        if self.primal is not None:
            return primal_sol[self.primal]
        if self.dual is not None:
            return dual_sol[self.dual]
        else:
            raise RuntimeError("")


@dataclass
class LayersContext:
    parameters: list[cp.Parameter]
    reduced_P: scipy.sparse.csr_array
    q: scipy.sparse.csr_array
    reduced_A: scipy.sparse.csr_array
    cone_dims: dict[str, int | list[int]]
    solver_ctx: object
    var_recover: list[VariableRecovery]
    user_order_to_col_order: dict[int, int]
    batch_sizes: list[int] = None  # Track which params are batched (0=unbatched, N=batch size)

    def validate_params(self, values):
        if len(values) != len(self.parameters):
            raise ValueError(
                f"A tensor must be provided for each CVXPY parameter; "
                f"received {len(values)} tensors, expected {len(self.parameters)}"
            )

        # Determine batch size from all parameters
        batch_sizes = []
        for i, (value, param) in enumerate(zip(values, self.parameters)):
            # Check if value has the right shape (with or without batch dimension)
            if len(value.shape) == len(param.shape):
                # No batch dimension for this parameter
                if value.shape != param.shape:
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected: {param.shape}, Got: {value.shape}"
                    )
                batch_sizes.append(0)
            elif len(value.shape) == len(param.shape) + 1:
                # Has batch dimension
                if value.shape[1:] != param.shape:
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected batched shape: (batch_size, {', '.join(map(str, param.shape))}), "
                        f"Got: {value.shape}"
                    )
                batch_sizes.append(value.shape[0])
            else:
                raise ValueError(
                    f"Invalid parameter dimensionality for parameter {i}. "
                    f"Expected {len(param.shape)} or {len(param.shape) + 1} dimensions, "
                    f"Got: {len(value.shape)} dimensions"
                )

        # Check that all non-zero batch sizes are the same
        nonzero_batch_sizes = [b for b in batch_sizes if b > 0]
        if nonzero_batch_sizes:
            batch_size = nonzero_batch_sizes[0]
            if not all(b == batch_size for b in nonzero_batch_sizes):
                raise ValueError(
                    f"Inconsistent batch sizes. Expected all batched parameters to have "
                    f"the same batch size, but got: {batch_sizes}"
                )
            # Store batch_sizes for use in forward pass
            self.batch_sizes = batch_sizes
            return (batch_size,)
        else:
            self.batch_sizes = batch_sizes
            return ()


def parse_args(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    solver: str,
    kwargs: dict[str, Any],
):
    if not problem.is_dcp(dpp=True):
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
    param_prob = data[cp.settings.PARAM_PROB]
    param_ids = [p.id for p in parameters]
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
            [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(parameters)]
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
                slice(start := param_prob.var_id_to_col[v.id], start + v.size), None
            )
            for v in variables
        ],
        user_order_to_col_order=user_order_to_col_order,
    )
