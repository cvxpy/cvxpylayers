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

    def validate_params(self, values):
        if len(values) != len(self.parameters):
            raise ValueError(
                f"A tensor must be provided for each CVXPY parameter; "
                f"received {len(values)} tensors, expected {len(self.parameters)}"
            )
        it = iter(zip(values, self.parameters, strict=True))
        value, param = next(it)
        if len(value.shape) == 0:
            batch = ()
        for i in range(len(value.shape)):
            if value.shape[i:] == param.shape:
                batch = value.shape[:i]
        for value, param in it:
            if value.shape != batch + param.shape:
                raise ValueError(
                    f"Invalid parameter shape. Expected: {batch + param.shape}\nGot: {value.shape}"
                )
        return batch


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
