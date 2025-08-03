import cvxpy as cp
from dataclasses import dataclass

@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None

    def recover(self, primal_sol, dual_sol):
        if primal is not None:
            return primal_sol[primal]
        if dual is not None:
            return dual_sol[dual]
        else:
            raise RuntimeError("")


@dataclass
class LayersContext:
    parameters: list[cp.Parameter]
    reduced_P: scipy.sparse.csr_array
    q: scipy.sparse.csr_array
    reduced_A: scipy.sparse.csr_array
    cone_dims: dict[str, int | list[int]]
    var_recover: list[VariableRecovery]
    user_order_to_col_order: dict[int, int]

    def validate_params(self, values):
        it = iter(zip(values, self.parameters, strict=True))
        value, param = next(it)
        for i in range(len(value.shape)):
            if value.shape[i:] == param.shape:
                batch = value.shape[:i]
        for value, param in it:
            if value.shape != batch + param.shape:
                raise RuntimeError(
                    f"Invalid parameter shape. Expected: {batch+param.shape}\nGot: {value.shape}")
        return batch


def parse_args(problem, variables, parameters, solver, kwargs):
    if not problem.is_dcp(dpp=True):
        raise ValueError('Problem must be DPP.')

    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match "
                         "problem.parameters")
    if not set(variables).issubset(set(problem.variables())):
        raise ValueError("Argument variables must be a subset of "
                         "problem.variables")
    if not isinstance(parameters, list) and \
            not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as "
                             "a list or tuple")
    if not isinstance(variables, list) and \
            not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as "
                         "a list or tuple")

       
    data, _, _ = problem.get_problem_data(solver=solver, **kwargs)
    param_prob = data[cp.settings.PARAM_PROB]
    param_ids = [p.id for p in param_order]
    cone_dims = dims_to_solver_dict(data["dims"])

    solver_ctx = cvxpylayer.interfaces.get_solver_ctx(
        solver, 
        param_prob,
        cone_dims,
        data,
        kwargs,
    )
    user_order_to_col = {
        i: col for col, i in sorted(
            [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(param_order)]
        )
    }
    user_order_to_col_order = {}
    for j, i in enumerate(user_order_to_col.keys()):
        user_order_to_col_order[i]= j

    return LayersContext(
            parameters,
            param_prob.reduced_P, param_prob.q, param_prob.reduced_A, cone_dims,
            solver_ctx,
            var_recover = [VariableRecovery(slice(start := param_prob.var_id_to_col[v.id], start + v.size)) for v in variables],
            user_order_to_col_order=user_order_to_col_order
    )
