
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
    reduced_P: scipy.sparse.csr_array
    q: scipy.sparse.csr_array
    reduced_A: scipy.sparse.csr_array
    cone_dims: dict[str, int | list[int]]
    var_recover: list[VariableRecovery]
    col_to_user_order: dict[int, int]


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

    return LayersContext(
            param_prob.reduced_P, param_prob.q, param_prob.reduced_A, cone_dims,
            var_recover = [VariableRecovery(slice(start := param_prob.var_id_to_col[v.id], start + v.size)) for v in variables]
            col_to_user_order = {
                col: i for col, i in sorted(
                    [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(param_order)]
                )
            }
    )
