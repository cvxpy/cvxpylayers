def get_solver_ctx(
    solver,
    param_prob,
    cone_dims,
    data,
    kwargs,
):
    ctx_cls = None
    match solver:
        case "MPAX":
            from cvxpylayers.interfaces.mpax_if import MPAX_ctx

            ctx_cls = MPAX_ctx
        case "CUCLARABEL":
            from cvxpylayers.interfaces.cuclarabel_if import (
                CUCLARABEL_ctx,
            )

            ctx_cls = CUCLARABEL_ctx
        case "DIFFCP":
            from cvxpylayers.interfaces.diffcp_if import DIFFCP_ctx

            ctx_cls = DIFFCP_ctx
        case "DIFFQCP":
            from cvxpylayers.interfaces.diffqcp_if import DIFFQCP_CTX

            return DIFFQCP_CTX(
                objective_structure=param_prob.reduced_P.problem_data_index,
                constraint_structure=param_prob.reduced_A.problem_data_index,
                data=data,
                options=kwargs
            )
        case _:
            raise RuntimeError(
                "Unknown solver. Check if your solver is supported by CVXPYlayers",
            )
    return ctx_cls(
        param_prob.reduced_P.problem_data_index,
        param_prob.reduced_A.problem_data_index,
        cone_dims,
        data.get("lower_bound"),
        data.get("upper_bound"),
        kwargs,
    )
