def get_solver_ctx(
    solver,
    param_prob,
    cone_dims,
    data,
    kwargs,
):
    ctx_cls = None
    if solver == "MPAX":
        from cvxpylayers.interfaces.mpax_if import MPAX_ctx

        ctx_cls = MPAX_ctx
    elif solver == "CUCLARABEL":
        from cvxpylayers.interfaces.cuclarabel_if import (
            CUCLARABEL_ctx,
        )

        ctx_cls = CUCLARABEL_ctx
    elif solver == "DIFFCP":
        from cvxpylayers.interfaces.diffcp_if import DIFFCP_ctx

        ctx_cls = DIFFCP_ctx
    else:
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
