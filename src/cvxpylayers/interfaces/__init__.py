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
            print(f"[MLX DEBUG] get_solver_ctx: Selected MPAX solver")
            from cvxpylayers.interfaces.mpax_if import MPAX_ctx

            ctx_cls = MPAX_ctx
        case "CUCLARABEL":
            from cvxpylayers.interfaces.cuclarabel_if import (
                CUCLARABEL_ctx,
            )

            ctx_cls = CUCLARABEL_ctx
        case "DIFFCP":
            print(f"[MLX DEBUG] get_solver_ctx: Selected DIFFCP solver")
            from cvxpylayers.interfaces.diffcp_if import DIFFCP_ctx

            ctx_cls = DIFFCP_ctx
        case _:
            raise RuntimeError(
                "Unknown solver. Check if your solver is supported by CVXPYlayers",
            )
    print(f"[MLX DEBUG] get_solver_ctx: Creating {ctx_cls.__name__} instance")
    return ctx_cls(
        param_prob.reduced_P.problem_data_index,
        param_prob.reduced_A.problem_data_index,
        cone_dims,
        data.get("lower_bound"),
        data.get("upper_bound"),
        kwargs,
    )
