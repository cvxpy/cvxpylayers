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

            # MPAX only supports QP/LP, not SOCP or other cones
            # Check that cone_dims only has linear inequalities/equalities
            # and no second-order cones, exponential, semidefinite, etc.
            unsupported_cones = []
            if hasattr(cone_dims, "soc") and cone_dims.soc and any(cone_dims.soc):
                unsupported_cones.append("second-order cone (soc)")
            if hasattr(cone_dims, "exp") and cone_dims.exp:
                unsupported_cones.append("exponential cone (exp)")
            if hasattr(cone_dims, "psd") and cone_dims.psd and any(cone_dims.psd):
                unsupported_cones.append("semidefinite cone (psd)")

            if unsupported_cones:
                raise ValueError(
                    f"MPAX solver only supports QP/LP problems with linear "
                    f"inequality and equality constraints. This problem contains "
                    f"{', '.join(unsupported_cones)}, which MPAX cannot handle. "
                    f"Try using solver='DIFFCP' instead, or reformulate your problem "
                    f"to avoid these cone constraints."
                )

            ctx_cls = MPAX_ctx
        case "CUCLARABEL":
            from cvxpylayers.interfaces.cuclarabel_if import (  # type: ignore[import-not-found]
                CUCLARABEL_ctx,
            )

            ctx_cls = CUCLARABEL_ctx
        case "DIFFCP":
            from cvxpylayers.interfaces.diffcp_if import DIFFCP_ctx

            ctx_cls = DIFFCP_ctx
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
