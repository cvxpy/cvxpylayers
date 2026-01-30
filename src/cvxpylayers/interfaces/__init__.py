def _merge_verbose(kwargs, verbose):
    """Merge verbose flag into kwargs if set."""
    if verbose:
        options = kwargs.copy() if kwargs else {}
        options["verbose"] = True
        return options
    return kwargs


def get_solver_ctx(
    solver,
    param_prob,
    cone_dims,
    data,
    kwargs,
    verbose=False,
):
    # Merge verbose into options for solvers that support it
    options = _merge_verbose(kwargs, verbose)

    match solver:
        case "MPAX":
            from cvxpylayers.interfaces.mpax_if import MPAX_ctx

            ctx_cls = MPAX_ctx
        case "CUCLARABEL":
            from cvxpylayers.interfaces.cuclarabel_if import CUCLARABEL_ctx

            ctx_cls = CUCLARABEL_ctx
        case "MOREAU":
            from cvxpylayers.interfaces.moreau_if import MOREAU_ctx

            # MOREAU needs actual matrices to detect if P/A are constant
            return MOREAU_ctx(
                param_prob.reduced_P.problem_data_index,
                param_prob.reduced_A.problem_data_index,
                cone_dims,
                options,
                reduced_P_mat=param_prob.reduced_P.reduced_mat,
                reduced_A_mat=param_prob.reduced_A.reduced_mat,
            )
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
        options,
    )


def get_torch_cvxpylayer(solver):
    """Get the _CvxpyLayer class for the given solver.

    Args:
        solver: Solver name string (e.g., "DIFFCP", "MOREAU", "CUCLARABEL", "MPAX")

    Returns:
        The _CvxpyLayer class for the specified solver
    """
    match solver:
        case "MPAX":
            from cvxpylayers.interfaces.mpax_if import _CvxpyLayer

            return _CvxpyLayer
        case "CUCLARABEL":
            from cvxpylayers.interfaces.cuclarabel_if import _CvxpyLayer

            return _CvxpyLayer
        case "MOREAU":
            from cvxpylayers.interfaces.moreau_if import _CvxpyLayer

            return _CvxpyLayer
        case "DIFFCP":
            from cvxpylayers.interfaces.diffcp_if import _CvxpyLayer

            return _CvxpyLayer
        case _:
            raise RuntimeError(
                "Unknown solver. Check if your solver is supported by CVXPYlayers",
            )
