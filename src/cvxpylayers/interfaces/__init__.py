from cvxpylayers.utils.solver_utils import convert_to_csr


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
    options = _merge_verbose(kwargs, verbose)

    if solver == "DIFFCP":
        from cvxpylayers.interfaces.diffcp_if import DIFFCP_ctx

        return DIFFCP_ctx(
            param_prob.reduced_P.problem_data_index,
            param_prob.reduced_A.problem_data_index,
            cone_dims,
            data.get("lower_bound"),
            data.get("upper_bound"),
            options,
        )

    # All non-DIFFCP solvers need CSR format.
    # convert_to_csr is pure; we apply the row permutation here so that
    # downstream matrix multiplies (matrix @ params) produce CSR-ordered values.
    csr, permuted_P_mat, permuted_A_mat = convert_to_csr(param_prob)
    if permuted_P_mat is not None:
        param_prob.reduced_P.reduced_mat = permuted_P_mat
    if permuted_A_mat is not None:
        param_prob.reduced_A.reduced_mat = permuted_A_mat

    match solver:
        case "MOREAU":
            from cvxpylayers.interfaces.moreau_if import MOREAU_ctx

            return MOREAU_ctx(
                csr, cone_dims, options,
                reduced_P_mat=permuted_P_mat,
                reduced_A_mat=permuted_A_mat,
            )
        case "CUCLARABEL":
            from cvxpylayers.interfaces.cuclarabel_if import CUCLARABEL_ctx

            return CUCLARABEL_ctx(csr, cone_dims, options)
        case "MPAX":
            from cvxpylayers.interfaces.mpax_if import MPAX_ctx

            return MPAX_ctx(
                csr, cone_dims,
                lower_bounds=data.get("lower_bound"),
                upper_bounds=data.get("upper_bound"),
                options=options,
            )
        case _:
            raise RuntimeError(
                "Unknown solver. Check if your solver is supported by CVXPYlayers",
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
