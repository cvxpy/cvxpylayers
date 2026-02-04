import numpy as np

from cvxpylayers.utils.solver_utils import convert_csc_structure_to_csr_structure


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
        case "MOREAU" | "CUCLARABEL" | "MPAX":
            # --- Common CSR conversion for all non-DIFFCP solvers ---
            P_structure_csc = param_prob.reduced_P.problem_data_index
            A_structure_csc = param_prob.reduced_A.problem_data_index

            # Determine n (problem dimension) for constructing empty structures
            if P_structure_csc is not None:
                n = P_structure_csc[2][0]  # P is square (n, n)
            elif A_structure_csc is not None:
                n = A_structure_csc[2][1] - 1  # A has n+1 columns (last is b)
            else:
                raise ValueError(
                    "Cannot determine problem dimension: both P and A are None"
                )

            if P_structure_csc is not None:
                P_perm, P_csr_structure, P_shape = convert_csc_structure_to_csr_structure(
                    P_structure_csc, False
                )
                nnz_P = len(P_perm)
                param_prob.reduced_P.reduced_mat = (
                    param_prob.reduced_P.reduced_mat[P_perm, :]
                )
            else:
                P_csr_structure = None
                P_shape = (n, n)
                nnz_P = 0

            if A_structure_csc is not None:
                A_perm, A_csr_structure, A_shape, b_idx = (
                    convert_csc_structure_to_csr_structure(A_structure_csc, True)
                )
                nnz_A = len(A_perm)
                # Permute constraint rows: [A values in CSR order | b values unchanged]
                nb = param_prob.reduced_A.reduced_mat.shape[0] - nnz_A
                full_A_perm = np.concatenate([A_perm, np.arange(nnz_A, nnz_A + nb)])
                param_prob.reduced_A.reduced_mat = (
                    param_prob.reduced_A.reduced_mat[full_A_perm, :]
                )
            else:
                A_csr_structure = (
                    np.array([], dtype=np.int64),
                    np.zeros(1, dtype=np.int64),
                )
                A_shape = (0, n)
                nnz_A = 0
                b_idx = np.array([], dtype=np.int64)

            # --- Per-solver construction ---
            match solver:
                case "MOREAU":
                    from cvxpylayers.interfaces.moreau_if import MOREAU_ctx

                    return MOREAU_ctx(
                        P_csr_structure, P_shape, nnz_P,
                        A_csr_structure, A_shape, nnz_A, b_idx,
                        cone_dims, options,
                        reduced_P_mat=param_prob.reduced_P.reduced_mat,
                        reduced_A_mat=param_prob.reduced_A.reduced_mat,
                    )
                case "CUCLARABEL":
                    from cvxpylayers.interfaces.cuclarabel_if import CUCLARABEL_ctx

                    return CUCLARABEL_ctx(
                        P_csr_structure, P_shape, nnz_P,
                        A_csr_structure, A_shape, nnz_A, b_idx,
                        cone_dims, options,
                    )
                case "MPAX":
                    from cvxpylayers.interfaces.mpax_if import MPAX_ctx

                    return MPAX_ctx(
                        P_csr_structure, P_shape, nnz_P,
                        A_csr_structure, A_shape, nnz_A, b_idx,
                        cone_dims,
                        data.get("lower_bound"), data.get("upper_bound"),
                        options,
                    )

        case "DIFFCP":
            from cvxpylayers.interfaces.diffcp_if import DIFFCP_ctx

            return DIFFCP_ctx(
                param_prob.reduced_P.problem_data_index,
                param_prob.reduced_A.problem_data_index,
                cone_dims,
                data.get("lower_bound"),
                data.get("upper_bound"),
                options,
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
