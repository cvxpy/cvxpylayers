from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class CsrProblemData:
    """CSR-format problem structure shared by all non-DIFFCP solvers.

    Bundles the fields that MOREAU, CUCLARABEL, and MPAX all need after
    converting from CVXPY's native CSC layout.
    """

    P_csr_structure: tuple[np.ndarray, np.ndarray] | None  # (col_indices, row_offsets)
    P_shape: tuple[int, int]
    nnz_P: int
    A_csr_structure: tuple[np.ndarray, np.ndarray]  # (col_indices, row_offsets)
    A_shape: tuple[int, int]
    nnz_A: int
    b_idx: np.ndarray


def convert_to_csr(
    param_prob,
) -> tuple["CsrProblemData", sp.sparray | None, sp.sparray | None]:
    """Convert a parametrized problem from CSC to CSR format.

    Reads ``param_prob.reduced_P.problem_data_index`` and
    ``reduced_A.problem_data_index``, calls
    ``convert_csc_structure_to_csr_structure`` for each, and computes
    row-permuted copies of the parametrization matrices so that
    ``matrix @ params`` produces values directly in CSR order.

    This function is pure — it does **not** mutate ``param_prob``.  The
    caller is responsible for assigning the returned matrices back (e.g.
    ``param_prob.reduced_P.reduced_mat = permuted_P_mat``).

    Args:
        param_prob: Parametrized problem from CVXPY canonicalization.
            Must expose ``reduced_P.problem_data_index``,
            ``reduced_P.reduced_mat``, ``reduced_A.problem_data_index``,
            and ``reduced_A.reduced_mat``.

    Returns:
        A 3-tuple ``(csr, permuted_P_mat, permuted_A_mat)`` where *csr*
        is a ``CsrProblemData`` and the matrices are the row-permuted
        parametrization matrices (or ``None`` when the corresponding
        structure is absent).
    """
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

    # --- P matrix ---
    if P_structure_csc is not None:
        P_perm, P_csr_structure, P_shape = convert_csc_structure_to_csr_structure(
            P_structure_csc, False
        )
        nnz_P = len(P_perm)
        permuted_P_mat = param_prob.reduced_P.reduced_mat[P_perm, :]
    else:
        P_csr_structure = None
        P_shape = (n, n)
        nnz_P = 0
        permuted_P_mat = None

    # --- A matrix (with last-column extraction for b) ---
    if A_structure_csc is not None:
        A_perm, A_csr_structure, A_shape, b_idx = (
            convert_csc_structure_to_csr_structure(A_structure_csc, True)
        )
        nnz_A = len(A_perm)
        # Permute constraint rows: [A values in CSR order | b values unchanged]
        nb = param_prob.reduced_A.reduced_mat.shape[0] - nnz_A
        full_A_perm = np.concatenate([A_perm, np.arange(nnz_A, nnz_A + nb)])
        permuted_A_mat = param_prob.reduced_A.reduced_mat[full_A_perm, :]
    else:
        A_csr_structure = (
            np.array([], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
        A_shape = (0, n)
        nnz_A = 0
        b_idx = np.array([], dtype=np.int64)
        permuted_A_mat = None

    csr = CsrProblemData(
        P_csr_structure=P_csr_structure,
        P_shape=P_shape,
        nnz_P=nnz_P,
        A_csr_structure=A_csr_structure,
        A_shape=A_shape,
        nnz_A=nnz_A,
        b_idx=b_idx,
    )
    return csr, permuted_P_mat, permuted_A_mat


@overload
def convert_csc_structure_to_csr_structure(
    structure: tuple[np.ndarray, np.ndarray, tuple[int, int]], extract_last_column: Literal[False]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]: ...


@overload
def convert_csc_structure_to_csr_structure(
    structure: tuple[np.ndarray, np.ndarray, tuple[int, int]], extract_last_column: Literal[True]
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int], np.ndarray]: ...


def convert_csc_structure_to_csr_structure(structure, extract_last_column):
    """
    CVXPY creates matrices in CSC format, many solvers need CSR.
    This converts the CSC structure into the information needed
    to construct CSR.

    Args:
        structure: Tuple of (indices, indptr, (n1, n2)) from CVXPY

    Returns:
        Tuple of (idxs, structure, shape, n) where:
            - idxs: Indices for shuffling the CSC-ordered parameters into the CSR-ordered parameters
            - structure: (indices, indptr) for sparse structure
            - shape: Shape (n1, n2 or n2-1) of matrix
        if extract_last_column is True:
            - b_idxs: indices of the last column
    """
    indices, ptr, (n1, n2) = structure
    if extract_last_column:
        b_idxs = indices[ptr[-2] : ptr[-1]]
        indices = indices[: ptr[-2]]
        ptr = ptr[:-1]
        n2 = n2 - 1

    # Convert to CSR format for efficient row access
    csr = sp.csc_array(
        (np.arange(indices.size), indices, ptr),
        shape=(n1, n2),
    ).tocsr()

    Q_idxs = csr.data
    Q_structure = csr.indices, csr.indptr
    Q_shape = (n1, n2)

    if extract_last_column:
        return Q_idxs, Q_structure, Q_shape, b_idxs
    else:
        return Q_idxs, Q_structure, Q_shape


def JuliaCuVector2CuPyArray(jl, jl_arr):
    """Taken from https://github.com/cvxgrp/CuClarabel/blob/main/src/python/jl2py.py."""
    import cupy as cp

    # Get the device pointer from Julia
    pDevice = jl.Int(jl.pointer(jl_arr))

    # Get array length and element type
    span = jl.size(jl_arr)
    dtype = jl.eltype(jl_arr)

    # Map Julia type to CuPy dtype
    if dtype == jl.Float64:
        dtype = cp.float64
    else:
        dtype = cp.float32

    # Compute memory size in bytes (assuming 1D vector)
    size_bytes = int(span[0] * cp.dtype(dtype).itemsize)

    # Create CuPy memory view from the Julia pointer
    mem = cp.cuda.UnownedMemory(pDevice, size_bytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)

    # Wrap into CuPy ndarray
    arr = cp.ndarray(shape=span, dtype=dtype, memptr=memptr)
    return arr
