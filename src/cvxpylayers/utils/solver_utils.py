from typing import Literal, overload

import numpy as np
import scipy.sparse as sp


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
