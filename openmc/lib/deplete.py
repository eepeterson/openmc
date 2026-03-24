"""Ctypes bindings for C++ depletion solvers."""

from ctypes import c_int, c_double, c_void_p

import numpy as np
from numpy.ctypeslib import ndpointer

from .error import _error_handler
from . import _dll

_array_1d_int = ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
_array_1d_dbl = ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')

_dll.openmc_cram_solve.restype = c_int
_dll.openmc_cram_solve.errcheck = _error_handler
_dll.openmc_cram_solve.argtypes = [
    c_int,          # n
    _array_1d_int,  # indptr
    _array_1d_int,  # indices
    _array_1d_dbl,  # data
    _array_1d_dbl,  # n0
    c_double,       # dt
    c_int,          # order
    c_void_p,       # perm (nullable)
    _array_1d_dbl,  # result
]

_dll.openmc_cram_solve_batch.restype = c_int
_dll.openmc_cram_solve_batch.errcheck = _error_handler
_dll.openmc_cram_solve_batch.argtypes = [
    c_int,          # n_materials
    _array_1d_int,  # dims
    _array_1d_int,  # all_indptr
    _array_1d_int,  # all_indices
    _array_1d_dbl,  # all_data
    _array_1d_int,  # nnz_per_mat
    _array_1d_dbl,  # all_n0
    c_double,       # dt
    c_int,          # order
    c_void_p,       # perm (nullable)
    _array_1d_dbl,  # all_results
]


def cram_solve(A, n0, dt, order=48, perm=None):
    """Solve a Bateman depletion system using C++ CRAM.

    When *perm* is ``None``, the general IPF CRAM solver is used (full LU
    factorization per pole). When *perm* is provided, the matrix is assumed
    to be a pure-decay matrix that becomes lower-triangular under the given
    topological permutation, and an optimized forward-substitution solver
    is used instead.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        Sparse transmutation matrix (n x n). ``A[j, i]`` is the rate at
        which nuclide *i* transmutes to nuclide *j*.
    n0 : numpy.ndarray
        Initial atom number vector of length *n*.
    dt : float
        Time step in seconds.
    order : int
        CRAM approximation order (16 or 48).
    perm : numpy.ndarray, optional
        Topological permutation vector of length *n*.
        ``perm[new_idx] = old_idx`` reorders a decay matrix into
        lower-triangular form. When provided, the fast triangular
        decay solver is used.

    Returns
    -------
    numpy.ndarray
        Final atom numbers after time *dt*.

    """
    if order not in (16, 48):
        raise ValueError(f"CRAM order must be 16 or 48, got {order}")

    n = A.shape[0]
    indptr = np.asarray(A.indptr, dtype=np.int32)
    indices = np.asarray(A.indices, dtype=np.int32)
    data = np.asarray(A.data, dtype=np.float64)
    n0_arr = np.asarray(n0, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)

    if perm is not None:
        perm_arr = np.ascontiguousarray(perm, dtype=np.int32)
        perm_ptr = perm_arr.ctypes.data
    else:
        perm_ptr = None

    _dll.openmc_cram_solve(
        n, indptr, indices, data, n0_arr, dt, order, perm_ptr, result)
    return result


def cram_solve_batch(matrices, n0_list, dt, order=48, perm=None):
    """Solve multiple Bateman systems in parallel using C++ CRAM with OpenMP.

    Each system is defined by a sparse transmutation matrix and an initial
    atom number vector. All systems are solved over the same time step.
    The matrices may have different dimensions and sparsity patterns.

    Parameters
    ----------
    matrices : list of scipy.sparse.csc_array
        Sparse transmutation matrices, one per material.
    n0_list : list of numpy.ndarray
        Initial atom number vectors, one per material. ``n0_list[m]`` must
        have length equal to ``matrices[m].shape[0]``.
    dt : float
        Time step in seconds (shared by all materials).
    order : int
        CRAM approximation order (16 or 48).
    perm : numpy.ndarray, optional
        Topological permutation vector for the decay solver.

    Returns
    -------
    list of numpy.ndarray
        Final atom numbers for each material.

    """
    if order not in (16, 48):
        raise ValueError(f"CRAM order must be 16 or 48, got {order}")

    n_materials = len(matrices)
    dims = np.array([m.shape[0] for m in matrices], dtype=np.int32)
    nnz_per_mat = np.array(
        [m.indptr[-1] for m in matrices], dtype=np.int32)

    all_indptr = np.concatenate(
        [np.asarray(m.indptr, dtype=np.int32) for m in matrices])
    all_indices = np.concatenate(
        [np.asarray(m.indices, dtype=np.int32) for m in matrices])
    all_data = np.concatenate(
        [np.asarray(m.data, dtype=np.float64) for m in matrices])
    all_n0 = np.concatenate(
        [np.asarray(v, dtype=np.float64) for v in n0_list])
    all_results = np.empty(dims.sum(), dtype=np.float64)

    if perm is not None:
        perm_arr = np.ascontiguousarray(perm, dtype=np.int32)
        perm_ptr = perm_arr.ctypes.data
    else:
        perm_ptr = None

    _dll.openmc_cram_solve_batch(
        n_materials, dims, all_indptr, all_indices, all_data,
        nnz_per_mat, all_n0, dt, order, perm_ptr, all_results)

    # Split concatenated results back into per-material arrays
    offsets = np.cumsum(dims)
    return [all_results[s:e] for s, e in zip(
        np.concatenate([[0], offsets[:-1]]), offsets)]
