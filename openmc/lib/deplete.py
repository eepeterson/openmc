"""Ctypes bindings for C++ depletion solvers."""

from ctypes import c_int, c_double

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
    _array_1d_dbl,  # result
]

_dll.openmc_cram_solve_decay.restype = c_int
_dll.openmc_cram_solve_decay.errcheck = _error_handler
_dll.openmc_cram_solve_decay.argtypes = [
    c_int,          # n
    _array_1d_int,  # indptr
    _array_1d_int,  # indices
    _array_1d_dbl,  # data
    _array_1d_dbl,  # n0
    c_double,       # dt
    c_int,          # order
    _array_1d_int,  # perm
    _array_1d_dbl,  # result
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
    n = A.shape[0]
    indptr = np.asarray(A.indptr, dtype=np.int32)
    indices = np.asarray(A.indices, dtype=np.int32)
    data = np.asarray(A.data, dtype=np.float64)
    n0_arr = np.asarray(n0, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)

    if perm is not None:
        perm_arr = np.asarray(perm, dtype=np.int32)
        _dll.openmc_cram_solve_decay(
            n, indptr, indices, data, n0_arr, dt, order, perm_arr, result)
    else:
        _dll.openmc_cram_solve(
            n, indptr, indices, data, n0_arr, dt, order, result)
    return result
