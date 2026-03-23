"""Ctypes bindings for C++ depletion solvers."""

from ctypes import c_int, c_double, c_bool

import numpy as np
from numpy.ctypeslib import ndpointer

from .error import _error_handler
from . import _dll

_array_1d_int = ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
_array_1d_dbl = ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')

_dll.openmc_cram_solve_batch.restype = c_int
_dll.openmc_cram_solve_batch.errcheck = _error_handler
_dll.openmc_cram_solve_batch.argtypes = [
    c_int,        # n_systems
    c_int,        # order
    _array_1d_int,  # dimensions
    _array_1d_int,  # indptr_offsets
    _array_1d_int,  # all_indptr
    _array_1d_int,  # indices_offsets
    _array_1d_int,  # all_indices
    _array_1d_dbl,  # all_data
    _array_1d_int,  # n0_offsets
    _array_1d_dbl,  # all_n0
    c_double,     # dt
    c_bool,       # decay_only
    _array_1d_dbl,  # all_results
]


def cram_solve_batch(matrices, n0_list, dt, order=48, decay_only=False):
    """Solve multiple independent Bateman systems using C++ CRAM with OpenMP.

    Parameters
    ----------
    matrices : list of scipy.sparse.csc_array
        Depletion matrices (one per material)
    n0_list : list of numpy.ndarray
        Initial atom number vectors (one per material)
    dt : float
        Time step in [s]
    order : int
        CRAM order (16 or 48)
    decay_only : bool
        If True, use the fast decay solver (forward substitution only)

    Returns
    -------
    list of numpy.ndarray
        Updated atom number vectors

    """
    n_systems = len(matrices)

    # Pack dimensions
    dimensions = np.array([m.shape[0] for m in matrices], dtype=np.int32)

    # Pack CSC matrices into contiguous arrays
    # indptr_offsets[i] = start of matrix i's indptr in all_indptr
    # indices_offsets[i] = start of matrix i's indices/data in all_indices/all_data
    indptr_parts = []
    indices_parts = []
    data_parts = []
    indptr_offsets = np.empty(n_systems + 1, dtype=np.int32)
    indices_offsets = np.empty(n_systems + 1, dtype=np.int32)
    indptr_offsets[0] = 0
    indices_offsets[0] = 0

    for i, m in enumerate(matrices):
        ip = np.asarray(m.indptr, dtype=np.int32)
        ix = np.asarray(m.indices, dtype=np.int32)
        d = np.asarray(m.data, dtype=np.float64)
        indptr_parts.append(ip)
        indices_parts.append(ix)
        data_parts.append(d)
        indptr_offsets[i + 1] = indptr_offsets[i] + len(ip)
        indices_offsets[i + 1] = indices_offsets[i] + len(ix)

    all_indptr = np.concatenate(indptr_parts)
    all_indices = np.concatenate(indices_parts)
    all_data = np.concatenate(data_parts)

    # Pack n0 vectors
    n0_offsets = np.empty(n_systems + 1, dtype=np.int32)
    n0_offsets[0] = 0
    for i, n0 in enumerate(n0_list):
        n0_offsets[i + 1] = n0_offsets[i] + len(n0)

    all_n0 = np.concatenate([np.asarray(v, dtype=np.float64) for v in n0_list])
    all_results = np.empty_like(all_n0)

    _dll.openmc_cram_solve_batch(
        n_systems, order,
        dimensions, indptr_offsets, all_indptr,
        indices_offsets, all_indices, all_data,
        n0_offsets, all_n0, dt, decay_only,
        all_results)

    # Unpack results
    return [all_results[n0_offsets[i]:n0_offsets[i + 1]]
            for i in range(n_systems)]
