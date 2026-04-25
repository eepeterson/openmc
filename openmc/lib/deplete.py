"""Ctypes bindings for C++ depletion solvers."""

from ctypes import c_int, c_double, c_char_p
import numbers
import os

import numpy as np
from numpy.ctypeslib import ndpointer

from .error import _error_handler
from . import _dll

_array_1d_int = ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
_array_1d_dbl = ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')

# --- CRAM single-material solve ---

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
    c_int,          # substeps
    c_int,          # is_decay
    _array_1d_dbl,  # result
]

_dll.openmc_load_depletion_chain.restype = c_int
_dll.openmc_load_depletion_chain.errcheck = _error_handler
_dll.openmc_load_depletion_chain.argtypes = [c_char_p]


def load_depletion_chain(filename):
    """Load a depletion chain XML file into the C++ runtime.

    Parameters
    ----------
    filename : str or path-like
        Path to chain XML file.

    """
    _dll.openmc_load_depletion_chain(str(os.fspath(filename)).encode())


def cram_solve(A, n0, dt, order=48, substeps=1, is_decay=False):
    """Solve a single Bateman system using C++ CRAM.

    Parameters
    ----------
    A : scipy.sparse.csc_array or scipy.sparse.csc_matrix or None
        Sparse transmutation matrix in CSC format. May be ``None`` when
        ``is_decay`` is True.
    n0 : numpy.ndarray
        Initial atom number vector.
    dt : float
        Time step in seconds.
    order : int
        CRAM approximation order (16 or 48).
    substeps : int
        Number of equal substeps to use within ``dt``.
    is_decay : bool
        If True, use the cached pure-decay solver built from the loaded
        depletion chain (``A`` is ignored). Call :func:`load_depletion_chain`
        first.

    Returns
    -------
    numpy.ndarray
        Final atom numbers.

    """
    if order not in (16, 48):
        raise ValueError(f"CRAM order must be 16 or 48, got {order}")
    if not isinstance(substeps, numbers.Integral):
        raise TypeError(f"substeps must be an integer, got {type(substeps)}")
    if substeps <= 0:
        raise ValueError(f"substeps must be positive, got {substeps}")

    n0 = np.asarray(n0, dtype=np.float64)

    if is_decay:
        n = n0.size
        # Pass empty arrays as placeholders; C++ ignores them when is_decay.
        indptr = np.zeros(n + 1, dtype=np.int32)
        indices = np.zeros(0, dtype=np.int32)
        data = np.zeros(0, dtype=np.float64)
    else:
        if A is None:
            raise ValueError("A must not be None when is_decay is False")
        n = A.shape[0]
        indptr = np.asarray(A.indptr, dtype=np.int32)
        indices = np.asarray(A.indices, dtype=np.int32)
        data = np.asarray(A.data, dtype=np.float64)

    result = np.empty(n, dtype=np.float64)

    _dll.openmc_cram_solve(
        n, indptr, indices, data, n0, dt, order, int(substeps),
        int(bool(is_decay)), result)

    return result
