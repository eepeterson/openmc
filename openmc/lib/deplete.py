"""Ctypes bindings for C++ depletion solvers."""

from ctypes import c_int, c_double, c_void_p, c_char_p, POINTER, byref

import numpy as np
from numpy.ctypeslib import ndpointer

from .error import _error_handler
from . import _dll

_array_1d_int = ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
_array_1d_dbl = ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')

# --- CRAM batch solve (standalone utility) ---

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


# --- CRAM matrix exponential ---

_dll.openmc_decay_reachability.restype = c_int
_dll.openmc_decay_reachability.errcheck = _error_handler
_dll.openmc_decay_reachability.argtypes = [
    c_int,          # n
    _array_1d_int,  # indptr
    _array_1d_int,  # indices
    _array_1d_dbl,  # data
    c_void_p,       # perm
    c_void_p,       # reach_indptr (nullable)
    c_void_p,       # reach_indices (nullable)
    POINTER(c_int), # total_reach
]

_dll.openmc_cram_expm.restype = c_int
_dll.openmc_cram_expm.errcheck = _error_handler
_dll.openmc_cram_expm.argtypes = [
    c_int,          # n
    _array_1d_int,  # indptr
    _array_1d_int,  # indices
    _array_1d_dbl,  # data
    c_double,       # dt
    c_int,          # order
    c_double,       # drop_tol
    c_void_p,       # perm
    c_void_p,       # reach_indptr (nullable)
    c_void_p,       # reach_indices (nullable)
    c_void_p,       # out_indptr
    c_void_p,       # out_indices (nullable for query)
    c_void_p,       # out_data (nullable for query)
    POINTER(c_int), # out_nnz
]


def compute_reachability(matrix, perm):
    """Compute structural reachability for a decay matrix.

    For each column j (in permuted space), computes the set of row indices
    reachable via transitive closure of the decay DAG. This determines the
    nonzero pattern of exp(A*dt) and is invariant for a given chain topology.
    The result can be passed to :func:`cram_expm` to skip recomputing
    reachability on every call.

    Parameters
    ----------
    matrix : scipy.sparse.csc_array or csc_matrix
        Sparse decay matrix A in CSC format.
    perm : numpy.ndarray
        Topological permutation vector: ``perm[new_idx] = old_idx``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(reach_indptr, reach_indices)`` — flat CSC-like arrays where
        ``reach_indptr[j]..reach_indptr[j+1]`` indexes into
        ``reach_indices`` for column j's reachable rows.

    """
    from scipy.sparse import csc_array

    matrix = csc_array(matrix)
    n = matrix.shape[0]
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    perm_arr = np.ascontiguousarray(perm, dtype=np.int32)
    perm_ptr = perm_arr.ctypes.data

    # Query call: get total_reach and reach_indptr
    reach_indptr = np.empty(n + 1, dtype=np.int32)
    total_reach = c_int(0)
    _dll.openmc_decay_reachability(
        n, indptr, indices, data, perm_ptr,
        reach_indptr.ctypes.data, None, byref(total_reach))

    # Fill call: get reach_indices
    reach_indices = np.empty(total_reach.value, dtype=np.int32)
    _dll.openmc_decay_reachability(
        n, indptr, indices, data, perm_ptr,
        reach_indptr.ctypes.data, reach_indices.ctypes.data,
        byref(total_reach))

    return reach_indptr, reach_indices


def cram_expm(matrix, dt, perm, order=48, drop_tol=0.0, reach=None):
    """Compute the matrix exponential exp(A*dt) of a decay matrix using CRAM.

    Uses the decay-optimized solver which exploits topological permutation
    to lower-triangular form for forward substitution with DAG sparsity.

    Parameters
    ----------
    matrix : scipy.sparse.csc_array or csc_matrix
        Sparse decay matrix A in CSC format.
    dt : float
        Time step in seconds.
    perm : numpy.ndarray
        Topological permutation vector: ``perm[new_idx] = old_idx``.
        Must reorder the decay matrix into lower-triangular form.
    order : int
        CRAM approximation order (16 or 48).
    drop_tol : float
        Drop entries with absolute value below this threshold.
    reach : tuple of numpy.ndarray, optional
        Precomputed reachability from :func:`compute_reachability`. When
        provided, skips the internal reachability computation and enables
        a single C++ call instead of two (the output nnz is known upfront).

    Returns
    -------
    scipy.sparse.csr_array
        Sparse matrix exponential exp(A*dt) in CSR format.

    """
    from scipy.sparse import csc_array, csr_array

    if order not in (16, 48):
        raise ValueError(f"CRAM order must be 16 or 48, got {order}")

    matrix = csc_array(matrix)
    n = matrix.shape[0]
    indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    data = np.ascontiguousarray(matrix.data, dtype=np.float64)

    perm_arr = np.ascontiguousarray(perm, dtype=np.int32)
    perm_ptr = perm_arr.ctypes.data

    if reach is not None:
        reach_indptr, reach_indices = reach
        reach_indptr = np.ascontiguousarray(reach_indptr, dtype=np.int32)
        reach_indices = np.ascontiguousarray(reach_indices, dtype=np.int32)

        # Pre-allocate output: max nnz = n + total_reach
        max_nnz = n + len(reach_indices)
        out_indptr = np.empty(n + 1, dtype=np.int32)
        out_indices = np.empty(max_nnz, dtype=np.int32)
        out_data = np.empty(max_nnz, dtype=np.float64)
        out_nnz = c_int(0)

        # Single C++ call with precomputed reachability
        _dll.openmc_cram_expm(
            n, indptr, indices, data, dt, order, drop_tol,
            perm_ptr, reach_indptr.ctypes.data,
            reach_indices.ctypes.data,
            out_indptr.ctypes.data, out_indices.ctypes.data,
            out_data.ctypes.data, byref(out_nnz))

        nnz = out_nnz.value
        result_csc = csc_array(
            (out_data[:nnz], out_indices[:nnz], out_indptr), shape=(n, n))
        return csr_array(result_csc)

    # No precomputed reach — two-call pattern
    out_indptr = np.empty(n + 1, dtype=np.int32)
    out_nnz = c_int(0)
    _dll.openmc_cram_expm(
        n, indptr, indices, data, dt, order, drop_tol,
        perm_ptr, None, None,
        out_indptr.ctypes.data, None, None, byref(out_nnz))

    nnz = out_nnz.value

    out_indices = np.empty(nnz, dtype=np.int32)
    out_data = np.empty(nnz, dtype=np.float64)
    _dll.openmc_cram_expm(
        n, indptr, indices, data, dt, order, drop_tol,
        perm_ptr, None, None,
        out_indptr.ctypes.data, out_indices.ctypes.data,
        out_data.ctypes.data, byref(out_nnz))

    # Build CSC matrix, convert to CSR for optimal SpMV
    result_csc = csc_array((out_data, out_indices, out_indptr), shape=(n, n))
    return csr_array(result_csc)


# --- Chain loading ---

_dll.openmc_load_depletion_chain.restype = c_int
_dll.openmc_load_depletion_chain.errcheck = _error_handler
_dll.openmc_load_depletion_chain.argtypes = [c_char_p]


def load_depletion_chain(filename):
    """Load a depletion chain XML file into the C++ global chain object.

    Parameters
    ----------
    filename : str or Path
        Path to the depletion chain XML file.

    """
    _dll.openmc_load_depletion_chain(str(filename).encode())


# --- Source rate type and normalization mode constants ---

# Must match C++ enum values
SOURCE_RATE_TYPE_POWER = 0
SOURCE_RATE_TYPE_POWER_DENSITY = 1
SOURCE_RATE_TYPE_SOURCE = 2

NORM_MODE_FISSION_Q = 0
NORM_MODE_ENERGY_DEPOSITION = 1


# --- Depletion kernel: configure, execute, free ---

_dll.openmc_depletion_set_config.restype = c_int
_dll.openmc_depletion_set_config.errcheck = _error_handler
_dll.openmc_depletion_set_config.argtypes = [
    c_int,          # n_materials
    _array_1d_int,  # material_indices
    _array_1d_dbl,  # volumes
    _array_1d_int,  # transportable
    c_int,          # rate_tally_idx
    c_int,          # heating_tally_idx
    c_int,          # n_reactions
    c_int,          # fission_rx_idx
    _array_1d_dbl,  # fission_q
    c_int,          # norm_mode
    c_int,          # source_rate_type
    c_int,          # solver_order
]


def depletion_set_config(material_indices, volumes, transportable,
                         rate_tally_idx, heating_tally_idx,
                         n_reactions, fission_rx_idx, fission_q,
                         norm_mode, source_rate_type, solver_order):
    """Configure the C++ depletion kernel's persistent state.

    Must be called after ``openmc.lib.init()`` and
    ``load_depletion_chain()``.

    Parameters
    ----------
    material_indices : numpy.ndarray of int32
        C-API indices of burnable materials.
    volumes : numpy.ndarray
        Volumes [cm^3] per burnable material.
    transportable : numpy.ndarray of int32
        Per-chain-nuclide mask (0/1) for cross-section availability.
    rate_tally_idx : int
        Index of the reaction-rate tally in ``model::tallies``.
    heating_tally_idx : int
        Index of heating tally, or -1 if not used.
    n_reactions : int
        Number of reaction scores in the rate tally.
    fission_rx_idx : int
        Index of 'fission' in reaction list, or -1.
    fission_q : numpy.ndarray
        Fission Q-value per chain nuclide [eV].
    norm_mode : int
        0 = fission_q, 1 = energy_deposition.
    source_rate_type : int
        0 = power, 1 = power_density, 2 = source.
    solver_order : int
        CRAM order (16 or 48).

    """
    n_materials = len(material_indices)
    mat_idx = np.ascontiguousarray(material_indices, dtype=np.int32)
    vols = np.ascontiguousarray(volumes, dtype=np.float64)
    trans = np.ascontiguousarray(transportable, dtype=np.int32)
    fq = np.ascontiguousarray(fission_q, dtype=np.float64)

    _dll.openmc_depletion_set_config(
        n_materials, mat_idx, vols, trans,
        rate_tally_idx, heating_tally_idx,
        n_reactions, fission_rx_idx, fq,
        norm_mode, source_rate_type, solver_order)


_dll.openmc_depletion_execute_step.restype = c_int
_dll.openmc_depletion_execute_step.errcheck = _error_handler
_dll.openmc_depletion_execute_step.argtypes = [
    c_char_p,       # scheme_name
    _array_1d_dbl,  # n_bos_flat
    c_double,       # dt
    c_double,       # source_rate
    c_double,       # prev_dt
    c_int,          # run_transport
    _array_1d_dbl,  # out_eos_flat
    POINTER(c_double),  # out_k_eff
]


def depletion_execute_step(scheme_name, n_bos_flat, dt, source_rate,
                           prev_dt, run_transport):
    """Execute one macro-timestep of a depletion scheme via C++.

    Previous-step BOS matrices are managed internally by the C++ kernel.

    Parameters
    ----------
    scheme_name : str
        Integration scheme name (e.g. 'cecm', 'leqi').
    n_bos_flat : numpy.ndarray
        BOS atom counts, flat array of shape ``(n_materials * n_chain,)``.
    dt : float
        Timestep in seconds.
    source_rate : float
        Power [W] or source rate [n/s].
    prev_dt : float
        Previous timestep in seconds (0.0 for first step).
    run_transport : bool
        Whether to run transport or reuse cached results.

    Returns
    -------
    eos_flat : numpy.ndarray
        EOS atom counts, same shape as *n_bos_flat*.
    k_eff : float
        Effective multiplication factor from last transport.

    """
    bos = np.ascontiguousarray(n_bos_flat, dtype=np.float64)
    eos = np.empty_like(bos)
    k_eff = c_double(0.0)

    _dll.openmc_depletion_execute_step(
        scheme_name.encode(), bos, dt, source_rate,
        prev_dt, int(run_transport), eos, byref(k_eff))

    return eos, k_eff.value


_dll.openmc_depletion_free.restype = c_int
_dll.openmc_depletion_free.errcheck = _error_handler
_dll.openmc_depletion_free.argtypes = []


def depletion_free():
    """Free the C++ depletion kernel state."""
    _dll.openmc_depletion_free()
