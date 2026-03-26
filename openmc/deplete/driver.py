"""Depletion manager that interprets integration scheme graphs.

This module provides :class:`DepletionManager`, a single entry point for
running depletion simulations.  It replaces the ``Integrator`` +
``Operator`` pattern with a cleaner separation: the declarative
integration schemes (in :mod:`openmc.deplete.integration_schemes`) define
*what* to compute, while the manager handles *how* to execute.

Heavy lifting (matrix assembly, scheme interpretation, and CRAM solves) is
delegated to C++ via :mod:`openmc.lib.deplete`.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openmc.data import DataLibrary

import openmc
import openmc.checkvalue as cv
import openmc.lib
from openmc.lib.deplete import (
    load_depletion_chain,
    depletion_set_config,
    depletion_execute_step,
    depletion_free,
)
from .chain import Chain
from .integration_schemes import SCHEMES, IntegrationScheme
from .stepresult import StepResult
from .reaction_rates import ReactionRates

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ['DepletionManager']

# eV per Joule
EV_PER_JOULE = 1.602176634e-19

# Unit multipliers to convert timesteps to seconds
_TIMESTEP_UNITS = {
    's': 1.0,
    'min': 60.0,
    'h': 3600.0,
    'd': 86400.0,
    'a': 86400.0 * 365.25,
    'MWd/kg': None,  # handled separately (burnup)
}


_nuclides_with_data_cache = None


def _get_nuclides_with_data():
    """Return set of nuclide names that have neutron cross-section data.

    Results are cached at module level since the cross-section library
    does not change during a session.
    """
    global _nuclides_with_data_cache
    if _nuclides_with_data_cache is not None:
        return _nuclides_with_data_cache
    cross_sections = openmc.config.get("cross_sections")
    if cross_sections is None:
        return set()
    nuclides = set()
    for lib in DataLibrary.from_xml(cross_sections).libraries:
        if lib['type'] == 'neutron':
            nuclides.update(lib['materials'])
    _nuclides_with_data_cache = nuclides
    return nuclides


class DepletionManager:
    """Run a depletion simulation by interpreting an integration scheme.

    Parameters
    ----------
    model : openmc.Model
        Model describing the geometry, materials, and settings.
    chain_file : str or Path
        Path to a depletion chain XML file.
    timesteps : iterable of float
        Sequence of timestep durations.
    source_rates : float or iterable of float
        Power [W], power density [W/gHM], or source rate [n/s] for each
        timestep.  A scalar is broadcast to all timesteps.
    source_rate_type : {'power', 'power_density', 'source'}, optional
        How *source_rates* should be interpreted.  Default ``'power'``.
    timestep_units : str, optional
        Units for *timesteps*: ``'s'``, ``'min'``, ``'h'``, ``'d'``,
        ``'a'`` (Julian year), or ``'MWd/kg'`` (burnup).
        Default ``'s'``.
    scheme : str, optional
        Integration scheme name (key in ``SCHEMES``).  Default
        ``'cecm'``.
    normalization_mode : {'fission-q', 'energy-deposition'}, optional
        How power is converted to a source rate.  Ignored when
        *source_rate_type* is ``'source'``.  Default ``'fission-q'``.
    solver_order : {16, 48}, optional
        CRAM approximation order.  Default ``48``.
    output_path : str or Path, optional
        HDF5 file path for depletion results.  Default
        ``'depletion_results.h5'``.
    transport_schedule : str or iterable of bool, optional
        Controls which timesteps run transport.  ``'every'`` (default)
        runs transport every step; ``'first'`` runs only on step 0;
        an iterable of bool gives per-step control.

    """

    def __init__(
        self,
        model: openmc.Model,
        chain_file: str | Path,
        timesteps: Iterable[float],
        source_rates: float | Iterable[float],
        *,
        source_rate_type: str = 'power',
        timestep_units: str = 's',
        scheme: str = 'cecm',
        normalization_mode: str = 'fission-q',
        solver_order: int = 48,
        output_path: str | Path = 'depletion_results.h5',
        transport_schedule: str | Iterable[bool] = 'every',
    ):
        # --- Validate inputs ---
        cv.check_type('model', model, openmc.Model)
        cv.check_type('chain_file', chain_file, (str, Path))
        cv.check_value('source_rate_type', source_rate_type,
                        ('power', 'power_density', 'source'))
        cv.check_value('timestep_units', timestep_units,
                        tuple(_TIMESTEP_UNITS))
        cv.check_value('normalization_mode', normalization_mode,
                        ('fission-q', 'energy-deposition'))
        cv.check_value('solver_order', solver_order, (16, 48))
        cv.check_type('scheme', scheme, str)
        cv.check_value('scheme', scheme, tuple(SCHEMES))

        self._model = model
        self._chain_file = Path(chain_file)
        self._source_rate_type = source_rate_type
        self._normalization_mode = normalization_mode
        self._solver_order = solver_order
        self._output_path = Path(output_path)
        self._scheme = SCHEMES[scheme]
        self._scheme_name = scheme

        # Fallback scheme name for C++ kernel (LE/QI → lower-order on step 0)
        _FALLBACK_NAMES = {'leqi': 'celi', 'si_leqi': 'si_celi'}
        self._scheme_fallback_name = _FALLBACK_NAMES.get(self._scheme_name)

        # Timesteps → seconds
        timesteps = np.asarray(list(timesteps), dtype=np.float64)
        if timestep_units == 'MWd/kg':
            # Burnup: deferred until we know heavy metal mass
            self._timesteps_burnup = timesteps.copy()
            self._timesteps_s = None  # resolved in _initialize
        else:
            self._timesteps_burnup = None
            self._timesteps_s = timesteps * _TIMESTEP_UNITS[timestep_units]

        n_steps = len(timesteps)

        # Source rates → 1-D array
        if np.ndim(source_rates) == 0:
            source_rates = np.full(n_steps, source_rates, dtype=np.float64)
        else:
            source_rates = np.asarray(list(source_rates), dtype=np.float64)
        if len(source_rates) != n_steps:
            raise ValueError(
                f"Length of source_rates ({len(source_rates)}) does not match "
                f"number of timesteps ({n_steps}).")
        self._source_rates = source_rates

        # Python chain (for metadata, decay matrix, nuclide info)
        self._chain = Chain.from_xml(self._chain_file)

        # Load the chain into C++ once — must be called again in
        # _initialize() after openmc.lib.init() since finalize() clears it.
        # We still load here to validate the file path.
        load_depletion_chain(str(self._chain_file))

        # Build fission Q vector for normalization (indexed by chain nuclide)
        self._fission_q = np.zeros(len(self._chain))
        for i, nuc in enumerate(self._chain.nuclides):
            for rx in nuc.reactions:
                if rx.type == 'fission':
                    self._fission_q[i] = rx.Q  # eV per fission
                    break

        # Transport schedule → per-step bool mask
        if isinstance(transport_schedule, str):
            cv.check_value('transport_schedule', transport_schedule,
                            ('every', 'first'))
            if transport_schedule == 'every':
                self._transport_mask = [True] * n_steps
            else:  # 'first'
                self._transport_mask = (
                    [True] + [False] * (n_steps - 1))
        else:
            self._transport_mask = [bool(x) for x in transport_schedule]
            if len(self._transport_mask) != n_steps:
                raise ValueError(
                    f"Length of transport_schedule "
                    f"({len(self._transport_mask)}) does not match "
                    f"number of timesteps ({n_steps}).")

        # These are populated in _initialize()
        self._burn_mat_ids = None  # list of str (sorted by int(id))
        self._burn_mat_names = None  # list of str
        self._volumes = None  # dict str→float
        self._n_chain = len(self._chain)
        self._nuclide_names = list(self._chain.nuclide_dict)
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization (deferred until run())
    # ------------------------------------------------------------------

    def _initialize(self):
        """Set up C API, tallies, and material metadata."""

        # Determine which nuclides have cross-section data so we can
        # strip decay-only nuclides from materials before transport.
        nucs_with_data = _get_nuclides_with_data()
        chain_nucs = set(self._nuclide_names)
        decay_only = chain_nucs - nucs_with_data

        self._model.export_to_xml(
            nuclides_to_ignore=decay_only)

        # Initialize the C library
        openmc.lib.init()

        # Re-load the depletion chain into C++ (openmc.lib.finalize()
        # clears the global chain, so we must reload after each init).
        load_depletion_chain(str(self._chain_file))

        # Enable pre-computation of depletion reaction cross sections
        # during transport.  Without this flag, tally scoring falls back
        # to expensive on-the-fly cross-section lookups for every
        # nuclide × reaction bin.
        openmc.lib.settings.need_depletion_rx = True

        # Identify depletable materials
        self._burn_mat_ids = []
        self._burn_mat_names = []
        self._volumes = {}
        heavy_metal_mass = 0.0

        for mat in self._model.materials:
            if mat.depletable:
                mat_id_str = str(mat.id)
                self._burn_mat_ids.append(mat_id_str)
                self._burn_mat_names.append(
                    mat.name if mat.name else '')
                if mat.volume is None:
                    raise RuntimeError(
                        f"Volume not specified for depletable material "
                        f"with ID={mat.id}.")
                cv.check_greater_than('volume', mat.volume, 0.0,
                                      equality=False)
                self._volumes[mat_id_str] = mat.volume
                heavy_metal_mass += mat.fissionable_mass

        # Sort by integer ID
        order = sorted(range(len(self._burn_mat_ids)),
                       key=lambda i: int(self._burn_mat_ids[i]))
        self._burn_mat_ids = [self._burn_mat_ids[i] for i in order]
        self._burn_mat_names = [self._burn_mat_names[i] for i in order]

        if not self._burn_mat_ids:
            raise RuntimeError(
                "No depletable materials found in the model.")

        # Resolve MWd/kg timesteps → seconds
        if self._timesteps_burnup is not None:
            if heavy_metal_mass == 0.0:
                raise RuntimeError(
                    "Cannot use MWd/kg with no fissionable mass.")
            # MWd/kg * kg * 1e6 W/MW * 86400 s/d = J
            # time = energy / power;  but source_rates IS the power
            # dt_s = burnup * HM_mass * 1e6 * 86400 / power
            self._timesteps_s = np.empty_like(self._timesteps_burnup)
            for i, bu in enumerate(self._timesteps_burnup):
                self._timesteps_s[i] = (
                    bu * heavy_metal_mass * 1e6 * 86400.0
                    / self._source_rates[i])

        # Power density → absolute power
        if self._source_rate_type == 'power_density':
            self._source_rates = self._source_rates * heavy_metal_mass * 1e3

        # Build reaction score list from chain
        self._reactions = list(self._chain.reactions)

        # Record which chain nuclides have cross-section data (used when
        # updating material compositions for transport)
        self._transportable = nucs_with_data & set(self._nuclide_names)

        # Set up the reaction-rate tally.  Include ALL transportable chain
        # nuclides (not just those currently in materials) so that fission
        # products produced during depletion are tallied once they appear
        # with nonzero density.
        mat_filter = openmc.lib.MaterialFilter(
            [openmc.lib.materials[int(m)] for m in self._burn_mat_ids])
        nuc_names = []
        nuc_chain_indices = []
        for name in self._nuclide_names:
            if name in self._transportable:
                nuc_names.append(name)
                nuc_chain_indices.append(self._chain.nuclide_dict[name])

        self._tally_nuclides = nuc_names
        self._tally_nuc_chain_idx = np.array(nuc_chain_indices, dtype=np.int32)

        self._rate_tally = openmc.lib.Tally()
        self._rate_tally.filters = [mat_filter]
        self._rate_tally.nuclides = nuc_names
        self._rate_tally.scores = self._reactions
        self._rate_tally.writable = False
        self._rate_tally.multiply_density = False

        # If energy-deposition, set up heating tally
        if self._normalization_mode == 'energy-deposition':
            self._heating_tally = openmc.lib.Tally()
            self._heating_tally.filters = [mat_filter]
            self._heating_tally.scores = ['heating-local']
            self._heating_tally.writable = False
        else:
            self._heating_tally = None

        # Cached transport results
        self._cached_rxn_matrices = None
        self._cached_k_eff = None

        self._initialized = True
        self._configure_cpp_kernel()

    def _configure_cpp_kernel(self):
        """Set up the C++ depletion kernel with configuration from Python."""
        from openmc.lib.deplete import (
            SOURCE_RATE_TYPE_POWER,
            SOURCE_RATE_TYPE_POWER_DENSITY,
            SOURCE_RATE_TYPE_SOURCE,
            NORM_MODE_FISSION_Q,
            NORM_MODE_ENERGY_DEPOSITION,
        )

        # Material C-API indices
        mat_indices = np.array([
            openmc.lib.materials[int(m)]._index
            for m in self._burn_mat_ids
        ], dtype=np.int32)

        # Volumes array
        volumes = np.array([
            self._volumes[m] for m in self._burn_mat_ids
        ], dtype=np.float64)

        # Transportable mask: 1 if chain nuclide has cross-section data
        transportable = np.array([
            1 if name in self._transportable else 0
            for name in self._nuclide_names
        ], dtype=np.int32)

        # Tally indices
        rate_tally_idx = self._rate_tally._index
        heating_tally_idx = (
            self._heating_tally._index
            if self._heating_tally is not None else -1)

        # Reaction info
        n_reactions = len(self._reactions)
        fission_rx_idx = (
            self._reactions.index('fission')
            if 'fission' in self._reactions else -1)

        # Normalization mode
        norm_mode = (NORM_MODE_ENERGY_DEPOSITION
                     if self._normalization_mode == 'energy-deposition'
                     else NORM_MODE_FISSION_Q)

        # Source rate type
        srt_map = {
            'power': SOURCE_RATE_TYPE_POWER,
            'power_density': SOURCE_RATE_TYPE_POWER_DENSITY,
            'source': SOURCE_RATE_TYPE_SOURCE,
        }
        source_rate_type = srt_map[self._source_rate_type]

        depletion_set_config(
            mat_indices, volumes, transportable,
            rate_tally_idx, heating_tally_idx,
            n_reactions, fission_rx_idx,
            self._fission_q,
            norm_mode, source_rate_type,
            self._solver_order)

    def _execute_step(self, scheme_name, n_bos_list, dt,
                      source_rate, prev_dt):
        """Execute one macro-timestep using the C++ kernel.

        Parameters
        ----------
        scheme_name : str
            Integration scheme name for C++ lookup.
        n_bos_list : list of numpy.ndarray
            BOS atom counts per material.
        dt : float
            Timestep in seconds.
        source_rate : float
            Power [W] or source rate [n/s].
        prev_dt : float
            Previous timestep in seconds (0.0 for first step).

        Returns
        -------
        n_end_list : list of numpy.ndarray
            EOS atom counts per material.
        k_eff : float

        """
        n_bos_flat = np.concatenate(n_bos_list)
        eos_flat, k_eff = depletion_execute_step(
            scheme_name, n_bos_flat, dt, source_rate,
            prev_dt, self._should_run_transport)

        # Unpack EOS into per-material arrays
        n_end_list = [
            eos_flat[i * self._n_chain:(i + 1) * self._n_chain]
            for i in range(len(self._burn_mat_ids))
        ]

        # Sync tally nuclide info from C library for _save_step
        if self._should_run_transport:
            self._tally_nuclides = self._rate_tally.nuclides
            self._tally_nuc_chain_idx = np.array([
                self._chain.nuclide_dict[n]
                for n in self._tally_nuclides
            ], dtype=np.int32)
            self._cached_rxn_matrices = True  # signal tally data available
            self._cached_k_eff = k_eff

        return n_end_list, k_eff

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_initial_compositions(self):
        """Extract initial atom counts from the model's materials.

        Uses the *Python* model materials (which include decay-only
        nuclides) rather than the C library materials (which exclude
        nuclides without cross-section data).

        Returns
        -------
        list of numpy.ndarray
            Atom counts per burnable material, each shape ``(n_chain,)``.

        """
        nuclide_dict = self._chain.nuclide_dict
        n_chain = self._n_chain
        # Build lookup from material ID to Python Material
        py_mats = {str(m.id): m for m in self._model.materials}
        result = []
        for mat_id in self._burn_mat_ids:
            py_mat = py_mats[mat_id]
            vol = self._volumes[mat_id]
            n = np.zeros(n_chain)
            atom_densities = py_mat.get_nuclide_atom_densities()
            for name, dens in atom_densities.items():
                if name in nuclide_dict:
                    # dens in atom/b-cm; convert to atoms
                    n[nuclide_dict[name]] = dens * vol * 1e24
            result.append(n)
        return result

    def _save_step(self, step_idx, n_bos_list, n_eos_list, t, dt,
                   source_rate, k_eff, proc_time):
        """Save a single depletion step to HDF5."""
        res = StepResult()
        res.k = (k_eff, 0.0)  # no uncertainty from single batch
        res.time = [t, t + dt]
        res.source_rate = source_rate
        res.proc_time = proc_time

        # Allocate storage
        burn_list = list(self._burn_mat_ids)
        nuc_list = list(self._nuclide_names)
        res.allocate(
            self._volumes,
            nuc_list,
            burn_list,
            burn_list,
            name_list=self._burn_mat_names,
        )

        # Fill EOS atom counts
        for i, mat_id in enumerate(self._burn_mat_ids):
            for j, nuc in enumerate(nuc_list):
                res.data[i, j] = n_eos_list[i][j]

        # Populate reaction rates from the last transport tally.
        # Rates are stored as (reactions/src * b-cm / atom) per the
        # convention used by the existing integrator framework.
        rates = ReactionRates(
            burn_list,
            nuc_list,
            self._reactions,
        )
        if self._cached_rxn_matrices is not None:
            tally_means = self._rate_tally.mean
            n_mats = len(self._burn_mat_ids)
            n_tallied = len(self._tally_nuclides)
            n_rxns = len(self._reactions)
            if n_tallied > 0:
                tally_means = tally_means.reshape(n_mats, n_tallied, n_rxns)
                for i, mat_id in enumerate(burn_list):
                    mat_idx = rates.index_mat[mat_id]
                    for j, name in enumerate(self._tally_nuclides):
                        if name not in rates.index_nuc:
                            continue
                        nuc_idx = rates.index_nuc[name]
                        for k, rx in enumerate(self._reactions):
                            rx_idx = rates.index_rx[rx]
                            rates[mat_idx, nuc_idx, rx_idx] = \
                                tally_means[i, j, k]
        res.rates = rates

        res.export_to_hdf5(str(self._output_path), step_idx,
                           write_rates=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Execute the depletion simulation.

        Returns
        -------
        openmc.deplete.Results
            All step results.

        """
        from .results import Results

        try:
            if not self._initialized:
                self._initialize()

            n = self._get_initial_compositions()
            prev_dt = None
            t = 0.0

            for i, (dt, source_rate) in enumerate(
                    zip(self._timesteps_s, self._source_rates)):
                self._should_run_transport = self._transport_mask[i]

                t0 = time.time()

                scheme_name = self._scheme_name
                if i == 0 and self._scheme_fallback_name is not None:
                    scheme_name = self._scheme_fallback_name
                n_end, k_eff = self._execute_step(
                    scheme_name, n, dt, source_rate,
                    prev_dt if prev_dt is not None else 0.0)

                proc_time = time.time() - t0

                self._save_step(i, n, n_end, t, dt, source_rate,
                                k_eff, proc_time)

                n = n_end
                prev_dt = dt
                t += dt

        finally:
            depletion_free()
            openmc.lib.finalize()
            self._initialized = False

        return Results(str(self._output_path))
