"""Depletion driver that interprets integration scheme graphs.

This module provides :class:`DepletionDriver`, a single entry point for
running depletion simulations.  It replaces the ``Integrator`` +
``Operator`` pattern with a cleaner separation: the declarative
integration schemes (in :mod:`openmc.deplete.integration_schemes`) define
*what* to compute, while the driver handles *how* to execute.

Heavy lifting (matrix assembly and CRAM solves) is delegated to C++ via
:mod:`openmc.lib.deplete`.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_array

from openmc.data import DataLibrary

import openmc
import openmc.checkvalue as cv
import openmc.lib
from openmc.lib.deplete import (
    load_depletion_chain,
    chain_form_rxn_matrix,
    cram_solve_batch,
)
from .chain import Chain
from .integration_schemes import (
    BOS, PREV_STEP, PREV_ITER,
    Transport, Expm, AverageMatrix, Iterate,
    IntegrationScheme, SCHEMES,
)
from .stepresult import StepResult
from .reaction_rates import ReactionRates

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = ['DepletionDriver']

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


class DepletionDriver:
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
    scheme : str or IntegrationScheme, optional
        Integration scheme name (key in ``SCHEMES``) or a custom
        :class:`IntegrationScheme` instance.  Default ``'cecm'``.
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
        scheme: str | IntegrationScheme = 'cecm',
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

        self._model = model
        self._chain_file = Path(chain_file)
        self._source_rate_type = source_rate_type
        self._normalization_mode = normalization_mode
        self._solver_order = solver_order
        self._output_path = Path(output_path)

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

        # Resolve scheme
        if isinstance(scheme, str):
            cv.check_value('scheme', scheme, tuple(SCHEMES))
            self._scheme = SCHEMES[scheme]
        else:
            cv.check_type('scheme', scheme, IntegrationScheme)
            self._scheme = scheme

        # Python chain (for metadata, decay matrix, nuclide info)
        self._chain = Chain.from_xml(self._chain_file)

        # Load the chain into C++ once — must be called again in
        # _initialize() after openmc.lib.init() since finalize() clears it.
        # We still load here to validate the file path.
        load_depletion_chain(str(self._chain_file))

        # Precompute the decay matrix (constant for all steps)
        self._A_decay = self._chain.decay_matrix

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
        self._cached_fission_energy = None

        self._initialized = True

    # ------------------------------------------------------------------
    # Transport dispatch
    # ------------------------------------------------------------------

    def _run_transport_and_extract(self, densities_per_mat):
        """Run OpenMC transport and extract reaction-rate matrices.

        Parameters
        ----------
        densities_per_mat : list of numpy.ndarray
            Atom counts (not densities) for each burnable material,
            indexed by chain nuclide.

        Returns
        -------
        rxn_matrices : list of csc_array
            Reaction-rate matrices A_rxn for each material.
        k_eff : float
            Effective multiplication factor.
        fission_energy : float
            Total fission energy in eV per source particle across all
            materials.

        """
        # Update material compositions (only nuclides with cross-section data)
        # and track which nuclides have nonzero density across all materials.
        nonzero_nucs = set()
        for i, mat_id in enumerate(self._burn_mat_ids):
            vol = self._volumes[mat_id]
            n_atoms = densities_per_mat[i]  # shape (n_chain,)
            nuclides = []
            atom_densities = []
            for j, name in enumerate(self._nuclide_names):
                if name not in self._transportable:
                    continue
                dens = n_atoms[j] / vol * 1e-24  # atom/b-cm
                if dens > 0.0:
                    nuclides.append(name)
                    atom_densities.append(dens)
                    nonzero_nucs.add(name)
            if nuclides:
                lib_mat = openmc.lib.materials[int(mat_id)]
                lib_mat.set_densities(nuclides, atom_densities)

        # Update tally nuclide list to only include nuclides with nonzero
        # density (avoids scoring 1000+ zero-density nuclides).
        nuc_names = [n for n in self._nuclide_names if n in nonzero_nucs]
        nuc_chain_idx = np.array(
            [self._chain.nuclide_dict[n] for n in nuc_names], dtype=np.int32)
        self._tally_nuclides = nuc_names
        self._tally_nuc_chain_idx = nuc_chain_idx
        self._rate_tally.nuclides = nuc_names

        # Run transport
        openmc.lib.reset()
        openmc.lib.run()

        # Extract k_eff from last statepoint
        k_eff = openmc.lib.keff()[0]

        # Extract tally results
        tally_means = self._rate_tally.mean
        n_mats = len(self._burn_mat_ids)
        n_nucs = len(self._tally_nuclides)
        n_rxns = len(self._reactions)
        # Reshape: (n_mats, n_nucs, n_rxns)
        tally_means = tally_means.reshape(n_mats, n_nucs, n_rxns)

        # Energy deposition tally
        if self._heating_tally is not None:
            heating_means = self._heating_tally.mean.reshape(n_mats)
        else:
            heating_means = None

        # For each material: form A_rxn, accumulate fission energy
        rxn_matrices = []
        total_fission_energy = 0.0

        for i, mat_id in enumerate(self._burn_mat_ids):
            vol = self._volumes[mat_id]
            n_atoms = densities_per_mat[i]

            # rates shape: (n_nucs, n_rxns)  in [(reactions/src)*b-cm/atom]
            rates = tally_means[i]

            # Divide by volume in b-cm to get [(reactions/src)/atom]
            vol_b_cm = vol * 1e24
            rates_per_atom = rates / vol_b_cm

            # Accumulate fission energy
            fission_idx = (self._reactions.index('fission')
                           if 'fission' in self._reactions else None)
            if self._normalization_mode == 'fission-q' and fission_idx is not None:
                for j, name in enumerate(self._tally_nuclides):
                    chain_idx = self._chain.nuclide_dict[name]
                    atom_per_bcm = n_atoms[chain_idx] / vol_b_cm
                    fission_rate = rates[j, fission_idx] * atom_per_bcm
                    total_fission_energy += fission_rate * self._fission_q[chain_idx]
            elif self._normalization_mode == 'energy-deposition':
                total_fission_energy += heating_means[i]

            # Form A_rxn via C++
            A_rxn = chain_form_rxn_matrix(
                rates_per_atom, self._tally_nuc_chain_idx, n_rxns)
            rxn_matrices.append(A_rxn)

        return rxn_matrices, k_eff, total_fission_energy

    def _handle_transport(self, node, densities, matrices, source_rate):
        """Process a Transport node.

        Parameters
        ----------
        node : Transport
        densities : dict
            Mapping from density references to list of ndarray.
        matrices : dict
            Mapping from matrix references to list of csc_array.
        source_rate : float
            Power [W] or source rate [n/s].

        Returns
        -------
        k_eff : float

        """
        # Resolve which density vector to use
        dens_per_mat = self._resolve_density(node.density, densities)

        # Run transport or use cache
        if self._should_run_transport:
            rxn_matrices, k_eff, fission_energy = \
                self._run_transport_and_extract(dens_per_mat)
            self._cached_rxn_matrices = rxn_matrices
            self._cached_k_eff = k_eff
            self._cached_fission_energy = fission_energy
        else:
            rxn_matrices = self._cached_rxn_matrices
            k_eff = self._cached_k_eff
            fission_energy = self._cached_fission_energy

        # Compute normalization factor s
        if self._source_rate_type == 'source':
            s = source_rate
        else:
            # source_rate is power in Watts
            if fission_energy == 0.0:
                s = 0.0
            else:
                s = source_rate / (fission_energy * EV_PER_JOULE)

        # Combine: A = A_decay + s * A_rxn
        combined = []
        for A_rxn in rxn_matrices:
            combined.append(self._A_decay + s * A_rxn)
        matrices[node] = combined

        return k_eff

    def _handle_expm(self, node, densities, matrices, dt, prev_dt,
                     avg_matrices=None):
        """Process an Expm node (matrix exponential / CRAM solve).

        Parameters
        ----------
        node : Expm
        densities : dict
        matrices : dict
        dt : float
            Timestep in seconds.
        prev_dt : float or None
            Previous timestep in seconds (for LE/QI weights).
        avg_matrices : dict or None
            Running average matrices for SI iterate.

        """
        # Build combined matrix per material
        n_mats = len(self._burn_mat_ids)
        combined = [csc_array((self._n_chain, self._n_chain))
                    for _ in range(n_mats)]

        for term in node.terms:
            # Resolve weight
            if callable(term.weight):
                w = term.weight(prev_dt, dt)
            else:
                w = term.weight

            # Resolve matrix reference
            if isinstance(term.matrix, AverageMatrix):
                mats = avg_matrices[term.matrix.source]
            elif term.matrix is PREV_STEP:
                mats = matrices[PREV_STEP]
            else:
                # Transport node
                mats = matrices[term.matrix]

            for k in range(n_mats):
                combined[k] = combined[k] + w * mats[k]

        # Resolve input density
        input_dens = self._resolve_density(node.density, densities)

        # CRAM solve
        results = cram_solve_batch(combined, input_dens, dt,
                                   order=self._solver_order)
        densities[node] = results

    def _handle_iterate(self, node, densities, matrices, dt, prev_dt,
                        source_rate, last_expm_before):
        """Process an Iterate node (stochastic implicit iterations).

        Parameters
        ----------
        node : Iterate
        densities : dict
        matrices : dict
        dt : float
        prev_dt : float or None
        source_rate : float
        last_expm_before : Expm or None
            Last Expm node computed before entering this Iterate, used
            to seed PREV_ITER.

        Returns
        -------
        k_eff : float
            k-effective from the last transport in the iterate body.

        """
        # Seed PREV_ITER from last computed Expm
        if last_expm_before is not None:
            densities[PREV_ITER] = densities[last_expm_before]
        else:
            densities[PREV_ITER] = densities[BOS]

        # Running averages for AverageMatrix nodes
        avg_matrices = {}

        k_eff = self._cached_k_eff or 1.0

        for j in range(1, node.n_iterations + 1):
            for op in node.body:
                if isinstance(op, Transport):
                    k_eff = self._handle_transport(
                        op, densities, matrices, source_rate)

                    # Update running average
                    if j == 1:
                        avg_matrices[op] = [
                            m.copy() for m in matrices[op]]
                    else:
                        alpha = 1.0 / j
                        for k in range(len(self._burn_mat_ids)):
                            avg_matrices[op][k] = (
                                alpha * matrices[op][k]
                                + (1 - alpha) * avg_matrices[op][k])

                elif isinstance(op, Expm):
                    self._handle_expm(
                        op, densities, matrices, dt, prev_dt,
                        avg_matrices=avg_matrices)

            # Update PREV_ITER to last Expm in body
            last_body_expm = self._find_last_expm(node.body)
            if last_body_expm is not None:
                densities[PREV_ITER] = densities[last_body_expm]

        return k_eff

    # ------------------------------------------------------------------
    # Scheme interpreter
    # ------------------------------------------------------------------

    def _execute_step(self, scheme, n_bos_list, dt, source_rate,
                      prev_bos_matrices, prev_dt):
        """Execute one macro-timestep of a depletion scheme.

        Parameters
        ----------
        scheme : IntegrationScheme
        n_bos_list : list of numpy.ndarray
            BOS atom counts per material (each shape ``(n_chain,)``).
        dt : float
            Timestep in seconds.
        source_rate : float
        prev_bos_matrices : list of csc_array or None
            BOS matrices from previous step (for PREV_STEP reference).
        prev_dt : float or None

        Returns
        -------
        n_end_list : list of numpy.ndarray
            EOS atom counts per material.
        next_prev_bos_matrices : list of csc_array
            BOS matrices for the *next* step's PREV_STEP reference.
        k_eff : float

        """
        densities = {BOS: n_bos_list}
        matrices = {}
        if prev_bos_matrices is not None:
            matrices[PREV_STEP] = prev_bos_matrices

        bos_transport = None
        k_eff = self._cached_k_eff or 1.0
        last_expm = None  # tracks last Expm before an Iterate

        for op in scheme.steps:
            if isinstance(op, Transport):
                k_eff = self._handle_transport(
                    op, densities, matrices, source_rate)
                if bos_transport is None:
                    bos_transport = op

            elif isinstance(op, Expm):
                self._handle_expm(
                    op, densities, matrices, dt, prev_dt)
                last_expm = op

            elif isinstance(op, Iterate):
                k_eff = self._handle_iterate(
                    op, densities, matrices, dt, prev_dt, source_rate,
                    last_expm)

        # Find final density (last Expm in the entire scheme)
        final_expm = self._find_last_expm(scheme.steps)
        n_end_list = densities[final_expm]

        # BOS matrices for next step's PREV_STEP
        next_prev_bos_matrices = (
            matrices[bos_transport] if bos_transport is not None else None)

        return n_end_list, next_prev_bos_matrices, k_eff

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_density(ref, densities):
        """Look up a density reference in the densities dict."""
        if ref in densities:
            return densities[ref]
        raise KeyError(f"Density reference {ref!r} not found. Available: "
                       f"{list(densities.keys())}")

    @staticmethod
    def _find_last_expm(steps):
        """Find the last Expm node in a step list, including Iterate bodies."""
        last = None
        for op in steps:
            if isinstance(op, Expm):
                last = op
            elif isinstance(op, Iterate):
                for inner in op.body:
                    if isinstance(inner, Expm):
                        last = inner
        return last

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

        if not self._initialized:
            self._initialize()

        try:
            n = self._get_initial_compositions()
            prev_bos_matrices = None
            prev_dt = None
            t = 0.0

            for i, (dt, source_rate) in enumerate(
                    zip(self._timesteps_s, self._source_rates)):
                self._should_run_transport = self._transport_mask[i]

                scheme = self._scheme
                if i == 0 and scheme.fallback is not None:
                    scheme = scheme.fallback

                t0 = time.time()
                n_end, prev_bos_matrices, k_eff = self._execute_step(
                    scheme, n, dt, source_rate, prev_bos_matrices, prev_dt)
                proc_time = time.time() - t0

                self._save_step(i, n, n_end, t, dt, source_rate,
                                k_eff, proc_time)

                n = n_end
                prev_dt = dt
                t += dt

        finally:
            openmc.lib.finalize()
            self._initialized = False

        return Results(str(self._output_path))
