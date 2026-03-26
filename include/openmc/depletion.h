//! \file depletion.h
//! \brief High-level depletion functions that combine transport results with
//!        chain data to produce transmutation matrices and CRAM solves.

#ifndef OPENMC_DEPLETION_H
#define OPENMC_DEPLETION_H

#include <cstdint>

#include "openmc/bateman_solvers.h"
#include "openmc/depletion_scheme.h"
#include "openmc/sparse_matrix.h"
#include "openmc/vector.h"

namespace openmc {

//! How power is converted to a source rate
enum class NormalizationMode { fission_q, energy_deposition };

//! How source_rate should be interpreted
enum class SourceRateType { power, power_density, source };

//! Result of extracting reaction rates from transport tallies and
//! combining them with the decay matrix.
struct DepletionRates {
  vector<CSCMatrix> combined_matrices; //!< A_decay + s * A_rxn per material
  double normalization_factor;         //!< source normalization s [src/s]
  double fission_energy;               //!< total fission energy [eV/src]
};

//! Extract reaction-rate matrices from tally data and build combined
//! depletion matrices (A_decay + s * A_rxn) for each material.
//!
//! \param tally_means  Flat tally output, row-major with shape
//!                     [n_materials, n_tallied_nucs, n_reactions].
//!                     Units: (reactions/src) * b-cm / atom.
//! \param n_materials   Number of burnable materials.
//! \param n_tallied_nucs  Number of nuclides scored in tallies.
//! \param n_reactions   Number of reaction scores.
//! \param nuc_chain_indices  Chain-nuclide index for each tallied nuclide.
//!                           Size n_tallied_nucs.
//! \param atom_counts   Atom counts per material, packed row-major
//!                      [n_materials * n_chain].
//! \param volumes       Volume of each material [cm^3], size n_materials.
//! \param source_rate   Power [W] or source rate [n/s].
//! \param source_rate_type  How source_rate is interpreted.
//! \param norm_mode     How to compute the normalization factor.
//! \param fission_q     Fission Q-value for each chain nuclide [eV/fission].
//!                      Size n_chain.  Only used when norm_mode == fission_q.
//! \param heating_means Per-material heating tally [eV/src], size n_materials.
//!                      Only used when norm_mode == energy_deposition.
//!                      May be nullptr otherwise.
//! \param fission_rx_idx  Index of 'fission' in the reaction score list,
//!                        or -1 if fission is not scored.
//! \return DepletionRates with combined matrices and normalization info.
DepletionRates compute_depletion_rates(
  const double* tally_means,
  int n_materials,
  int n_tallied_nucs,
  int n_reactions,
  const int* nuc_chain_indices,
  const double* atom_counts,
  const double* volumes,
  double source_rate,
  SourceRateType source_rate_type,
  NormalizationMode norm_mode,
  const double* fission_q,
  const double* heating_means,
  int fission_rx_idx);

//! Update depletable material compositions and return the set of nuclides
//! that have nonzero density in at least one material.
//!
//! For each material, converts atom counts to atom/b-cm, filters nuclides
//! to those flagged as transportable and with positive density, then calls
//! Material::set_densities().
//!
//! \param n_materials    Number of burnable materials.
//! \param material_indices  C-API index into model::materials for each
//!                          burnable material. Size n_materials.
//! \param n_chain        Total number of nuclides in the depletion chain.
//! \param atom_counts    Atom counts per material, packed row-major
//!                       [n_materials * n_chain].
//! \param volumes        Volume of each material [cm^3], size n_materials.
//! \param transportable  Bit mask (0/1) per chain nuclide indicating whether
//!                       it has transport cross-section data. Size n_chain.
//! \param[out] nonzero_nuc_indices  Chain indices of nuclides with nonzero
//!             density in at least one material.  Caller allocates [n_chain].
//! \return  Number of nonzero nuclide indices written to nonzero_nuc_indices.
int update_depletable_materials(
  int n_materials,
  const int32_t* material_indices,
  int n_chain,
  const double* atom_counts,
  const double* volumes,
  const int* transportable,
  int* nonzero_nuc_indices);

//==============================================================================
//! Persistent depletion configuration for the macro-timestep kernel.
//==============================================================================

struct DepletionState {
  // --- Material configuration ---
  int n_materials {0};
  vector<int32_t> material_indices; //!< C-API indices into model::materials
  vector<double> volumes;           //!< Material volumes [cm^3]
  vector<int> transportable;        //!< Per-chain-nuclide mask (0/1)

  // --- Tally configuration ---
  int32_t rate_tally_idx {-1};    //!< Index of rate tally in model::tallies
  int32_t heating_tally_idx {-1}; //!< Index of heating tally, or -1

  // --- Chain/reaction configuration ---
  int n_reactions {0};            //!< Number of reaction scores
  int fission_rx_idx {-1};       //!< Index of "fission" in reaction list
  vector<double> fission_q;      //!< Fission Q per chain nuclide [eV]
  NormalizationMode norm_mode {NormalizationMode::fission_q};
  SourceRateType source_rate_type {SourceRateType::power};
  int solver_order {48};          //!< CRAM order (16 or 48)

  // --- Tally nuclide tracking (updated each transport) ---
  vector<int> nuc_chain_indices;  //!< Chain indices of tallied nuclides
  int n_tallied_nucs {0};

  // --- Solver instance ---
  IPFCramSolver cram_solver {IPFCramSolver::Order::cram48};
};

//! Result of executing one macro-timestep.
struct SchemeStepResult {
  //! EOS atom counts per material, each length n_chain
  vector<vector<double>> eos_densities;
  //! BOS combined matrices (for PREV_STEP on next step)
  vector<CSCMatrix> bos_matrices;
  //! k-effective from last transport in this step
  double k_eff {1.0};
};

//! Execute one macro-timestep of a depletion integration scheme.
//!
//! Interprets the scheme DAG, running transport (update materials →
//! openmc_reset → openmc_run → extract rates) for Transport nodes
//! and CRAM solves for Expm nodes.
//!
//! \param scheme         Integration scheme to execute.
//! \param state          Depletion configuration (modified: nuc_chain_indices
//!                       updated per transport).
//! \param n_bos          BOS atom counts per material [n_materials][n_chain].
//! \param dt             Timestep in seconds.
//! \param source_rate    Power [W] or source rate [n/s].
//! \param prev_step_matrices  BOS matrices from previous macro-step for
//!                            LE/QI PREV_STEP reference. nullptr if none.
//! \param prev_dt        Previous timestep in seconds (for LE/QI weights).
//! \param run_transport  If false, reuse cached results for all Transport
//!                       nodes (transport_schedule support).
//! \return SchemeStepResult with EOS densities, BOS matrices, and k_eff.
SchemeStepResult execute_scheme_step(
  const IntegrationScheme& scheme,
  DepletionState& state,
  const vector<vector<double>>& n_bos,
  double dt,
  double source_rate,
  const vector<CSCMatrix>* prev_step_matrices,
  double prev_dt,
  bool run_transport);

} // namespace openmc

#endif // OPENMC_DEPLETION_H
