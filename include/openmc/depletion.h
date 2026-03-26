//! \file depletion.h
//! \brief High-level depletion functions that combine transport results with
//!        chain data to produce transmutation matrices and CRAM solves.

#ifndef OPENMC_DEPLETION_H
#define OPENMC_DEPLETION_H

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

} // namespace openmc

#endif // OPENMC_DEPLETION_H
