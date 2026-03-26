//! \file depletion.cpp
//! \brief Implementation of high-level depletion functions

#include "openmc/depletion.h"

#include <algorithm> // for copy
#include <cstring>   // for memcpy

#include "openmc/capi.h"
#include "openmc/chain.h"
#include "openmc/error.h"
#include "openmc/sparse_matrix.h"

namespace openmc {

// eV per Joule
static constexpr double EV_PER_JOULE = 1.602176634e-19;

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
  int fission_rx_idx)
{
  auto& chain = *data::depletion_chain;
  int n_chain = chain.size();
  const auto& A_decay = chain.decay_matrix();

  DepletionRates result;
  result.combined_matrices.reserve(n_materials);
  double total_fission_energy = 0.0;

  // Temporary buffer for rates_per_atom for one material
  vector<double> rates_per_atom(n_tallied_nucs * n_reactions);

  for (int m = 0; m < n_materials; ++m) {
    double vol = volumes[m];
    double vol_b_cm = vol * 1.0e24;

    // Pointer to this material's tally block:
    //   tally_means[m * n_tallied_nucs * n_reactions ... ]
    const double* rates = tally_means + m * n_tallied_nucs * n_reactions;

    // Pointer to this material's atom counts:
    //   atom_counts[m * n_chain ... ]
    const double* n_atoms = atom_counts + m * n_chain;

    // Compute rates_per_atom = rates / vol_b_cm
    for (int k = 0; k < n_tallied_nucs * n_reactions; ++k) {
      rates_per_atom[k] = rates[k] / vol_b_cm;
    }

    // Accumulate fission energy [eV/src]
    if (norm_mode == NormalizationMode::fission_q && fission_rx_idx >= 0) {
      for (int j = 0; j < n_tallied_nucs; ++j) {
        int chain_idx = nuc_chain_indices[j];
        double atom_per_bcm = n_atoms[chain_idx] / vol_b_cm;
        // rates[j * n_reactions + fission_rx_idx] is in
        // (reactions/src)*b-cm/atom
        double fission_rate =
          rates[j * n_reactions + fission_rx_idx] * atom_per_bcm;
        total_fission_energy += fission_rate * fission_q[chain_idx];
      }
    } else if (norm_mode == NormalizationMode::energy_deposition &&
               heating_means != nullptr) {
      total_fission_energy += heating_means[m];
    }

    // Form A_rxn via chain
    CSCMatrix A_rxn = chain.form_rxn_matrix(
      rates_per_atom.data(), n_tallied_nucs, n_reactions, nuc_chain_indices);

    result.combined_matrices.push_back(std::move(A_rxn));
  }

  // Compute normalization factor s
  double s = 0.0;
  if (source_rate_type == SourceRateType::source) {
    s = source_rate;
  } else {
    // source_rate is power in Watts
    if (total_fission_energy != 0.0) {
      s = source_rate / (total_fission_energy * EV_PER_JOULE);
    }
    // If total_fission_energy == 0, s stays 0 (decay-only)
  }

  // Combine: A = A_decay + s * A_rxn for each material
  for (int m = 0; m < n_materials; ++m) {
    result.combined_matrices[m] = A_decay + s * result.combined_matrices[m];
  }

  result.normalization_factor = s;
  result.fission_energy = total_fission_energy;
  return result;
}

} // namespace openmc

//==============================================================================
// C API
//==============================================================================

extern "C" int openmc_compute_depletion_rates(
  const double* tally_means,
  int n_materials,
  int n_tallied_nucs,
  int n_reactions,
  const int* nuc_chain_indices,
  const double* atom_counts,
  const double* volumes,
  double source_rate,
  int source_rate_type_int,
  int norm_mode_int,
  const double* fission_q,
  const double* heating_means,
  int fission_rx_idx,
  int* out_indptr,
  int* out_indices,
  double* out_data,
  int* out_nnz_per_mat,
  int* out_n_chain,
  double* out_normalization_factor,
  double* out_fission_energy)
{
  using namespace openmc;
  try {
    if (!data::depletion_chain) {
      set_errmsg("Depletion chain not loaded.");
      return OPENMC_E_UNASSIGNED;
    }

    auto srt = static_cast<SourceRateType>(source_rate_type_int);
    auto nm = static_cast<NormalizationMode>(norm_mode_int);

    DepletionRates rates = compute_depletion_rates(
      tally_means, n_materials, n_tallied_nucs, n_reactions,
      nuc_chain_indices, atom_counts, volumes, source_rate,
      srt, nm, fission_q, heating_means, fission_rx_idx);

    int n_chain = data::depletion_chain->size();
    *out_n_chain = n_chain;
    *out_normalization_factor = rates.normalization_factor;
    *out_fission_energy = rates.fission_energy;

    // Populate nnz_per_mat (always needed)
    for (int m = 0; m < n_materials; ++m) {
      out_nnz_per_mat[m] = rates.combined_matrices[m].nnz();
    }

    // If out_indices is null, caller is querying sizes only
    if (out_indices == nullptr || out_data == nullptr) {
      return 0;
    }

    // Pack CSC arrays contiguously
    int indptr_offset = 0;
    int data_offset = 0;
    for (int m = 0; m < n_materials; ++m) {
      const auto& mat = rates.combined_matrices[m];
      int mat_nnz = mat.nnz();

      std::copy(
        mat.indptr().begin(), mat.indptr().end(), out_indptr + indptr_offset);
      std::copy(
        mat.indices().begin(), mat.indices().end(), out_indices + data_offset);
      std::copy(
        mat.data().begin(), mat.data().end(), out_data + data_offset);

      indptr_offset += n_chain + 1;
      data_offset += mat_nnz;
    }

    return 0;
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}
