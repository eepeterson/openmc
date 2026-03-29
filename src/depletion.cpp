//! \file depletion.cpp
//! \brief Implementation of high-level depletion functions

#include "openmc/depletion.h"

#include <algorithm> // for copy, sort
#include <cstring>   // for memcpy
#include <unordered_map>

#include "openmc/capi.h"
#include "openmc/chain.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/simulation.h"
#include "openmc/sparse_matrix.h"
#include "openmc/tallies/tally.h"

namespace openmc {

// Joules per eV
static constexpr double JOULE_PER_EV = 1.602176634e-19;

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
      s = source_rate / (total_fission_energy * JOULE_PER_EV);
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

int update_depletable_materials(
  int n_materials,
  const int32_t* material_indices,
  int n_chain,
  const double* atom_counts,
  const double* volumes,
  const int* transportable,
  int* nonzero_nuc_indices)
{
  auto& chain = *data::depletion_chain;

  // Track which chain indices have nonzero density in any material
  vector<bool> has_nonzero(n_chain, false);

  for (int m = 0; m < n_materials; ++m) {
    const double* n_atoms = atom_counts + m * n_chain;
    double vol = volumes[m];

    vector<std::string> nuclides;
    vector<double> densities;

    for (int j = 0; j < n_chain; ++j) {
      if (!transportable[j])
        continue;
      double dens = n_atoms[j] / vol * 1.0e-24; // atom/b-cm
      if (dens > 0.0) {
        nuclides.push_back(chain.nuclide(j).name());
        densities.push_back(dens);
        has_nonzero[j] = true;
      }
    }

    if (!nuclides.empty()) {
      int32_t idx = material_indices[m];
      model::materials[idx]->set_densities(nuclides, densities);
    }
  }

  // Collect sorted nonzero chain indices
  int count = 0;
  for (int j = 0; j < n_chain; ++j) {
    if (has_nonzero[j]) {
      nonzero_nuc_indices[count++] = j;
    }
  }
  return count;
}

//==============================================================================
// Macro-timestep kernel
//==============================================================================

// Helper: extract tally means from a tally's results buffer.
// Returns a vector of length n_filter_bins * n_nucs * n_scores.
static vector<double> get_tally_means(int32_t tally_idx)
{
  auto& tally = *model::tallies[tally_idx];
  int n_real = tally.n_realizations_;
  const auto& res = tally.results();
  // results shape: [n_filter_bins, n_nuclides * n_scores, 3]
  // where the last axis is (VALUE, SUM, SUM_SQ)
  int n_filter_bins = res.shape()[0];
  int n_nuc_scores = res.shape()[1];
  int n_bins = n_filter_bins * n_nuc_scores;
  vector<double> means(n_bins);
  if (n_real > 0) {
    for (int fb = 0; fb < n_filter_bins; ++fb) {
      for (int ns = 0; ns < n_nuc_scores; ++ns) {
        means[fb * n_nuc_scores + ns] = res(fb, ns, 1) / n_real;
      }
    }
  }
  return means;
}

// Helper: run transport for one node in the scheme.  Updates materials,
// resets tallies, calls openmc_run, extracts combined matrices.
static void handle_transport(
  DepletionState& state,
  const vector<double>& dens_flat,
  double source_rate,
  bool run_transport,
  vector<CSCMatrix>& out_matrices,
  double& out_k_eff,
  double& out_fission_energy)
{
  auto& chain = *data::depletion_chain;
  int n_chain = chain.size();

  if (run_transport) {
    // Update material compositions
    vector<int> nonzero_idx(n_chain);
    int n_nonzero = update_depletable_materials(
      state.n_materials, state.material_indices.data(), n_chain,
      dens_flat.data(), state.volumes.data(),
      state.transportable.data(), nonzero_idx.data());

    // Update tally nuclide list to match nonzero nuclides
    vector<std::string> nuc_names;
    state.nuc_chain_indices.clear();
    for (int k = 0; k < n_nonzero; ++k) {
      int ci = nonzero_idx[k];
      nuc_names.push_back(chain.nuclide(ci).name());
      state.nuc_chain_indices.push_back(ci);
    }
    state.n_tallied_nucs = n_nonzero;

    model::tallies[state.rate_tally_idx]->set_nuclides(nuc_names);

    // Reset and run transport
    openmc_reset();
    openmc_run();

    // Read k_eff
    out_k_eff = simulation::keff;

    // Extract tally means
    auto rate_means = get_tally_means(state.rate_tally_idx);

    // Get heating means if energy-deposition normalization
    const double* heating_ptr = nullptr;
    vector<double> heating_means;
    if (state.norm_mode == NormalizationMode::energy_deposition &&
        state.heating_tally_idx >= 0) {
      heating_means = get_tally_means(state.heating_tally_idx);
      heating_ptr = heating_means.data();
    }

    // Compute combined matrices (A_decay + s * A_rxn)
    auto rates = compute_depletion_rates(
      rate_means.data(), state.n_materials, state.n_tallied_nucs,
      state.n_reactions, state.nuc_chain_indices.data(),
      dens_flat.data(), state.volumes.data(), source_rate,
      state.source_rate_type, state.norm_mode,
      state.fission_q.data(), heating_ptr, state.fission_rx_idx);

    out_matrices = std::move(rates.combined_matrices);
    out_fission_energy = rates.fission_energy;
  }
  // If !run_transport, out_matrices retains its previous values (cached)
}

// Helper: execute an Expm step — weighted matrix sum + CRAM solve
static void handle_expm(
  const SchemeStep& step,
  DepletionState& state,
  int n_chain,
  double dt,
  double prev_dt,
  // maps from step index to matrices per material:
  const std::unordered_map<int, vector<CSCMatrix>>& matrix_store,
  // maps from step index to densities per material:
  const std::unordered_map<int, vector<vector<double>>>& density_store,
  const vector<CSCMatrix>* prev_step_matrices,
  const std::unordered_map<int, vector<CSCMatrix>>* avg_matrix_store,
  vector<vector<double>>& out_densities)
{
  int n_mats = state.n_materials;

  // Build weighted matrix sum per material
  vector<CSCMatrix> combined;
  combined.reserve(n_mats);
  for (int m = 0; m < n_mats; ++m) {
    combined.push_back(CSCMatrix(n_chain));
  }

  for (const auto& term : step.terms) {
    // Resolve weight
    double w;
    if (term.weight_fn == WeightFn::STATIC) {
      w = term.static_weight;
    } else {
      w = eval_weight_fn(term.weight_fn, prev_dt, dt);
    }

    // Resolve matrix source
    const vector<CSCMatrix>* mats = nullptr;
    if (term.matrix_source == REF_PREV_STEP) {
      mats = prev_step_matrices;
    } else if (term.use_average && avg_matrix_store) {
      auto it = avg_matrix_store->find(term.matrix_source);
      if (it != avg_matrix_store->end())
        mats = &it->second;
    } else {
      auto it = matrix_store.find(term.matrix_source);
      if (it != matrix_store.end())
        mats = &it->second;
    }

    if (mats) {
      for (int m = 0; m < n_mats; ++m) {
        combined[m] += w * (*mats)[m];
      }
    }
  }

  // Resolve input density
  const vector<vector<double>>* input_dens = nullptr;
  if (step.density_source == REF_BOS) {
    auto it = density_store.find(REF_BOS);
    if (it != density_store.end())
      input_dens = &it->second;
  } else if (step.density_source == REF_PREV_ITER) {
    auto it = density_store.find(REF_PREV_ITER);
    if (it != density_store.end())
      input_dens = &it->second;
  } else {
    auto it = density_store.find(step.density_source);
    if (it != density_store.end())
      input_dens = &it->second;
  }

  // CRAM solve per material
  if (!input_dens) {
    fatal_error("Depletion Expm step: density source not found in store.");
  }
  out_densities.resize(n_mats);
  for (int m = 0; m < n_mats; ++m) {
    out_densities[m] =
      state.cram_solver.solve(combined[m], (*input_dens)[m], dt);
  }
}

SchemeStepResult execute_scheme_step(
  const IntegrationScheme& scheme,
  DepletionState& state,
  const vector<vector<double>>& n_bos,
  double dt,
  double source_rate,
  const vector<CSCMatrix>* prev_step_matrices,
  double prev_dt,
  bool run_transport)
{
  auto& chain = *data::depletion_chain;
  int n_chain = chain.size();
  int n_mats = state.n_materials;

  // Storage for results indexed by step index
  std::unordered_map<int, vector<vector<double>>> density_store;
  std::unordered_map<int, vector<CSCMatrix>> matrix_store;

  // Store BOS density under REF_BOS sentinel
  density_store[REF_BOS] = n_bos;

  // Cache for transport results (reused when !run_transport)
  vector<CSCMatrix> cached_matrices;
  double cached_k_eff = 1.0;
  double cached_fission_energy = 0.0;
  int bos_transport_idx = -1;

  // Track last Expm index before an Iterate (for PREV_ITER seeding)
  int last_expm_idx = -1;

  for (int i = 0; i < static_cast<int>(scheme.steps.size()); ++i) {
    const auto& step = scheme.steps[i];

    switch (step.type) {
    case StepType::TRANSPORT: {
      // Resolve density to use for material update
      const vector<vector<double>>* dens_ptr = nullptr;
      if (step.density_source == REF_BOS) {
        dens_ptr = &density_store[REF_BOS];
      } else if (step.density_source == REF_PREV_ITER) {
        dens_ptr = &density_store[REF_PREV_ITER];
      } else {
        dens_ptr = &density_store[step.density_source];
      }

      // Flatten densities for the C-style functions
      vector<double> dens_flat(n_mats * n_chain);
      for (int m = 0; m < n_mats; ++m) {
        std::copy((*dens_ptr)[m].begin(), (*dens_ptr)[m].end(),
          dens_flat.data() + m * n_chain);
      }

      handle_transport(state, dens_flat, source_rate, run_transport,
        cached_matrices, cached_k_eff, cached_fission_energy);

      // Store matrices under this step's index
      matrix_store[i] = cached_matrices;

      if (bos_transport_idx < 0) {
        bos_transport_idx = i;
      }
      break;
    }

    case StepType::EXPM: {
      vector<vector<double>> result;
      handle_expm(step, state, n_chain, dt, prev_dt,
        matrix_store, density_store, prev_step_matrices,
        nullptr, result);
      density_store[i] = std::move(result);
      last_expm_idx = i;
      break;
    }

    case StepType::ITERATE_BEGIN: {
      // Seed PREV_ITER
      if (last_expm_idx >= 0) {
        density_store[REF_PREV_ITER] = density_store[last_expm_idx];
      } else {
        density_store[REF_PREV_ITER] = density_store[REF_BOS];
      }

      // Find body range [i+1 .. iterate_end)
      int body_start = i + 1;
      int body_end = body_start;
      for (int j = body_start; j < static_cast<int>(scheme.steps.size()); ++j) {
        if (scheme.steps[j].type == StepType::ITERATE_END) {
          body_end = j;
          break;
        }
      }

      // Running average matrices per transport step index
      std::unordered_map<int, vector<CSCMatrix>> avg_matrices;

      for (int iter = 1; iter <= step.n_iterations; ++iter) {
        int body_last_expm = -1;

        for (int b = body_start; b < body_end; ++b) {
          const auto& bstep = scheme.steps[b];

          if (bstep.type == StepType::TRANSPORT) {
            // Resolve density
            const vector<vector<double>>* dens_ptr = nullptr;
            if (bstep.density_source == REF_PREV_ITER) {
              dens_ptr = &density_store[REF_PREV_ITER];
            } else if (bstep.density_source == REF_BOS) {
              dens_ptr = &density_store[REF_BOS];
            } else {
              dens_ptr = &density_store[bstep.density_source];
            }

            vector<double> dens_flat(n_mats * n_chain);
            for (int m = 0; m < n_mats; ++m) {
              std::copy((*dens_ptr)[m].begin(), (*dens_ptr)[m].end(),
                dens_flat.data() + m * n_chain);
            }

            handle_transport(state, dens_flat, source_rate, run_transport,
              cached_matrices, cached_k_eff, cached_fission_energy);
            matrix_store[b] = cached_matrices;

            // Update running average
            if (iter == 1) {
              avg_matrices[b] = cached_matrices;
            } else {
              double alpha = 1.0 / iter;
              for (int m = 0; m < n_mats; ++m) {
                // avg = alpha * current + (1 - alpha) * avg
                avg_matrices[b][m] =
                  alpha * cached_matrices[m] +
                  (1.0 - alpha) * avg_matrices[b][m];
              }
            }

          } else if (bstep.type == StepType::EXPM) {
            vector<vector<double>> result;
            handle_expm(bstep, state, n_chain, dt, prev_dt,
              matrix_store, density_store, prev_step_matrices,
              &avg_matrices, result);
            density_store[b] = std::move(result);
            body_last_expm = b;
          }
        }

        // Update PREV_ITER to last Expm in body
        if (body_last_expm >= 0) {
          density_store[REF_PREV_ITER] = density_store[body_last_expm];
        }
      }

      // Skip past ITERATE_END
      i = body_end;
      break;
    }

    case StepType::ITERATE_END:
      // Should not be reached (handled by ITERATE_BEGIN)
      break;
    }
  }

  // Find final Expm (last one in the scheme)
  int final_expm_idx = -1;
  for (int i = static_cast<int>(scheme.steps.size()) - 1; i >= 0; --i) {
    if (scheme.steps[i].type == StepType::EXPM) {
      final_expm_idx = i;
      break;
    }
    // Also check inside iterate body (last Expm before ITERATE_END)
    if (scheme.steps[i].type == StepType::ITERATE_END) {
      for (int j = i - 1; j >= 0; --j) {
        if (scheme.steps[j].type == StepType::EXPM) {
          final_expm_idx = j;
          break;
        }
        if (scheme.steps[j].type == StepType::ITERATE_BEGIN) {
          break;
        }
      }
      if (final_expm_idx >= 0)
        break;
    }
  }

  SchemeStepResult result;
  if (final_expm_idx >= 0) {
    result.eos_densities = std::move(density_store[final_expm_idx]);
  } else {
    result.eos_densities = n_bos; // fallback
  }
  result.k_eff = cached_k_eff;

  // BOS matrices for next step's PREV_STEP reference
  if (bos_transport_idx >= 0) {
    result.bos_matrices = std::move(matrix_store[bos_transport_idx]);
  }

  return result;
}

} // namespace openmc

//==============================================================================
// C API
//==============================================================================

// Global depletion state (owned by the library, configured via C API)
static openmc::unique_ptr<openmc::DepletionState> g_depletion_state;

extern "C" int openmc_depletion_set_config(
  int n_materials,
  const int32_t* material_indices,
  const double* volumes,
  const int* transportable,
  int32_t rate_tally_idx,
  int32_t heating_tally_idx,
  int n_reactions,
  int fission_rx_idx,
  const double* fission_q,
  int norm_mode,
  int source_rate_type,
  int solver_order)
{
  using namespace openmc;
  try {
    if (!data::depletion_chain) {
      set_errmsg("Depletion chain not loaded.");
      return OPENMC_E_UNASSIGNED;
    }
    int n_chain = data::depletion_chain->size();

    auto state = make_unique<DepletionState>();
    state->n_materials = n_materials;
    state->material_indices.assign(material_indices,
      material_indices + n_materials);
    state->volumes.assign(volumes, volumes + n_materials);
    state->transportable.assign(transportable, transportable + n_chain);
    state->rate_tally_idx = rate_tally_idx;
    state->heating_tally_idx = heating_tally_idx;
    state->n_reactions = n_reactions;
    state->fission_rx_idx = fission_rx_idx;
    state->fission_q.assign(fission_q, fission_q + n_chain);
    state->norm_mode = static_cast<NormalizationMode>(norm_mode);
    state->source_rate_type = static_cast<SourceRateType>(source_rate_type);
    state->solver_order = solver_order;

    if (solver_order == 16) {
      state->cram_solver = IPFCramSolver(IPFCramSolver::Order::cram16);
    } else {
      state->cram_solver = IPFCramSolver(IPFCramSolver::Order::cram48);
    }

    g_depletion_state = std::move(state);
    return 0;
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}

extern "C" int openmc_depletion_execute_step(
  const char* scheme_name,
  const double* n_bos_flat,
  double dt,
  double source_rate,
  double prev_dt,
  int run_transport,
  double* out_eos_flat,
  double* out_k_eff)
{
  using namespace openmc;
  try {
    if (!g_depletion_state) {
      set_errmsg("Depletion state not configured.");
      return OPENMC_E_UNASSIGNED;
    }
    if (!data::depletion_chain) {
      set_errmsg("Depletion chain not loaded.");
      return OPENMC_E_UNASSIGNED;
    }

    auto& state = *g_depletion_state;
    int n_chain = data::depletion_chain->size();
    int n_mats = state.n_materials;

    const IntegrationScheme* scheme = get_scheme(scheme_name);
    if (!scheme) {
      set_errmsg("Unknown integration scheme: " + std::string(scheme_name));
      return OPENMC_E_INVALID_ARGUMENT;
    }

    // Unpack BOS densities from flat array
    vector<vector<double>> n_bos(n_mats);
    for (int m = 0; m < n_mats; ++m) {
      n_bos[m].assign(n_bos_flat + m * n_chain,
        n_bos_flat + (m + 1) * n_chain);
    }

    // Use previous step matrices from internal state
    const vector<CSCMatrix>* prev_mats_ptr =
      state.prev_bos_matrices.empty() ? nullptr : &state.prev_bos_matrices;

    // Execute the scheme step
    SchemeStepResult result = execute_scheme_step(
      *scheme, state, n_bos, dt, source_rate,
      prev_mats_ptr, prev_dt, run_transport != 0);

    // Store BOS matrices for next step's PREV_STEP reference
    state.prev_bos_matrices = std::move(result.bos_matrices);
    state.prev_dt = dt;

    // Pack EOS densities
    if (out_eos_flat) {
      for (int m = 0; m < n_mats; ++m) {
        std::copy(result.eos_densities[m].begin(),
          result.eos_densities[m].end(),
          out_eos_flat + m * n_chain);
      }
    }

    if (out_k_eff) {
      *out_k_eff = result.k_eff;
    }

    return 0;
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
}

extern "C" int openmc_depletion_free()
{
  g_depletion_state.reset();
  return 0;
}
