//! \file chain.cpp
//! \brief Depletion chain and associated information

#include "openmc/chain.h"

#include <algorithm> // for sort, min_element
#include <cmath>     // for log
#include <cstdlib>   // for getenv
#include <memory>    // for make_unique
#include <sstream>   // for istringstream
#include <string>    // for stod, stoi
#include <unordered_set>

#include <fmt/core.h>
#include <pugixml.hpp>

#include "openmc/capi.h"
#include "openmc/distribution.h" // for distribution_from_xml
#include "openmc/error.h"
#include "openmc/reaction.h"
#include "openmc/xml_interface.h" // for get_node_value

namespace openmc {

//==============================================================================
// Reaction secondaries map
//
// Maps reaction type strings to the light nuclide(s) produced as secondaries.
// Mirrors the Python REACTIONS dict in openmc/deplete/chain.py.
//==============================================================================

namespace {

// Initialized once; looked up via the function below
const std::unordered_map<std::string, vector<std::string>>&
reaction_secondaries_map()
{
  static const std::unordered_map<std::string, vector<std::string>> m = {
    {"(n,2nd)", {"H2"}},
    {"(n,2n)", {}},
    {"(n,3n)", {}},
    {"(n,na)", {"He4"}},
    {"(n,n3a)", {"He4", "He4", "He4"}},
    {"(n,2na)", {"He4"}},
    {"(n,3na)", {"He4"}},
    {"(n,np)", {"H1"}},
    {"(n,n2a)", {"He4", "He4"}},
    {"(n,2n2a)", {"He4", "He4"}},
    {"(n,nd)", {"H2"}},
    {"(n,nt)", {"H3"}},
    {"(n,n3He)", {"He3"}},
    {"(n,nd2a)", {"H2", "He4", "He4"}},
    {"(n,nt2a)", {"H3", "He4", "He4"}},
    {"(n,4n)", {}},
    {"(n,2np)", {"H1"}},
    {"(n,3np)", {"H1"}},
    {"(n,n2p)", {"H1", "H1"}},
    {"(n,npa)", {"H1", "He4"}},
    {"(n,gamma)", {}},
    {"(n,p)", {"H1"}},
    {"(n,d)", {"H2"}},
    {"(n,t)", {"H3"}},
    {"(n,3He)", {"He3"}},
    {"(n,a)", {"He4"}},
    {"(n,2a)", {"He4", "He4"}},
    {"(n,3a)", {"He4", "He4", "He4"}},
    {"(n,2p)", {"H1", "H1"}},
    {"(n,pa)", {"H1", "He4"}},
    {"(n,t2a)", {"H3", "He4", "He4"}},
    {"(n,d2a)", {"H2", "He4", "He4"}},
    {"(n,pd)", {"H1", "H2"}},
    {"(n,pt)", {"H1", "H3"}},
    {"(n,da)", {"H2", "He4"}},
    {"(n,5n)", {}},
    {"(n,6n)", {}},
    {"(n,2nt)", {"H3"}},
    {"(n,ta)", {"H3", "He4"}},
    {"(n,4np)", {"H1"}},
    {"(n,3nd)", {"H2"}},
    {"(n,nda)", {"H2", "He4"}},
    {"(n,2npa)", {"H1", "He4"}},
    {"(n,7n)", {}},
    {"(n,8n)", {}},
    {"(n,5np)", {"H1"}},
    {"(n,6np)", {"H1"}},
    {"(n,7np)", {"H1"}},
    {"(n,4na)", {"He4"}},
    {"(n,5na)", {"He4"}},
    {"(n,6na)", {"He4"}},
    {"(n,7na)", {"He4"}},
    {"(n,4nd)", {"H2"}},
    {"(n,5nd)", {"H2"}},
    {"(n,6nd)", {"H2"}},
    {"(n,3nt)", {"H3"}},
    {"(n,4nt)", {"H3"}},
    {"(n,5nt)", {"H3"}},
    {"(n,6nt)", {"H3"}},
    {"(n,2n3He)", {"He3"}},
    {"(n,3n3He)", {"He3"}},
    {"(n,4n3He)", {"He3"}},
    {"(n,3n2p)", {"H1", "H1"}},
    {"(n,3n2a)", {"He4", "He4"}},
    {"(n,3npa)", {"H1", "He4"}},
    {"(n,dt)", {"H2", "H3"}},
    {"(n,npd)", {"H1", "H2"}},
    {"(n,npt)", {"H1", "H3"}},
    {"(n,ndt)", {"H2", "H3"}},
    {"(n,np3He)", {"H1", "He3"}},
    {"(n,nd3He)", {"H2", "He3"}},
    {"(n,nt3He)", {"H3", "He3"}},
    {"(n,nta)", {"H3", "He4"}},
    {"(n,2n2p)", {"H1", "H1"}},
    {"(n,p3He)", {"H1", "He3"}},
    {"(n,d3He)", {"H2", "He3"}},
    {"(n,3Hea)", {"He3", "He4"}},
    {"(n,4n2p)", {"H1", "H1"}},
    {"(n,4n2a)", {"He4", "He4"}},
    {"(n,4npa)", {"H1", "He4"}},
    {"(n,3p)", {"H1", "H1", "H1"}},
    {"(n,n3p)", {"H1", "H1", "H1"}},
    {"(n,3n2pa)", {"H1", "H1", "He4"}},
    {"(n,5n2p)", {"H1", "H1"}},
  };
  return m;
}

} // anonymous namespace

//==============================================================================
// ChainNuclide implementation
//==============================================================================

ChainNuclide::ChainNuclide(pugi::xml_node node)
{
  name_ = get_node_value(node, "name");
  if (check_for_node(node, "half_life")) {
    half_life_ = std::stod(get_node_value(node, "half_life"));
  }
  if (check_for_node(node, "decay_energy")) {
    decay_energy_ = std::stod(get_node_value(node, "decay_energy"));
  }

  // Read decay modes
  for (pugi::xml_node decay_node : node.children("decay")) {
    DecayMode dm;
    dm.type = get_node_value(decay_node, "type");
    if (check_for_node(decay_node, "target")) {
      std::string target_str = get_node_value(decay_node, "target");
      // Treat "Nothing" as empty (no target in chain)
      if (target_str != "Nothing" && target_str != "nothing") {
        dm.target = target_str;
      }
    }
    dm.branching_ratio =
      std::stod(get_node_value(decay_node, "branching_ratio"));
    decay_modes_.push_back(std::move(dm));
  }

  // Read reactions
  for (pugi::xml_node reaction_node : node.children("reaction")) {
    std::string rx_name = get_node_value(reaction_node, "type");

    // Build TransmutationRxn
    TransmutationRxn txn;
    txn.type = rx_name;
    if (reaction_node.attribute("target")) {
      std::string target_str = get_node_value(reaction_node, "target");
      if (target_str != "Nothing" && target_str != "nothing") {
        txn.target = target_str;
      }
    }
    txn.branching_ratio = 1.0;
    if (reaction_node.attribute("branching_ratio")) {
      txn.branching_ratio =
        std::stod(get_node_value(reaction_node, "branching_ratio"));
    }
    transmutation_rxns_.push_back(std::move(txn));

    // Also populate MT -> product map (used by transport for D1S)
    if (!reaction_node.attribute("target"))
      continue;
    std::string rx_target = get_node_value(reaction_node, "target");
    if (rx_target == "Nothing" || rx_target == "nothing")
      continue;
    double branching_ratio = 1.0;
    if (reaction_node.attribute("branching_ratio")) {
      branching_ratio =
        std::stod(get_node_value(reaction_node, "branching_ratio"));
    }
    int mt = reaction_mt(rx_name);
    reaction_products_[mt].push_back({rx_target, branching_ratio});
  }

  // Read decay photon source
  for (pugi::xml_node source_node : node.children("source")) {
    auto particle = get_node_value(source_node, "particle");
    if (particle == "photon") {
      photon_energy_ = distribution_from_xml(source_node);
      break;
    }
  }

  // Read fission product yields
  pugi::xml_node fpy_node = node.child("neutron_fission_yields");
  if (fpy_node) {
    // Check for borrowing from another nuclide (handled at chain level)
    if (!fpy_node.attribute("parent")) {
      FissionYieldData fyd;

      // Read energies
      pugi::xml_node energies_node = fpy_node.child("energies");
      if (energies_node) {
        std::istringstream iss(energies_node.child_value());
        double e;
        while (iss >> e) {
          fyd.energies.push_back(e);
        }
      }

      // Read yield data for each energy
      for (pugi::xml_node fy_node : fpy_node.children("fission_yields")) {
        // Read products
        if (fyd.products.empty()) {
          pugi::xml_node prod_node = fy_node.child("products");
          if (prod_node) {
            std::istringstream iss(prod_node.child_value());
            std::string prod;
            while (iss >> prod) {
              fyd.products.push_back(prod);
            }
          }
        }

        // Read yield values
        pugi::xml_node data_node = fy_node.child("data");
        if (data_node) {
          vector<double> yields;
          std::istringstream iss(data_node.child_value());
          double y;
          while (iss >> y) {
            yields.push_back(y);
          }
          fyd.yield_matrix.push_back(std::move(yields));
        }
      }

      if (!fyd.energies.empty() && !fyd.products.empty()) {
        fission_yields_ = std::move(fyd);
      }
    }
  }
}

//==============================================================================
// DecayPhotonAngleEnergy implementation
//==============================================================================

void DecayPhotonAngleEnergy::sample(
  double E_in, double& E_out, double& mu, uint64_t* seed) const
{
  E_out = photon_energy_->sample(seed).first;
  mu = Uniform(-1., 1.).sample(seed).first;
}

double DecayPhotonAngleEnergy::sample_energy_and_pdf(
  double E_in, double mu, double& E_out, uint64_t* seed) const
{
  E_out = photon_energy_->sample(seed).first;
  return 0.5;
}

//==============================================================================
// DepletionChain implementation
//==============================================================================

int DepletionChain::nuclide_index(const std::string& name) const
{
  auto it = nuclide_map_.find(name);
  return it != nuclide_map_.end() ? it->second : -1;
}

void DepletionChain::load_xml(const std::string& filename)
{
  pugi::xml_document doc;
  auto result = doc.load_file(filename.c_str());
  if (!result) {
    fatal_error(fmt::format("Error processing chain file: {}", filename));
  }

  pugi::xml_node root = doc.document_element();

  // Parse all nuclides
  for (auto node : root.children("nuclide")) {
    auto nuc = std::make_unique<ChainNuclide>(node);
    nuclide_map_[nuc->name()] = nuclides_.size();

    // Build reactions list
    for (const auto& rxn : nuc->transmutation_reactions()) {
      if (reaction_map_.find(rxn.type) == reaction_map_.end()) {
        reaction_map_[rxn.type] = reactions_.size();
        reactions_.push_back(rxn.type);
      }
    }

    nuclides_.push_back(std::move(nuc));
  }

  // Handle borrowed fission yields (parent="..." attribute).
  // A second pass is needed because the parent nuclide may appear later
  // in the file.
  for (auto node : root.children("nuclide")) {
    pugi::xml_node fpy_node = node.child("neutron_fission_yields");
    if (!fpy_node)
      continue;
    if (!fpy_node.attribute("parent"))
      continue;

    std::string nuc_name = get_node_value(node, "name");
    std::string parent_name = fpy_node.attribute("parent").value();
    int nuc_idx = nuclide_index(nuc_name);
    int parent_idx = nuclide_index(parent_name);

    if (nuc_idx < 0 || parent_idx < 0)
      continue;
    if (!nuclides_[parent_idx]->has_fission_yields())
      continue;

    // Copy fission yield data from parent
    nuclides_[nuc_idx]->set_fission_yields(
      nuclides_[parent_idx]->fission_yields());
  }

  // Build and cache the decay matrix (constant across materials/timesteps)
  build_decay_matrix();
}

std::unordered_map<int, std::unordered_map<int, double>>
DepletionChain::get_default_fission_yields() const
{
  std::unordered_map<int, std::unordered_map<int, double>> result;

  for (int i = 0; i < size(); ++i) {
    const auto& nuc = *nuclides_[i];
    if (!nuc.has_fission_yields())
      continue;

    const auto& fyd = nuc.fission_yields();
    // Use lowest energy
    int min_idx = 0;
    if (fyd.energies.size() > 1) {
      min_idx = std::min_element(fyd.energies.begin(), fyd.energies.end()) -
                fyd.energies.begin();
    }

    std::unordered_map<int, double> product_yields;
    const auto& yields = fyd.yield_matrix[min_idx];
    for (int j = 0; j < static_cast<int>(fyd.products.size()); ++j) {
      if (yields[j] != 0.0) {
        int k = nuclide_index(fyd.products[j]);
        if (k >= 0) {
          product_yields[k] = yields[j];
        }
      }
    }
    result[i] = std::move(product_yields);
  }

  return result;
}

void DepletionChain::build_decay_matrix()
{
  int n = size();
  vector<int> rows, cols;
  vector<double> vals;
  rows.reserve(n * 2);
  cols.reserve(n * 2);
  vals.reserve(n * 2);

  for (int i = 0; i < n; ++i) {
    const auto& nuc = *nuclides_[i];

    if (!nuc.stable()) {
      double decay_const = nuc.decay_constant();

      // Diagonal loss term
      if (decay_const != 0.0) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(-decay_const);
      }

      // Off-diagonal gain terms
      for (const auto& dm : nuc.decay_modes()) {
        double branch_val = dm.branching_ratio * decay_const;
        if (branch_val == 0.0)
          continue;

        if (!dm.target.empty()) {
          int k = nuclide_index(dm.target);
          if (k >= 0) {
            rows.push_back(k);
            cols.push_back(i);
            vals.push_back(branch_val);
          }
        }

        // Produce alphas and protons from decay
        if (dm.type.find("alpha") != std::string::npos) {
          int k = nuclide_index("He4");
          if (k >= 0) {
            int count = 0;
            std::string::size_type pos = 0;
            while ((pos = dm.type.find("alpha", pos)) != std::string::npos) {
              ++count;
              pos += 5;
            }
            rows.push_back(k);
            cols.push_back(i);
            vals.push_back(count * branch_val);
          }
        } else if (dm.type.find('p') != std::string::npos) {
          int k = nuclide_index("H1");
          if (k >= 0) {
            int count = 0;
            for (char c : dm.type) {
              if (c == 'p')
                ++count;
            }
            rows.push_back(k);
            cols.push_back(i);
            vals.push_back(count * branch_val);
          }
        }
      }
    }
  }

  decay_matrix_ = CSCMatrix::from_triplets(n, rows, cols, vals);
}

CSCMatrix DepletionChain::form_rxn_matrix(
  const double* rates,
  int n_nucs_with_rates,
  int n_reactions,
  const int* nuc_to_chain_idx,
  const std::unordered_map<int, std::unordered_map<int, double>>&
    fission_yields) const
{
  // Use default yields if none provided
  const auto& fy = fission_yields.empty()
                      ? get_default_fission_yields()
                      : fission_yields;

  const auto& sec_map = reaction_secondaries_map();
  int n = size();

  vector<int> rows, cols;
  vector<double> vals;
  rows.reserve(n * 2);
  cols.reserve(n * 2);
  vals.reserve(n * 2);

  // Build reverse map: chain index -> rate-array nuclide index
  std::unordered_map<int, int> chain_to_rate_idx;
  for (int r = 0; r < n_nucs_with_rates; ++r) {
    chain_to_rate_idx[nuc_to_chain_idx[r]] = r;
  }

  for (int i = 0; i < n; ++i) {
    const auto& nuc = *nuclides_[i];

    auto rate_it = chain_to_rate_idx.find(i);
    if (rate_it == chain_to_rate_idx.end())
      continue;

    int rate_nuc_idx = rate_it->second;
    const double* nuc_rates = rates + rate_nuc_idx * n_reactions;

    // Track which reaction types we've already counted the loss for
    std::unordered_set<std::string> seen_rx;

    for (const auto& rxn : nuc.transmutation_reactions()) {
      auto rx_map_it = reaction_map_.find(rxn.type);
      if (rx_map_it == reaction_map_.end())
        continue;

      int rx_idx = rx_map_it->second;
      double path_rate = nuc_rates[rx_idx];

      // Loss term — only count once per reaction type
      if (seen_rx.find(rxn.type) == seen_rx.end()) {
        seen_rx.insert(rxn.type);
        if (path_rate != 0.0) {
          rows.push_back(i);
          cols.push_back(i);
          vals.push_back(-path_rate);
        }
      }

      // Gain term
      if (rxn.type != "fission") {
        if (!rxn.target.empty() && path_rate != 0.0) {
          int k = nuclide_index(rxn.target);
          if (k >= 0) {
            rows.push_back(k);
            cols.push_back(i);
            vals.push_back(path_rate * rxn.branching_ratio);
          }
        }

        // Light nuclide secondaries
        auto sec_it = sec_map.find(rxn.type);
        if (sec_it != sec_map.end()) {
          for (const auto& light_nuc : sec_it->second) {
            int k = nuclide_index(light_nuc);
            if (k >= 0) {
              rows.push_back(k);
              cols.push_back(i);
              vals.push_back(path_rate * rxn.branching_ratio);
            }
          }
        }
      } else {
        // Fission: add yield contributions
        auto fy_it = fy.find(i);
        if (fy_it != fy.end()) {
          for (const auto& [product_idx, yield_val] : fy_it->second) {
            double val = yield_val * path_rate;
            if (val != 0.0) {
              rows.push_back(product_idx);
              cols.push_back(i);
              vals.push_back(val);
            }
          }
        }
      }
    }
  }

  return CSCMatrix::from_triplets(n, rows, cols, vals);
}

CSCMatrix DepletionChain::form_matrix(
  const double* rates,
  int n_nucs_with_rates,
  int n_reactions,
  const int* nuc_to_chain_idx,
  const std::unordered_map<int, std::unordered_map<int, double>>&
    fission_yields) const
{
  return decay_matrix_ + form_rxn_matrix(
    rates, n_nucs_with_rates, n_reactions, nuc_to_chain_idx, fission_yields);
}

//==============================================================================
// Global variables
//==============================================================================

namespace data {

std::unordered_map<std::string, int> chain_nuclide_map;
unique_ptr<DepletionChain> depletion_chain;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

void read_chain_file_xml()
{
  char* chain_file_path = std::getenv("OPENMC_CHAIN_FILE");
  if (!chain_file_path) {
    return;
  }

  write_message(5, "Reading chain file: {}...", chain_file_path);

  // Load chain (DepletionChain owns all ChainNuclide instances)
  data::depletion_chain = make_unique<DepletionChain>();
  data::depletion_chain->load_xml(chain_file_path);

  // Populate chain_nuclide_map for backward compatibility with transport
  // code (D1S photon handling, parent nuclide tally filtering)
  data::chain_nuclide_map = data::depletion_chain->nuclide_map();
}

} // namespace openmc

//==============================================================================
// C API
//==============================================================================

extern "C" int openmc_load_depletion_chain(const char* filename)
{
  using namespace openmc;
  try {
    data::depletion_chain = make_unique<DepletionChain>();
    data::depletion_chain->load_xml(filename);
    data::chain_nuclide_map = data::depletion_chain->nuclide_map();
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_chain_form_matrix(
  const double* rates, int n_nucs_with_rates, int n_reactions,
  const int* nuc_chain_indices,
  int n_fission_parents,
  const int* fy_parent_indices,
  const int* fy_product_indices,
  const double* fy_yields,
  const int* fy_products_per_parent,
  int* out_indptr, int* out_indices, double* out_data,
  int* out_nnz, int* out_n)
{
  using namespace openmc;
  try {
    if (!data::depletion_chain) {
      set_errmsg("Depletion chain not loaded. Call openmc_load_depletion_chain "
                 "first.");
      return OPENMC_E_UNASSIGNED;
    }

    const auto& chain = *data::depletion_chain;

    // Build fission yields map if provided
    std::unordered_map<int, std::unordered_map<int, double>> fy_map;
    if (n_fission_parents > 0 && fy_parent_indices && fy_product_indices &&
        fy_yields && fy_products_per_parent) {
      int offset = 0;
      for (int p = 0; p < n_fission_parents; ++p) {
        int parent_idx = fy_parent_indices[p];
        int n_prods = fy_products_per_parent[p];
        std::unordered_map<int, double> prod_map;
        for (int j = 0; j < n_prods; ++j) {
          prod_map[fy_product_indices[offset + j]] = fy_yields[offset + j];
        }
        fy_map[parent_idx] = std::move(prod_map);
        offset += n_prods;
      }
    }

    // Form the matrix
    CSCMatrix mat = chain.form_matrix(
      rates, n_nucs_with_rates, n_reactions, nuc_chain_indices, fy_map);

    // Populate output
    *out_n = mat.n();
    *out_nnz = mat.nnz();

    // If out_indices is null, caller is just querying sizes
    if (out_indices == nullptr || out_data == nullptr) {
      return 0;
    }

    // Copy CSC data to output arrays
    const auto& indptr = mat.indptr();
    const auto& indices = mat.indices();
    const auto& data = mat.data();
    std::copy(indptr.begin(), indptr.end(), out_indptr);
    std::copy(indices.begin(), indices.end(), out_indices);
    std::copy(data.begin(), data.end(), out_data);

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_chain_form_rxn_matrix(
  const double* rates, int n_nucs_with_rates, int n_reactions,
  const int* nuc_chain_indices,
  int n_fission_parents,
  const int* fy_parent_indices,
  const int* fy_product_indices,
  const double* fy_yields,
  const int* fy_products_per_parent,
  int* out_indptr, int* out_indices, double* out_data,
  int* out_nnz, int* out_n)
{
  using namespace openmc;
  try {
    if (!data::depletion_chain) {
      set_errmsg("Depletion chain not loaded. Call openmc_load_depletion_chain "
                 "first.");
      return OPENMC_E_UNASSIGNED;
    }

    const auto& chain = *data::depletion_chain;

    // Build fission yields map if provided
    std::unordered_map<int, std::unordered_map<int, double>> fy_map;
    if (n_fission_parents > 0 && fy_parent_indices && fy_product_indices &&
        fy_yields && fy_products_per_parent) {
      int offset = 0;
      for (int p = 0; p < n_fission_parents; ++p) {
        int parent_idx = fy_parent_indices[p];
        int n_prods = fy_products_per_parent[p];
        std::unordered_map<int, double> prod_map;
        for (int j = 0; j < n_prods; ++j) {
          prod_map[fy_product_indices[offset + j]] = fy_yields[offset + j];
        }
        fy_map[parent_idx] = std::move(prod_map);
        offset += n_prods;
      }
    }

    // Form the reaction-rate-only matrix (no decay)
    CSCMatrix mat = chain.form_rxn_matrix(
      rates, n_nucs_with_rates, n_reactions, nuc_chain_indices, fy_map);

    // Populate output
    *out_n = mat.n();
    *out_nnz = mat.nnz();

    // If out_indices is null, caller is just querying sizes
    if (out_indices == nullptr || out_data == nullptr) {
      return 0;
    }

    // Copy CSC data to output arrays
    std::copy(mat.indptr().begin(), mat.indptr().end(), out_indptr);
    std::copy(mat.indices().begin(), mat.indices().end(), out_indices);
    std::copy(mat.data().begin(), mat.data().end(), out_data);

  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}
