//! \file chain.cpp
//! \brief Depletion chain and associated information

#include "openmc/chain.h"

#include <algorithm> // for min_element
#include <cstdlib>   // for getenv
#include <memory>    // for make_unique
#include <sstream>   // for istringstream
#include <string>    // for stod

#include <fmt/core.h>
#include <pugixml.hpp>

#include "openmc/distribution.h" // for distribution_from_xml
#include "openmc/error.h"
#include "openmc/reaction.h"
#include "openmc/xml_interface.h" // for get_node_value

namespace openmc {

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
      // Treat "Nothing" as empty (no target nuclide)
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

    // Build TransmutationRxn entry
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
  if (fpy_node && !fpy_node.attribute("parent")) {
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
      // Read products list (only on first energy)
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

  // Handle borrowed fission yields (parent="..." attribute). A second pass
  // is needed because the parent nuclide may appear later in the file.
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

    nuclides_[nuc_idx]->set_fission_yields(
      nuclides_[parent_idx]->fission_yields());
  }

  // Build cached decay matrix and permuted lower-triangular structure.
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
    if (nuc.stable())
      continue;

    double decay_const = nuc.decay_constant();
    if (decay_const == 0.0)
      continue;

    // Diagonal loss term
    rows.push_back(i);
    cols.push_back(i);
    vals.push_back(-decay_const);

    // Off-diagonal gain terms
    for (const auto& dm : nuc.decay_modes()) {
      double branch_val = dm.branching_ratio * decay_const;
      if (branch_val == 0.0)
        continue;

      // Skip spontaneous fission targets — sf products are handled through
      // fission yield data, not direct decay targets.
      if (!dm.target.empty() && dm.type.find("sf") == std::string::npos) {
        int k = nuclide_index(dm.target);
        if (k >= 0) {
          rows.push_back(k);
          cols.push_back(i);
          vals.push_back(branch_val);
        }
      }

      // Produce alphas and protons from decay.
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

  decay_matrix_ = CSCMatrix::from_triplets(n, rows, cols, vals);

  // Compute topological permutation + structural reachability of the decay
  // DAG once. These are reused by the decay solver.
  decay_perm_ = decay_matrix_.pattern().topological_sort();
  decay_matrix_.pattern().reachability(
    decay_perm_, decay_reach_indptr_, decay_reach_indices_);

  // Build permuted lower-triangular structure for the decay solver: split
  // the diagonal and off-diagonal entries, with off-diagonal row indices
  // mapped into permuted space.
  const auto& a_indptr = decay_matrix_.indptr();
  const auto& a_indices = decay_matrix_.indices();
  const auto& a_data = decay_matrix_.data();

  vector<int> inv_perm(n);
  for (int i = 0; i < n; ++i) {
    inv_perm[decay_perm_[i]] = i;
  }

  decay_diag_.assign(n, 0.0);
  decay_lt_indptr_.assign(n + 1, 0);

  // First pass: count off-diagonal entries per topological column.
  for (int j = 0; j < n; ++j) {
    int orig_col = decay_perm_[j];
    int count = 0;
    for (int p = a_indptr[orig_col]; p < a_indptr[orig_col + 1]; ++p) {
      if (a_indices[p] == orig_col) {
        decay_diag_[j] = a_data[p];
      } else {
        ++count;
      }
    }
    decay_lt_indptr_[j + 1] = decay_lt_indptr_[j] + count;
  }

  // Second pass: fill off-diagonal arrays.
  int lt_nnz = decay_lt_indptr_[n];
  decay_lt_rowidx_.assign(lt_nnz, 0);
  decay_lt_data_.assign(lt_nnz, 0.0);
  for (int j = 0; j < n; ++j) {
    int orig_col = decay_perm_[j];
    int pos = decay_lt_indptr_[j];
    for (int p = a_indptr[orig_col]; p < a_indptr[orig_col + 1]; ++p) {
      int orig_row = a_indices[p];
      if (orig_row != orig_col) {
        decay_lt_rowidx_[pos] = inv_perm[orig_row];
        decay_lt_data_[pos] = a_data[p];
        ++pos;
      }
    }
  }
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

  data::depletion_chain = make_unique<DepletionChain>();
  data::depletion_chain->load_xml(chain_file_path);

  // Mirror nuclide map for transport code (D1S photon handling, parent
  // nuclide tally filtering).
  data::chain_nuclide_map = data::depletion_chain->nuclide_map();
}

} // namespace openmc
