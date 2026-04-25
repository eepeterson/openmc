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
