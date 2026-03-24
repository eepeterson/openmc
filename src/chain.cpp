//! \file chain.cpp
//! \brief Depletion chain and associated information

#include "openmc/chain.h"

#include <cstdlib> // for getenv
#include <memory>  // for make_unique
#include <string>  // for stod, stoi

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
    dm.target = get_node_value(decay_node, "target");
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
      txn.target = get_node_value(reaction_node, "target");
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
    nuclides_.push_back(std::move(nuc));
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

  // Load chain (DepletionChain owns all ChainNuclide instances)
  data::depletion_chain = make_unique<DepletionChain>();
  data::depletion_chain->load_xml(chain_file_path);

  // Populate chain_nuclide_map for backward compatibility with transport
  // code (D1S photon handling, parent nuclide tally filtering)
  data::chain_nuclide_map = data::depletion_chain->nuclide_map();
}

} // namespace openmc
