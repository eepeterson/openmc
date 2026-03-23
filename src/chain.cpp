//! \file chain.cpp
//! \brief Depletion chain and associated information

#include "openmc/chain.h"

#include <cstdlib> // for getenv
#include <memory>  // for make_unique
#include <queue>   // for priority_queue
#include <string>  // for stod, stoi
#include <tuple>   // for tuple

#include <fmt/core.h>
#include <pugixml.hpp>

#include "openmc/distribution.h" // for distribution_from_xml
#include "openmc/error.h"
#include "openmc/particle_type.h" // for parse_gnds_nuclide
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

  // Precompute derived quantities
  compute_topo_permutation();
  compute_decay_matrix();
  compute_bateman_pattern();
}

void DepletionChain::compute_topo_permutation()
{
  int n = nuclides_.size();
  // Build adjacency list: parent -> children (via decay and reactions)
  vector<vector<int>> children(n);
  vector<int> in_degree(n, 0);

  for (int i = 0; i < n; ++i) {
    const auto& nuc = *nuclides_[i];

    // Decay edges
    for (const auto& dm : nuc.decay_modes()) {
      int j = nuclide_index(dm.target);
      if (j >= 0) {
        children[i].push_back(j);
        ++in_degree[j];
      }
    }

    // Reaction edges
    for (const auto& rxn : nuc.transmutation_reactions()) {
      int j = nuclide_index(rxn.target);
      if (j >= 0) {
        children[i].push_back(j);
        ++in_degree[j];
      }
    }
  }

  // Kahn's algorithm with tie-breaking by (-A, -Z, -m) so that
  // heavier nuclides come first in the topological ordering
  struct ZamKey {
    int neg_A, neg_Z, neg_m, index;
    bool operator>(const ZamKey& o) const
    {
      return std::tie(neg_A, neg_Z, neg_m) >
             std::tie(o.neg_A, o.neg_Z, o.neg_m);
    }
  };

  // Pre-parse ZAM for all nuclides
  vector<ZamKey> keys(n);
  for (int i = 0; i < n; ++i) {
    int Z = 0, A = 0, m = 0;
    parse_gnds_nuclide(nuclides_[i]->name(), Z, A, m);
    keys[i] = {-A, -Z, -m, i};
  }

  // Min-heap ordered by (-A,-Z,-m) so that largest A comes out first
  std::priority_queue<ZamKey, vector<ZamKey>, std::greater<ZamKey>> pq;
  for (int i = 0; i < n; ++i) {
    if (in_degree[i] == 0) {
      pq.push(keys[i]);
    }
  }

  topo_perm_.clear();
  topo_perm_.reserve(n);
  while (!pq.empty()) {
    auto [neg_A, neg_Z, neg_m, u] = pq.top();
    pq.pop();
    topo_perm_.push_back(u);
    for (int v : children[u]) {
      if (--in_degree[v] == 0) {
        pq.push(keys[v]);
      }
    }
  }

  if (static_cast<int>(topo_perm_.size()) != n) {
    warning("Depletion chain has cycles; topological sort is incomplete.");
  }
}

void DepletionChain::compute_decay_matrix()
{
  int n = nuclides_.size();

  // Collect COO triplets for the decay matrix
  vector<int> rows, cols;
  vector<double> vals;

  for (int i = 0; i < n; ++i) {
    const auto& nuc = *nuclides_[i];
    if (nuc.stable())
      continue;

    double lambda = nuc.decay_constant();

    // Diagonal: loss term
    rows.push_back(i);
    cols.push_back(i);
    vals.push_back(-lambda);

    // Off-diagonal: gain terms from decay
    for (const auto& dm : nuc.decay_modes()) {
      int j = nuclide_index(dm.target);
      if (j >= 0) {
        rows.push_back(j);
        cols.push_back(i);
        vals.push_back(lambda * dm.branching_ratio);
      }
    }
  }

  decay_matrix_ = CSCMatrix::from_triplets(n, rows, cols, vals);
  perm_decay_matrix_ = decay_matrix_.permute(topo_perm_);
}

void DepletionChain::compute_bateman_pattern()
{
  int n = nuclides_.size();

  // The Bateman pattern includes every (row, col) that could ever be nonzero
  // in the full Bateman matrix: diagonal, decay channels, and reaction channels
  vector<int> rows, cols;

  for (int i = 0; i < n; ++i) {
    const auto& nuc = *nuclides_[i];

    // Diagonal is always present (loss term from any reaction/decay)
    rows.push_back(i);
    cols.push_back(i);

    // Decay channels
    for (const auto& dm : nuc.decay_modes()) {
      int j = nuclide_index(dm.target);
      if (j >= 0) {
        rows.push_back(j);
        cols.push_back(i);
      }
    }

    // Reaction channels
    for (const auto& rxn : nuc.transmutation_reactions()) {
      int j = nuclide_index(rxn.target);
      if (j >= 0) {
        rows.push_back(j);
        cols.push_back(i);
      }
    }
  }

  bateman_pattern_ = CSCPattern::from_triplets(n, rows, cols);
  perm_bateman_pattern_ = bateman_pattern_.permute(topo_perm_);
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
