//! \file chain.h
//! \brief Depletion chain and associated information

#ifndef OPENMC_CHAIN_H
#define OPENMC_CHAIN_H

#include <cmath>
#include <string>
#include <unordered_map>

#include "pugixml.hpp"

#include "openmc/angle_energy.h"  // for AngleEnergy
#include "openmc/distribution.h"  // for UPtrDist
#include "openmc/memory.h"        // for unique_ptr
#include "openmc/sparse_matrix.h" // for CSCMatrix, CSCPattern
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Data for a single nuclide in the depletion chain
//==============================================================================

class ChainNuclide {
public:
  //! Information about a reaction product (used by the transport D1S method)
  struct Product {
    std::string name;       //!< Reaction product name
    double branching_ratio; //!< Branching ratio
  };

  //! Decay mode information
  struct DecayMode {
    std::string type;       //!< Decay type, e.g. "beta-", "alpha"
    std::string target;     //!< Product nuclide name (empty if "nothing")
    double branching_ratio; //!< Branching ratio
  };

  //! Transmutation reaction information
  struct TransmutationRxn {
    std::string type;       //!< Reaction type, e.g. "(n,gamma)", "fission"
    std::string target;     //!< Product nuclide name (empty for fission)
    double branching_ratio; //!< Branching ratio
  };

  // Constructors
  explicit ChainNuclide(pugi::xml_node node);

  // Accessors
  const std::string& name() const { return name_; }
  double half_life() const { return half_life_; }
  double decay_energy() const { return decay_energy_; }

  //! Compute the decay constant for the nuclide
  //! \return Decay constant in [1/s]
  double decay_constant() const { return std::log(2.0) / half_life_; }

  //! Whether this nuclide is stable (infinite half-life)
  bool stable() const { return half_life_ == 0.0; }

  const Distribution* photon_energy() const { return photon_energy_.get(); }

  const std::unordered_map<int, vector<Product>>& reaction_products() const
  {
    return reaction_products_;
  }

  const vector<DecayMode>& decay_modes() const { return decay_modes_; }

  const vector<TransmutationRxn>& transmutation_reactions() const
  {
    return transmutation_rxns_;
  }

private:
  std::string name_;          //!< Name of nuclide (GNDS format)
  double half_life_ {0.0};    //!< Half-life in [s] (0.0 = stable)
  double decay_energy_ {0.0}; //!< Decay energy in [eV]
  std::unordered_map<int, vector<Product>>
    reaction_products_;                         //!< Map of MT to products
  UPtrDist photon_energy_;                      //!< Decay photon distribution
  vector<DecayMode> decay_modes_;               //!< Decay modes with targets
  vector<TransmutationRxn> transmutation_rxns_; //!< Transmutation reactions
};

//==============================================================================
// Angle-energy distribution for decay photon
//==============================================================================

class DecayPhotonAngleEnergy : public AngleEnergy {
public:
  explicit DecayPhotonAngleEnergy(const Distribution* dist)
    : photon_energy_(dist)
  {}

  //! Sample distribution for an angle and energy
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[out] mu Outgoing cosine with respect to current direction
  //! \param[inout] seed Pseudorandom seed pointer
  void sample(
    double E_in, double& E_out, double& mu, uint64_t* seed) const override;

  //! Sample an outgoing energy and evaluate the angular PDF
  //! \param[in] E_in Incoming energy in [eV]
  //! \param[in] mu Scattering cosine with respect to current direction
  //! \param[out] E_out Outgoing energy in [eV]
  //! \param[inout] seed Pseudorandom seed pointer
  //! \return Probability density for the scattering cosine
  double sample_energy_and_pdf(
    double E_in, double mu, double& E_out, uint64_t* seed) const override;

private:
  const Distribution* photon_energy_;
};

//==============================================================================
//! Depletion chain: owns nuclides and precomputed sparse matrices
//==============================================================================

class DepletionChain {
public:
  //! Load a depletion chain from an XML file
  //! \param[in] filename Path to the chain XML file
  void load_xml(const std::string& filename);

  // --- Nuclide access ---

  //! Number of nuclides in the chain
  int size() const { return nuclides_.size(); }

  //! Get a nuclide by index
  const ChainNuclide& nuclide(int i) const { return *nuclides_[i]; }

  //! Return the index of a nuclide by name, or -1 if not found
  int nuclide_index(const std::string& name) const;

  //! Map from nuclide name to index into nuclides_
  const std::unordered_map<std::string, int>& nuclide_map() const
  {
    return nuclide_map_;
  }

  // --- Precomputed data ---

  //! Topological permutation vector (maps new index → old index)
  const vector<int>& topo_permutation() const { return topo_perm_; }

  //! Decay matrix in the original (unpermuted) ordering
  const CSCMatrix& decay_matrix() const { return decay_matrix_; }

  //! Decay matrix in topologically permuted ordering
  const CSCMatrix& perm_decay_matrix() const { return perm_decay_matrix_; }

  //! Sparsity pattern of the Bateman matrix (union of all reaction channels)
  const CSCPattern& bateman_pattern() const { return bateman_pattern_; }

  //! Permuted Bateman pattern
  const CSCPattern& perm_bateman_pattern() const
  {
    return perm_bateman_pattern_;
  }

private:
  // --- Computation helpers (called at end of load_xml) ---
  void compute_topo_permutation();
  void compute_decay_matrix();
  void compute_bateman_pattern();

  // --- Data members ---
  vector<unique_ptr<ChainNuclide>> nuclides_;
  std::unordered_map<std::string, int> nuclide_map_;

  vector<int> topo_perm_;
  CSCMatrix decay_matrix_;
  CSCMatrix perm_decay_matrix_;
  CSCPattern bateman_pattern_;
  CSCPattern perm_bateman_pattern_;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

// Map from nuclide name to index in depletion chain (used by transport
// code for D1S photon handling and parent nuclide tally filtering)
extern std::unordered_map<std::string, int> chain_nuclide_map;

// Depletion chain (owns all ChainNuclide instances)
extern unique_ptr<DepletionChain> depletion_chain;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

void read_chain_file_xml();

} // namespace openmc

#endif // OPENMC_CHAIN_H
