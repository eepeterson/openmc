//! \file chain.h
//! \brief Depletion chain and associated information

#ifndef OPENMC_CHAIN_H
#define OPENMC_CHAIN_H

#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>

#include "pugixml.hpp"

#include "openmc/angle_energy.h" // for AngleEnergy
#include "openmc/distribution.h" // for UPtrDist
#include "openmc/memory.h"       // for unique_ptr
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
    std::string target;     //!< Product nuclide name (empty if "Nothing")
    double branching_ratio; //!< Branching ratio
  };

  //! Transmutation reaction information
  struct TransmutationRxn {
    std::string type;       //!< Reaction type, e.g. "(n,gamma)", "fission"
    std::string target;     //!< Product nuclide name (empty for fission)
    double branching_ratio; //!< Branching ratio
  };

  //! Fission yield data at one or more energies
  struct FissionYieldData {
    vector<double> energies;             //!< Energies in [eV]
    vector<std::string> products;        //!< Product nuclide names
    vector<vector<double>> yield_matrix; //!< yield_matrix[i_energy][i_product]
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

  //! Whether this nuclide is stable
  bool stable() const { return half_life_ == 0.0; }

  //! Whether this nuclide has fission yield data
  bool has_fission_yields() const { return fission_yields_.has_value(); }

  //! Get fission yield data (must check has_fission_yields() first)
  const FissionYieldData& fission_yields() const { return *fission_yields_; }

  //! Set fission yield data (used for borrowed yields)
  void set_fission_yields(const FissionYieldData& fyd) { fission_yields_ = fyd; }

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
  std::string name_;          //!< Name of nuclide
  double half_life_ {0.0};    //!< Half-life in [s] (0.0 = stable)
  double decay_energy_ {0.0}; //!< Decay energy in [eV]
  std::unordered_map<int, vector<Product>>
    reaction_products_;                            //!< Map of MT to products
  UPtrDist photon_energy_;                         //!< Decay photon distribution
  vector<DecayMode> decay_modes_;                  //!< Decay modes with targets
  vector<TransmutationRxn> transmutation_rxns_;    //!< Transmutation reactions
  std::optional<FissionYieldData> fission_yields_; //!< Fission product yields
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
//! Depletion chain: owns nuclides parsed from a chain XML file
//==============================================================================

class DepletionChain {
public:
  //! Load a depletion chain from an XML file
  //! \param[in] filename Path to the chain XML file
  void load_xml(const std::string& filename);

  //! Number of nuclides in the chain
  int size() const { return nuclides_.size(); }

  //! Get a nuclide by index
  const ChainNuclide& nuclide(int i) const { return *nuclides_[i]; }

  //! Return the index of a nuclide by name, or -1 if not found
  int nuclide_index(const std::string& name) const;

  //! Map from nuclide name to index in nuclides_
  const std::unordered_map<std::string, int>& nuclide_map() const
  {
    return nuclide_map_;
  }

  //! List of unique reaction types present in the chain
  const vector<std::string>& reactions() const { return reactions_; }

  //! Map from reaction type string to index in reactions()
  const std::unordered_map<std::string, int>& reaction_map() const
  {
    return reaction_map_;
  }

  //! Get default fission yields (lowest energy) for all nuclides with yield
  //! data. Returns map of parent chain index -> {product chain index -> yield}.
  std::unordered_map<int, std::unordered_map<int, double>>
  get_default_fission_yields() const;

private:
  vector<unique_ptr<ChainNuclide>> nuclides_;
  std::unordered_map<std::string, int> nuclide_map_;
  vector<std::string> reactions_;
  std::unordered_map<std::string, int> reaction_map_;
};

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Map from nuclide name to index in depletion chain. Mirrors
//! depletion_chain->nuclide_map() and is populated by read_chain_file_xml.
//! Used by transport code (D1S photon handling, parent-nuclide tally filter).
extern std::unordered_map<std::string, int> chain_nuclide_map;

//! Depletion chain (owns all ChainNuclide instances)
extern unique_ptr<DepletionChain> depletion_chain;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

void read_chain_file_xml();

} // namespace openmc

#endif // OPENMC_CHAIN_H
