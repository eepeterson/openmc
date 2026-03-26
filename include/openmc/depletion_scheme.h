//! \file depletion_scheme.h
//! \brief Declarative integration scheme representation for depletion.
//!
//! Mirrors the Python IntegrationScheme structure. Schemes are flat arrays
//! of SchemeStep nodes that the depletion kernel interprets sequentially.

#ifndef OPENMC_DEPLETION_SCHEME_H
#define OPENMC_DEPLETION_SCHEME_H

#include <string>

#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Sentinel reference values
//==============================================================================

//! Special reference values for density/matrix sources.
//! Non-negative values are indices into the scheme's step array.
enum : int {
  REF_BOS = -1,       //!< Beginning-of-step density
  REF_PREV_STEP = -2, //!< Matrix from previous macro-step's first Transport
  REF_PREV_ITER = -3  //!< Density from previous SI iteration
};

//==============================================================================
// Weight function IDs for LE/QI schemes
//==============================================================================

//! Identifies how a matrix weight is computed.
//! STATIC means the weight is a compile-time constant.
//! Other values name LE/QI weight functions of (prev_dt, dt).
enum class WeightFn {
  STATIC,
  LEQI_W1_PREV,
  LEQI_W1_BOS,
  LEQI_W2_PREV,
  LEQI_W2_BOS,
  LEQI_W3_PREV,
  LEQI_W3_BOS,
  LEQI_W3_EOS,
  LEQI_W4_PREV,
  LEQI_W4_BOS,
  LEQI_W4_EOS
};

//! Evaluate a LE/QI weight function.
//! \param fn  Weight function identifier (must not be STATIC).
//! \param prev_dt  Previous timestep in seconds.
//! \param dt  Current timestep in seconds.
//! \return Scalar weight value.
double eval_weight_fn(WeightFn fn, double prev_dt, double dt);

//==============================================================================
// Scheme step types
//==============================================================================

enum class StepType {
  TRANSPORT,      //!< Run transport on a density → produce matrix set
  EXPM,           //!< Weighted matrix sum + CRAM solve → produce density
  ITERATE_BEGIN,  //!< Start an SI iteration loop
  ITERATE_END     //!< End of SI iteration loop body
};

//==============================================================================
// Weighted matrix term (used by EXPM steps)
//==============================================================================

//! One term in an Expm node's weighted matrix sum: w * A_source.
struct WeightedTerm {
  WeightFn weight_fn;    //!< How to compute the weight
  double static_weight;  //!< Value when weight_fn == STATIC
  int matrix_source;     //!< Transport step index, or REF_PREV_STEP
  bool use_average;      //!< True → use running average (SI iterate)
};

//==============================================================================
// Scheme step
//==============================================================================

//! A single step in an integration scheme.
struct SchemeStep {
  StepType type;

  //! Density source (Transport/Expm input).
  //! REF_BOS, REF_PREV_ITER, or index of an Expm step.
  int density_source {REF_BOS};

  //! Matrix terms for EXPM steps.
  vector<WeightedTerm> terms;

  //! Number of iterations for ITERATE_BEGIN steps.
  int n_iterations {0};
};

//==============================================================================
// Integration scheme
//==============================================================================

struct IntegrationScheme {
  std::string name;
  vector<SchemeStep> steps;
  int fallback_idx {-1}; //!< Index into schemes vector, or -1 if none
};

//! Get a scheme by name. Returns nullptr if not found.
//! Valid names: "predictor", "cecm", "celi", "cf4", "epc_rk4", "leqi",
//! "si_celi", "si_leqi".
const IntegrationScheme* get_scheme(const std::string& name);

} // namespace openmc

#endif // OPENMC_DEPLETION_SCHEME_H
