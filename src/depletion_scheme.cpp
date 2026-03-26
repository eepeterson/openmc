//! \file depletion_scheme.cpp
//! \brief Integration scheme definitions and weight functions.

#include "openmc/depletion_scheme.h"

#include <unordered_map>

#include "openmc/error.h"

namespace openmc {

//==============================================================================
// LE/QI weight functions
//==============================================================================

double eval_weight_fn(WeightFn fn, double prev_dt, double dt)
{
  switch (fn) {
  case WeightFn::LEQI_W1_PREV:
    return -dt / (12.0 * prev_dt);
  case WeightFn::LEQI_W1_BOS:
    return (dt + 6.0 * prev_dt) / (12.0 * prev_dt);
  case WeightFn::LEQI_W2_PREV:
    return -5.0 * dt / (12.0 * prev_dt);
  case WeightFn::LEQI_W2_BOS:
    return (5.0 * dt + 6.0 * prev_dt) / (12.0 * prev_dt);
  case WeightFn::LEQI_W3_PREV: {
    double d = 12.0 * prev_dt * (dt + prev_dt);
    return -dt * dt / d;
  }
  case WeightFn::LEQI_W3_BOS: {
    double d = 12.0 * prev_dt * (dt + prev_dt);
    return (dt * dt + 6.0 * dt * prev_dt + 5.0 * prev_dt * prev_dt) / d;
  }
  case WeightFn::LEQI_W3_EOS:
    return prev_dt / (12.0 * (dt + prev_dt));
  case WeightFn::LEQI_W4_PREV: {
    double d = 12.0 * prev_dt * (dt + prev_dt);
    return -dt * dt / d;
  }
  case WeightFn::LEQI_W4_BOS: {
    double d = 12.0 * prev_dt * (dt + prev_dt);
    return (dt * dt + 2.0 * dt * prev_dt + prev_dt * prev_dt) / d;
  }
  case WeightFn::LEQI_W4_EOS:
    return (4.0 * dt + 5.0 * prev_dt) / (12.0 * (dt + prev_dt));
  default:
    fatal_error("eval_weight_fn called with STATIC weight");
    return 0.0; // unreachable
  }
}

//==============================================================================
// Helper: build a WeightedTerm with a static weight
//==============================================================================

static WeightedTerm static_term(double w, int source, bool avg = false)
{
  return {WeightFn::STATIC, w, source, avg};
}

static WeightedTerm fn_term(WeightFn fn, int source, bool avg = false)
{
  return {fn, 0.0, source, avg};
}

//==============================================================================
// Scheme builders
//==============================================================================

// Node indices are assigned sequentially in the steps vector.
// We use local constants for readability.

static IntegrationScheme build_predictor()
{
  // 0: Transport(BOS)
  // 1: Expm(1.0 * A[0], BOS)
  IntegrationScheme s;
  s.name = "predictor";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {static_term(1.0, 0)};
  s.steps.push_back(std::move(e1));

  return s;
}

static IntegrationScheme build_cecm()
{
  // 0: Transport(BOS)
  // 1: Expm(0.5 * A[0], BOS)
  // 2: Transport(n[1])
  // 3: Expm(1.0 * A[2], BOS)
  IntegrationScheme s;
  s.name = "cecm";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {static_term(0.5, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep t2;
  t2.type = StepType::TRANSPORT;
  t2.density_source = 1;
  s.steps.push_back(std::move(t2));

  SchemeStep e3;
  e3.type = StepType::EXPM;
  e3.density_source = REF_BOS;
  e3.terms = {static_term(1.0, 2)};
  s.steps.push_back(std::move(e3));

  return s;
}

static IntegrationScheme build_celi()
{
  // 0: Transport(BOS)
  // 1: Expm(1.0 * A[0], BOS)       — predictor
  // 2: Transport(n[1])
  // 3: Expm(5/12 * A[0], 1/12 * A[2], BOS)
  // 4: Expm(1/12 * A[0], 5/12 * A[2], n[3])
  IntegrationScheme s;
  s.name = "celi";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {static_term(1.0, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep t2;
  t2.type = StepType::TRANSPORT;
  t2.density_source = 1;
  s.steps.push_back(std::move(t2));

  SchemeStep e3;
  e3.type = StepType::EXPM;
  e3.density_source = REF_BOS;
  e3.terms = {static_term(5.0 / 12.0, 0), static_term(1.0 / 12.0, 2)};
  s.steps.push_back(std::move(e3));

  SchemeStep e4;
  e4.type = StepType::EXPM;
  e4.density_source = 3;
  e4.terms = {static_term(1.0 / 12.0, 0), static_term(5.0 / 12.0, 2)};
  s.steps.push_back(std::move(e4));

  return s;
}

static IntegrationScheme build_cf4()
{
  // 0: Transport(BOS)
  // 1: Expm(1/2 * A[0], BOS)
  // 2: Transport(n[1])
  // 3: Expm(1/2 * A[2], BOS)
  // 4: Transport(n[3])
  // 5: Expm(-1/2 * A[0], 1.0 * A[4], n[1])
  // 6: Transport(n[5])
  // 7: Expm(-1/12 * A[0], 1/6 * A[2], 1/6 * A[4], 1/4 * A[6], BOS)
  // 8: Expm(1/4 * A[0], 1/6 * A[2], 1/6 * A[4], -1/12 * A[6], n[7])
  IntegrationScheme s;
  s.name = "cf4";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {static_term(0.5, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep t2;
  t2.type = StepType::TRANSPORT;
  t2.density_source = 1;
  s.steps.push_back(std::move(t2));

  SchemeStep e3;
  e3.type = StepType::EXPM;
  e3.density_source = REF_BOS;
  e3.terms = {static_term(0.5, 2)};
  s.steps.push_back(std::move(e3));

  SchemeStep t4;
  t4.type = StepType::TRANSPORT;
  t4.density_source = 3;
  s.steps.push_back(std::move(t4));

  SchemeStep e5;
  e5.type = StepType::EXPM;
  e5.density_source = 1;
  e5.terms = {static_term(-0.5, 0), static_term(1.0, 4)};
  s.steps.push_back(std::move(e5));

  SchemeStep t6;
  t6.type = StepType::TRANSPORT;
  t6.density_source = 5;
  s.steps.push_back(std::move(t6));

  SchemeStep e7;
  e7.type = StepType::EXPM;
  e7.density_source = REF_BOS;
  e7.terms = {static_term(-1.0 / 12.0, 0), static_term(1.0 / 6.0, 2),
    static_term(1.0 / 6.0, 4), static_term(0.25, 6)};
  s.steps.push_back(std::move(e7));

  SchemeStep e8;
  e8.type = StepType::EXPM;
  e8.density_source = 7;
  e8.terms = {static_term(0.25, 0), static_term(1.0 / 6.0, 2),
    static_term(1.0 / 6.0, 4), static_term(-1.0 / 12.0, 6)};
  s.steps.push_back(std::move(e8));

  return s;
}

static IntegrationScheme build_epc_rk4()
{
  // 0: Transport(BOS)
  // 1: Expm(1/2 * A[0], BOS)
  // 2: Transport(n[1])
  // 3: Expm(1/2 * A[2], BOS)
  // 4: Transport(n[3])
  // 5: Expm(1.0 * A[4], BOS)
  // 6: Transport(n[5])
  // 7: Expm(1/6 * A[0], 1/3 * A[2], 1/3 * A[4], 1/6 * A[6], BOS)
  IntegrationScheme s;
  s.name = "epc_rk4";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {static_term(0.5, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep t2;
  t2.type = StepType::TRANSPORT;
  t2.density_source = 1;
  s.steps.push_back(std::move(t2));

  SchemeStep e3;
  e3.type = StepType::EXPM;
  e3.density_source = REF_BOS;
  e3.terms = {static_term(0.5, 2)};
  s.steps.push_back(std::move(e3));

  SchemeStep t4;
  t4.type = StepType::TRANSPORT;
  t4.density_source = 3;
  s.steps.push_back(std::move(t4));

  SchemeStep e5;
  e5.type = StepType::EXPM;
  e5.density_source = REF_BOS;
  e5.terms = {static_term(1.0, 4)};
  s.steps.push_back(std::move(e5));

  SchemeStep t6;
  t6.type = StepType::TRANSPORT;
  t6.density_source = 5;
  s.steps.push_back(std::move(t6));

  SchemeStep e7;
  e7.type = StepType::EXPM;
  e7.density_source = REF_BOS;
  e7.terms = {static_term(1.0 / 6.0, 0), static_term(1.0 / 3.0, 2),
    static_term(1.0 / 3.0, 4), static_term(1.0 / 6.0, 6)};
  s.steps.push_back(std::move(e7));

  return s;
}

static IntegrationScheme build_leqi()
{
  // 0: Transport(BOS)
  // 1: Expm(w1_prev * PREV_STEP, w1_bos * A[0], BOS)       — LE half
  // 2: Expm(w2_prev * PREV_STEP, w2_bos * A[0], n[1])      — LE full
  // 3: Transport(n[2])
  // 4: Expm(w3_prev * PREV_STEP, w3_bos * A[0], w3_eos * A[3], BOS) — QI half
  // 5: Expm(w4_prev * PREV_STEP, w4_bos * A[0], w4_eos * A[3], n[4]) — QI full
  IntegrationScheme s;
  s.name = "leqi";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {fn_term(WeightFn::LEQI_W1_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W1_BOS, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep e2;
  e2.type = StepType::EXPM;
  e2.density_source = 1;
  e2.terms = {fn_term(WeightFn::LEQI_W2_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W2_BOS, 0)};
  s.steps.push_back(std::move(e2));

  SchemeStep t3;
  t3.type = StepType::TRANSPORT;
  t3.density_source = 2;
  s.steps.push_back(std::move(t3));

  SchemeStep e4;
  e4.type = StepType::EXPM;
  e4.density_source = REF_BOS;
  e4.terms = {fn_term(WeightFn::LEQI_W3_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W3_BOS, 0),
    fn_term(WeightFn::LEQI_W3_EOS, 3)};
  s.steps.push_back(std::move(e4));

  SchemeStep e5;
  e5.type = StepType::EXPM;
  e5.density_source = 4;
  e5.terms = {fn_term(WeightFn::LEQI_W4_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W4_BOS, 0),
    fn_term(WeightFn::LEQI_W4_EOS, 3)};
  s.steps.push_back(std::move(e5));

  return s;
}

static IntegrationScheme build_si_celi()
{
  // 0: Transport(BOS)
  // 1: Expm(1.0 * A[0], BOS)             — predictor
  // 2: ITERATE_BEGIN(n_iterations)
  // 3:   Transport(PREV_ITER)
  // 4:   Expm(5/12 * A[0], 1/12 * avg(A[3]), BOS)
  // 5:   Expm(1/12 * A[0], 5/12 * avg(A[3]), n[4])
  // 6: ITERATE_END
  IntegrationScheme s;
  s.name = "si_celi";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {static_term(1.0, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep ib;
  ib.type = StepType::ITERATE_BEGIN;
  ib.n_iterations = 10;
  s.steps.push_back(std::move(ib));

  SchemeStep t3;
  t3.type = StepType::TRANSPORT;
  t3.density_source = REF_PREV_ITER;
  s.steps.push_back(std::move(t3));

  SchemeStep e4;
  e4.type = StepType::EXPM;
  e4.density_source = REF_BOS;
  e4.terms = {static_term(5.0 / 12.0, 0),
    static_term(1.0 / 12.0, 3, /*avg=*/true)};
  s.steps.push_back(std::move(e4));

  SchemeStep e5;
  e5.type = StepType::EXPM;
  e5.density_source = 4;
  e5.terms = {static_term(1.0 / 12.0, 0),
    static_term(5.0 / 12.0, 3, /*avg=*/true)};
  s.steps.push_back(std::move(e5));

  SchemeStep ie;
  ie.type = StepType::ITERATE_END;
  s.steps.push_back(std::move(ie));

  return s;
}

static IntegrationScheme build_si_leqi()
{
  // 0: Transport(BOS)
  // 1: Expm(w1_prev * PREV_STEP, w1_bos * A[0], BOS)   — LE half
  // 2: Expm(w2_prev * PREV_STEP, w2_bos * A[0], n[1])  — LE full
  // 3: ITERATE_BEGIN(n_iterations)
  // 4:   Transport(PREV_ITER)
  // 5:   Expm(w3_prev * PREV_STEP, w3_bos * A[0], w3_eos * avg(A[4]), BOS)
  // 6:   Expm(w4_prev * PREV_STEP, w4_bos * A[0], w4_eos * avg(A[4]), n[5])
  // 7: ITERATE_END
  IntegrationScheme s;
  s.name = "si_leqi";

  SchemeStep t0;
  t0.type = StepType::TRANSPORT;
  t0.density_source = REF_BOS;
  s.steps.push_back(std::move(t0));

  SchemeStep e1;
  e1.type = StepType::EXPM;
  e1.density_source = REF_BOS;
  e1.terms = {fn_term(WeightFn::LEQI_W1_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W1_BOS, 0)};
  s.steps.push_back(std::move(e1));

  SchemeStep e2;
  e2.type = StepType::EXPM;
  e2.density_source = 1;
  e2.terms = {fn_term(WeightFn::LEQI_W2_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W2_BOS, 0)};
  s.steps.push_back(std::move(e2));

  SchemeStep ib;
  ib.type = StepType::ITERATE_BEGIN;
  ib.n_iterations = 10;
  s.steps.push_back(std::move(ib));

  SchemeStep t4;
  t4.type = StepType::TRANSPORT;
  t4.density_source = REF_PREV_ITER;
  s.steps.push_back(std::move(t4));

  SchemeStep e5;
  e5.type = StepType::EXPM;
  e5.density_source = REF_BOS;
  e5.terms = {fn_term(WeightFn::LEQI_W3_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W3_BOS, 0),
    fn_term(WeightFn::LEQI_W3_EOS, 4, /*avg=*/true)};
  s.steps.push_back(std::move(e5));

  SchemeStep e6;
  e6.type = StepType::EXPM;
  e6.density_source = 5;
  e6.terms = {fn_term(WeightFn::LEQI_W4_PREV, REF_PREV_STEP),
    fn_term(WeightFn::LEQI_W4_BOS, 0),
    fn_term(WeightFn::LEQI_W4_EOS, 4, /*avg=*/true)};
  s.steps.push_back(std::move(e6));

  SchemeStep ie;
  ie.type = StepType::ITERATE_END;
  s.steps.push_back(std::move(ie));

  return s;
}

//==============================================================================
// Scheme registry
//==============================================================================

static vector<IntegrationScheme> build_all_schemes()
{
  vector<IntegrationScheme> schemes;
  schemes.push_back(build_predictor());   // 0
  schemes.push_back(build_cecm());        // 1
  schemes.push_back(build_celi());        // 2
  schemes.push_back(build_cf4());         // 3
  schemes.push_back(build_epc_rk4());     // 4
  schemes.push_back(build_leqi());        // 5
  schemes.push_back(build_si_celi());     // 6
  schemes.push_back(build_si_leqi());     // 7

  // Set fallback indices
  schemes[5].fallback_idx = 2; // leqi → celi
  schemes[7].fallback_idx = 6; // si_leqi → si_celi

  return schemes;
}

const IntegrationScheme* get_scheme(const std::string& name)
{
  // Constructed once on first call
  static const vector<IntegrationScheme> schemes = build_all_schemes();

  for (const auto& s : schemes) {
    if (s.name == name)
      return &s;
  }
  return nullptr;
}

} // namespace openmc
