#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_approx.hpp"
#include "openmc/depletion_scheme.h"

using namespace openmc;

TEST_CASE("Scheme lookup by name")
{
  REQUIRE(get_scheme("predictor") != nullptr);
  REQUIRE(get_scheme("cecm") != nullptr);
  REQUIRE(get_scheme("celi") != nullptr);
  REQUIRE(get_scheme("cf4") != nullptr);
  REQUIRE(get_scheme("epc_rk4") != nullptr);
  REQUIRE(get_scheme("leqi") != nullptr);
  REQUIRE(get_scheme("si_celi") != nullptr);
  REQUIRE(get_scheme("si_leqi") != nullptr);
  REQUIRE(get_scheme("nonexistent") == nullptr);
}

TEST_CASE("Predictor scheme structure")
{
  auto* s = get_scheme("predictor");
  REQUIRE(s->steps.size() == 2);

  // Step 0: Transport(BOS)
  CHECK(s->steps[0].type == StepType::TRANSPORT);
  CHECK(s->steps[0].density_source == REF_BOS);

  // Step 1: Expm(1.0 * A[0], BOS)
  CHECK(s->steps[1].type == StepType::EXPM);
  CHECK(s->steps[1].density_source == REF_BOS);
  REQUIRE(s->steps[1].terms.size() == 1);
  CHECK(s->steps[1].terms[0].static_weight == 1.0);
  CHECK(s->steps[1].terms[0].matrix_source == 0);

  CHECK(s->fallback_idx == -1);
}

TEST_CASE("CECM scheme structure")
{
  auto* s = get_scheme("cecm");
  REQUIRE(s->steps.size() == 4);

  CHECK(s->steps[0].type == StepType::TRANSPORT);
  CHECK(s->steps[1].type == StepType::EXPM);
  CHECK(s->steps[1].terms[0].static_weight == 0.5);
  CHECK(s->steps[2].type == StepType::TRANSPORT);
  CHECK(s->steps[2].density_source == 1);
  CHECK(s->steps[3].type == StepType::EXPM);
  CHECK(s->steps[3].terms[0].static_weight == 1.0);
  CHECK(s->steps[3].terms[0].matrix_source == 2);
}

TEST_CASE("CELI scheme structure")
{
  auto* s = get_scheme("celi");
  REQUIRE(s->steps.size() == 5);

  // Corrector half-step
  auto& e3 = s->steps[3];
  REQUIRE(e3.terms.size() == 2);
  CHECK(e3.terms[0].static_weight == Catch::Approx(5.0 / 12.0));
  CHECK(e3.terms[0].matrix_source == 0);
  CHECK(e3.terms[1].static_weight == Catch::Approx(1.0 / 12.0));
  CHECK(e3.terms[1].matrix_source == 2);

  // Corrector full-step
  auto& e4 = s->steps[4];
  CHECK(e4.density_source == 3);
  REQUIRE(e4.terms.size() == 2);
  CHECK(e4.terms[0].static_weight == Catch::Approx(1.0 / 12.0));
  CHECK(e4.terms[1].static_weight == Catch::Approx(5.0 / 12.0));
}

TEST_CASE("CF4 scheme structure")
{
  auto* s = get_scheme("cf4");
  REQUIRE(s->steps.size() == 9);

  // Final step: Expm(1/4 * A[0], 1/6 * A[2], 1/6 * A[4], -1/12 * A[6], n[7])
  auto& e8 = s->steps[8];
  CHECK(e8.density_source == 7);
  REQUIRE(e8.terms.size() == 4);
  CHECK(e8.terms[0].static_weight == Catch::Approx(0.25));
  CHECK(e8.terms[3].static_weight == Catch::Approx(-1.0 / 12.0));
  CHECK(e8.terms[3].matrix_source == 6);
}

TEST_CASE("LE/QI scheme uses callable weights and has fallback")
{
  auto* s = get_scheme("leqi");
  REQUIRE(s->steps.size() == 6);

  // First Expm: fn weights, one references PREV_STEP
  auto& e1 = s->steps[1];
  CHECK(e1.type == StepType::EXPM);
  REQUIRE(e1.terms.size() == 2);
  CHECK(e1.terms[0].weight_fn == WeightFn::LEQI_W1_PREV);
  CHECK(e1.terms[0].matrix_source == REF_PREV_STEP);
  CHECK(e1.terms[1].weight_fn == WeightFn::LEQI_W1_BOS);
  CHECK(e1.terms[1].matrix_source == 0);

  // Has fallback to celi
  CHECK(s->fallback_idx >= 0);
}

TEST_CASE("SI-CELI has iterate block with averages")
{
  auto* s = get_scheme("si_celi");
  REQUIRE(s->steps.size() == 7);

  CHECK(s->steps[2].type == StepType::ITERATE_BEGIN);
  CHECK(s->steps[2].n_iterations == 10);
  CHECK(s->steps[6].type == StepType::ITERATE_END);

  // Transport inside iterate uses PREV_ITER
  CHECK(s->steps[3].type == StepType::TRANSPORT);
  CHECK(s->steps[3].density_source == REF_PREV_ITER);

  // Expm uses average of transport[3]
  auto& e4 = s->steps[4];
  REQUIRE(e4.terms.size() == 2);
  CHECK(e4.terms[1].use_average == true);
  CHECK(e4.terms[1].matrix_source == 3);
}

TEST_CASE("SI-LEQI has iterate + callable weights + fallback")
{
  auto* s = get_scheme("si_leqi");
  REQUIRE(s->steps.size() == 8);

  // LE predictor steps before iterate
  CHECK(s->steps[0].type == StepType::TRANSPORT);
  CHECK(s->steps[1].type == StepType::EXPM);
  CHECK(s->steps[2].type == StepType::EXPM);

  // Iterate block
  CHECK(s->steps[3].type == StepType::ITERATE_BEGIN);
  CHECK(s->steps[7].type == StepType::ITERATE_END);

  // Inside iterate: QI corrector with avg + callable weights
  auto& e5 = s->steps[5];
  REQUIRE(e5.terms.size() == 3);
  CHECK(e5.terms[0].weight_fn == WeightFn::LEQI_W3_PREV);
  CHECK(e5.terms[2].weight_fn == WeightFn::LEQI_W3_EOS);
  CHECK(e5.terms[2].use_average == true);

  // Fallback to si_celi
  CHECK(s->fallback_idx >= 0);
}

TEST_CASE("LE/QI weight functions evaluate correctly")
{
  double prev_dt = 100.0;
  double dt = 200.0;

  // w1_prev = -dt / (12 * prev_dt) = -200 / 1200 = -1/6
  CHECK(eval_weight_fn(WeightFn::LEQI_W1_PREV, prev_dt, dt) ==
        Catch::Approx(-1.0 / 6.0));

  // w1_bos = (dt + 6*prev_dt) / (12*prev_dt) = 800/1200 = 2/3
  CHECK(eval_weight_fn(WeightFn::LEQI_W1_BOS, prev_dt, dt) ==
        Catch::Approx(2.0 / 3.0));

  // Weights should sum correctly for each pair:
  // w1_prev + w1_bos = 1/2 (half-step)
  CHECK(eval_weight_fn(WeightFn::LEQI_W1_PREV, prev_dt, dt) +
            eval_weight_fn(WeightFn::LEQI_W1_BOS, prev_dt, dt) ==
        Catch::Approx(0.5));

  // w2_prev + w2_bos = 0.5 (second half-step in split LE)
  CHECK(eval_weight_fn(WeightFn::LEQI_W2_PREV, prev_dt, dt) +
            eval_weight_fn(WeightFn::LEQI_W2_BOS, prev_dt, dt) ==
        Catch::Approx(0.5));

  // w3_prev + w3_bos + w3_eos should sum to 0.5 (half-step QI)
  CHECK(eval_weight_fn(WeightFn::LEQI_W3_PREV, prev_dt, dt) +
            eval_weight_fn(WeightFn::LEQI_W3_BOS, prev_dt, dt) +
            eval_weight_fn(WeightFn::LEQI_W3_EOS, prev_dt, dt) ==
        Catch::Approx(0.5));

  // w4_prev + w4_bos + w4_eos should sum to 0.5 (second half-step in split QI)
  CHECK(eval_weight_fn(WeightFn::LEQI_W4_PREV, prev_dt, dt) +
            eval_weight_fn(WeightFn::LEQI_W4_BOS, prev_dt, dt) +
            eval_weight_fn(WeightFn::LEQI_W4_EOS, prev_dt, dt) ==
        Catch::Approx(0.5));
}
