#include "openmc/stellarator.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

TEST_CASE("Stellarator cubic integration and rejection bounds", "[stellarator]")
{
  for (int degree = 0; degree < 4; ++degree) {
    openmc::array<double, 4> c {};
    c[degree] = 1;
    REQUIRE(openmc::stellarator_integral(c, 1) == 1.0 / (degree + 1));
    REQUIRE(openmc::stellarator_rejection_bound(c) >= 1);
  }
  // PDF vanishes at both endpoints, and has a negative monomial coefficient.
  openmc::array<double, 4> c {0, 1, -1, 0};
  REQUIRE(std::abs(openmc::stellarator_integral(c, 1) - 1.0 / 6) < 1.e-15);
  REQUIRE(openmc::stellarator_rejection_bound(c) >= 0.25);
  // t*(1-t)^2 has its maximum at t=1/3, despite zero endpoint values.
  c = {0, 1, -2, 1};
  REQUIRE(std::abs(openmc::stellarator_integral(c, 1) - 1.0 / 12) < 1.e-15);
  REQUIRE(openmc::stellarator_rejection_bound(c) >= 4.0 / 27);
  // A cubic may have negative Bernstein coefficients and still be nonnegative.
  c = {1, -4, 4, 0};
  REQUIRE(openmc::stellarator_rejection_bound(c) >= 1);
}
