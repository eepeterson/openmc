#include "openmc/constants.h"
#include "openmc/stellarator.h"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

TEST_CASE("Stellarator cubic CDF inversion", "[stellarator]")
{
  for (int degree = 0; degree < 4; ++degree) {
    openmc::array<double, 4> c {};
    c[degree] = 1;
    for (double u :
      {0.0, 1.e-30, 1.e-12, 0.01, 0.5, 0.99, std::nextafter(1.0, 0.0), 1.0}) {
      double t = openmc::stellarator_invert(c, u);
      REQUIRE(std::abs(t - std::pow(u, 1.0 / (degree + 1))) < 4.e-15);
    }
  }
  // PDF vanishes at both endpoints, and has a negative monomial coefficient.
  openmc::array<double, 4> c {0, 1, -1, 0};
  for (double t : {0.0, 0.001, 0.2, 0.5, 0.99, 1.0}) {
    double u = 3 * t * t - 2 * t * t * t;
    REQUIRE(std::abs(openmc::stellarator_invert(c, u) - t) < 1.e-13);
  }
}
