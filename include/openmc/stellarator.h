#ifndef OPENMC_STELLARATOR_H
#define OPENMC_STELLARATOR_H

#include <algorithm>
#include <cmath>
#include <limits>

#include "openmc/array.h"

namespace openmc {

// Integral on [0,t] of a cubic density, evaluated without the radial bin width.
inline double stellarator_integral(const array<double, 4>& c, double t)
{
  return t * (c[0] + t * (c[1] / 2 + t * (c[2] / 3 + t * c[3] / 4)));
}

// Bernstein basis functions are nonnegative and sum to one on [0,1], so the
// largest Bernstein coefficient bounds the cubic for rejection sampling.
inline double stellarator_rejection_bound(const array<double, 4>& c)
{
  double bound = std::max({c[0], c[0] + c[1] / 3,
    c[0] + 2 * c[1] / 3 + c[2] / 3, c[0] + c[1] + c[2] + c[3]});
  // Allow for rounding in both the bound and the polynomial evaluation.
  double magnitude = 0;
  for (double value : c)
    magnitude += std::abs(value);
  return bound + 64 * std::numeric_limits<double>::epsilon() * magnitude;
}

} // namespace openmc
#endif
