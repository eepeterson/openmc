#ifndef OPENMC_STELLARATOR_H
#define OPENMC_STELLARATOR_H

#include <cmath>
#include <limits>

#include "openmc/array.h"

namespace openmc {

// Integral on [0,t] of a cubic density, evaluated without the radial bin width.
inline double stellarator_integral(const array<double, 4>& c, double t)
{
  return t * (c[0] + t * (c[1] / 2 + t * (c[2] / 3 + t * c[3] / 4)));
}

// Invert the normalized integral of a nonnegative cubic. Bisection guarantees
// progress even at a zero of the PDF. Stop at a bracket width of 8 epsilon;
// this does not bound rounding error in evaluating an ill-conditioned CDF.
inline double stellarator_invert(const array<double, 4>& c, double u)
{
  if (u <= 0)
    return 0;
  if (u >= 1)
    return 1;
  double target = u * stellarator_integral(c, 1);
  double lo = 0, hi = 1, t = u;
  for (int iteration = 0; iteration < 512; ++iteration) {
    double residual = stellarator_integral(c, t) - target;
    if (residual > 0)
      hi = t;
    else
      lo = t;
    if (hi - lo <= 8 * std::numeric_limits<double>::epsilon())
      break;
    double pdf = c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    double next = pdf > 0 ? t - residual / pdf : (lo + hi) / 2;
    // Keep Newton steps away from the bracket edges to bound iteration count.
    t = next > lo + 0.1 * (hi - lo) && next < hi - 0.1 * (hi - lo)
          ? next
          : (lo + hi) / 2;
  }
  return (lo + hi) / 2;
}

} // namespace openmc
#endif
