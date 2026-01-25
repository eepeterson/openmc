#ifndef OPENMC_INTERVAL_H
#define OPENMC_INTERVAL_H

#include <algorithm> // for min, max
#include <cmath>     // for sqrt, abs

namespace openmc {

//==============================================================================
//! A closed interval [lo, hi] for interval arithmetic.
//!
//! Used for computing conservative bounds on function evaluations over
//! axis-aligned boxes, enabling tri-state classification of spatial regions.
//==============================================================================

struct Interval {
  double lo;
  double hi;

  // Constructors
  constexpr Interval() : lo{0.0}, hi{0.0} {}
  constexpr Interval(double val) : lo{val}, hi{val} {}
  constexpr Interval(double lo_, double hi_) : lo{lo_}, hi{hi_} {}

  //! Check if interval is entirely positive
  constexpr bool is_positive() const { return lo > 0.0; }

  //! Check if interval is entirely negative
  constexpr bool is_negative() const { return hi < 0.0; }

  //! Check if interval contains zero
  constexpr bool contains_zero() const { return lo <= 0.0 && hi >= 0.0; }

  //! Width of the interval
  constexpr double width() const { return hi - lo; }

  //! Midpoint of the interval
  constexpr double midpoint() const { return 0.5 * (lo + hi); }

  // Compound assignment operators
  Interval& operator+=(const Interval& other)
  {
    lo += other.lo;
    hi += other.hi;
    return *this;
  }

  Interval& operator-=(const Interval& other)
  {
    lo -= other.hi;
    hi -= other.lo;
    return *this;
  }

  Interval& operator*=(double val)
  {
    if (val >= 0.0) {
      lo *= val;
      hi *= val;
    } else {
      double tmp = lo * val;
      lo = hi * val;
      hi = tmp;
    }
    return *this;
  }

  Interval& operator/=(double val) { return *this *= (1.0 / val); }

  // Unary negation
  Interval operator-() const { return {-hi, -lo}; }
};

//==============================================================================
// Binary operators (Interval, Interval)
//==============================================================================

inline Interval operator+(Interval a, const Interval& b)
{
  return a += b;
}

inline Interval operator-(Interval a, const Interval& b)
{
  return a -= b;
}

inline Interval operator*(const Interval& a, const Interval& b)
{
  double p1 = a.lo * b.lo;
  double p2 = a.lo * b.hi;
  double p3 = a.hi * b.lo;
  double p4 = a.hi * b.hi;
  return {std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4})};
}

//==============================================================================
// Binary operators (Interval, scalar) and (scalar, Interval)
//==============================================================================

inline Interval operator+(Interval a, double val)
{
  a.lo += val;
  a.hi += val;
  return a;
}

inline Interval operator+(double val, Interval a)
{
  return a + val;
}

inline Interval operator-(Interval a, double val)
{
  a.lo -= val;
  a.hi -= val;
  return a;
}

inline Interval operator-(double val, const Interval& a)
{
  return {val - a.hi, val - a.lo};
}

inline Interval operator*(Interval a, double val)
{
  return a *= val;
}

inline Interval operator*(double val, Interval a)
{
  return a *= val;
}

inline Interval operator/(Interval a, double val)
{
  return a /= val;
}

//==============================================================================
// Mathematical functions
//==============================================================================

//! Square of an interval, correctly handling sign changes
//!
//! For x in [lo, hi]:
//!   - If lo >= 0: x^2 in [lo^2, hi^2]
//!   - If hi <= 0: x^2 in [hi^2, lo^2]
//!   - If lo < 0 < hi: x^2 in [0, max(lo^2, hi^2)]
inline Interval interval_sqr(const Interval& x)
{
  if (x.lo >= 0.0) {
    return {x.lo * x.lo, x.hi * x.hi};
  } else if (x.hi <= 0.0) {
    return {x.hi * x.hi, x.lo * x.lo};
  } else {
    double max_sq = std::max(x.lo * x.lo, x.hi * x.hi);
    return {0.0, max_sq};
  }
}

//! Square root of an interval (assumes non-negative input)
inline Interval interval_sqrt(const Interval& x)
{
  double lo = x.lo > 0.0 ? std::sqrt(x.lo) : 0.0;
  double hi = x.hi > 0.0 ? std::sqrt(x.hi) : 0.0;
  return {lo, hi};
}

//! Euclidean norm sqrt(x^2 + y^2) with tight bounds
//!
//! Computes exact bounds on the 2D Euclidean distance over a rectangular
//! region, avoiding the overestimation that would occur from naive interval
//! composition interval_sqrt(interval_sqr(x) + interval_sqr(y)).
inline Interval interval_hypot(const Interval& x, const Interval& y)
{
  // Minimum distance: find closest point to origin in the rectangle
  double x_near, y_near;
  if (x.lo > 0.0) {
    x_near = x.lo;
  } else if (x.hi < 0.0) {
    x_near = x.hi;
  } else {
    x_near = 0.0;
  }

  if (y.lo > 0.0) {
    y_near = y.lo;
  } else if (y.hi < 0.0) {
    y_near = y.hi;
  } else {
    y_near = 0.0;
  }

  double min_val = std::sqrt(x_near * x_near + y_near * y_near);

  // Maximum distance: farthest corner from origin
  double x_far = std::max(std::abs(x.lo), std::abs(x.hi));
  double y_far = std::max(std::abs(y.lo), std::abs(y.hi));
  double max_val = std::sqrt(x_far * x_far + y_far * y_far);

  return {min_val, max_val};
}

//! Euclidean norm sqrt(x^2 + y^2 + z^2) with tight bounds
inline Interval interval_hypot(
  const Interval& x, const Interval& y, const Interval& z)
{
  // Helper to find coordinate nearest to zero
  auto nearest = [](const Interval& i) -> double {
    if (i.lo > 0.0)
      return i.lo;
    if (i.hi < 0.0)
      return i.hi;
    return 0.0;
  };

  // Minimum distance: closest point to origin in the box
  double x_near = nearest(x);
  double y_near = nearest(y);
  double z_near = nearest(z);
  double min_val =
    std::sqrt(x_near * x_near + y_near * y_near + z_near * z_near);

  // Maximum distance: farthest corner from origin
  double x_far = std::max(std::abs(x.lo), std::abs(x.hi));
  double y_far = std::max(std::abs(y.lo), std::abs(y.hi));
  double z_far = std::max(std::abs(z.lo), std::abs(z.hi));
  double max_val = std::sqrt(x_far * x_far + y_far * y_far + z_far * z_far);

  return {min_val, max_val};
}

} // namespace openmc

#endif // OPENMC_INTERVAL_H