#ifndef OPENMC_BOUNDING_BOX_H
#define OPENMC_BOUNDING_BOX_H

#include <algorithm> // for min, max
#include <tuple>

#include "openmc/constants.h"
#include "openmc/interval.h"
#include "openmc/position.h"

namespace openmc {

//==============================================================================
//! Coordinates for an axis-aligned cuboid that bounds a geometric object.
//==============================================================================

struct BoundingBox {
  Position min = {-INFTY, -INFTY, -INFTY};
  Position max = {INFTY, INFTY, INFTY};

  // Constructors
  BoundingBox() = default;
  BoundingBox(Position min_, Position max_) : min {min_}, max {max_} {}

  // Static factory methods
  static BoundingBox infinite() { return {}; }
  static BoundingBox inverted()
  {
    return {{INFTY, INFTY, INFTY}, {-INFTY, -INFTY, -INFTY}};
  }

  inline BoundingBox operator&(const BoundingBox& other)
  {
    BoundingBox result = *this;
    return result &= other;
  }

  inline BoundingBox operator|(const BoundingBox& other)
  {
    BoundingBox result = *this;
    return result |= other;
  }

  // intersect operator
  inline BoundingBox& operator&=(const BoundingBox& other)
  {
    min.x = std::max(min.x, other.min.x);
    min.y = std::max(min.y, other.min.y);
    min.z = std::max(min.z, other.min.z);
    max.x = std::min(max.x, other.max.x);
    max.y = std::min(max.y, other.max.y);
    max.z = std::min(max.z, other.max.z);
    return *this;
  }

  // union operator
  inline BoundingBox& operator|=(const BoundingBox& other)
  {
    min.x = std::min(min.x, other.min.x);
    min.y = std::min(min.y, other.min.y);
    min.z = std::min(min.z, other.min.z);
    max.x = std::max(max.x, other.max.x);
    max.y = std::max(max.y, other.max.y);
    max.z = std::max(max.z, other.max.z);
    return *this;
  }

  //! Get intervals for all three coordinates at once
  //! \return Tuple of (x, y, z) intervals for use with structured bindings
  std::tuple<Interval, Interval, Interval> intervals() const
  {
    return {{min.x, max.x}, {min.y, max.y}, {min.z, max.z}};
  }

  //! Get interval for x coordinate
  Interval x_interval() const { return {min.x, max.x}; }

  //! Get interval for y coordinate
  Interval y_interval() const { return {min.y, max.y}; }

  //! Get interval for z coordinate
  Interval z_interval() const { return {min.z, max.z}; }

  //! Get the center point of the bounding box
  Position center() const
  {
    return {0.5 * (min.x + max.x), 0.5 * (min.y + max.y), 0.5 * (min.z + max.z)};
  }

  //! Get the volume of the bounding box
  double volume() const
  {
    return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
  }

  //! Get the i-th octant (child box) of this bounding box
  //!
  //! The octants are numbered 0-7 using bit flags:
  //!   bit 0 (1): high-x vs low-x
  //!   bit 1 (2): high-y vs low-y
  //!   bit 2 (4): high-z vs low-z
  //! \param i Octant index in [0, 7]
  //! \return The bounding box for that octant
  BoundingBox octant(int i) const
  {
    Position c = center();
    return {
      {(i & 1) ? c.x : min.x, (i & 2) ? c.y : min.y, (i & 4) ? c.z : min.z},
      {(i & 1) ? max.x : c.x, (i & 2) ? max.y : c.y, (i & 4) ? max.z : c.z}};
  }
};

} // namespace openmc

#endif // OPENMC_BOUNDING_BOX_H
