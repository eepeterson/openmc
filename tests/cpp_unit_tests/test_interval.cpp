#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/interval.h"

using namespace openmc;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

//==============================================================================
// Constructor tests
//==============================================================================

TEST_CASE("Interval default constructor")
{
  Interval a;
  REQUIRE(a.lo == 0.0);
  REQUIRE(a.hi == 0.0);
}

TEST_CASE("Interval scalar constructor")
{
  Interval a(3.5);
  REQUIRE(a.lo == 3.5);
  REQUIRE(a.hi == 3.5);
}

TEST_CASE("Interval range constructor")
{
  Interval a(1.0, 5.0);
  REQUIRE(a.lo == 1.0);
  REQUIRE(a.hi == 5.0);
}

TEST_CASE("Interval constexpr construction")
{
  // Verify constexpr works at compile time
  constexpr Interval a(2.0, 4.0);
  static_assert(a.lo == 2.0, "constexpr lo failed");
  static_assert(a.hi == 4.0, "constexpr hi failed");
  REQUIRE(a.lo == 2.0);
}

//==============================================================================
// Member function tests
//==============================================================================

TEST_CASE("Interval classification methods")
{
  SECTION("Positive interval")
  {
    Interval a(1.0, 5.0);
    REQUIRE(a.is_positive());
    REQUIRE_FALSE(a.is_negative());
    REQUIRE_FALSE(a.contains_zero());
  }

  SECTION("Negative interval")
  {
    Interval a(-5.0, -1.0);
    REQUIRE_FALSE(a.is_positive());
    REQUIRE(a.is_negative());
    REQUIRE_FALSE(a.contains_zero());
  }

  SECTION("Interval spanning zero")
  {
    Interval a(-2.0, 3.0);
    REQUIRE_FALSE(a.is_positive());
    REQUIRE_FALSE(a.is_negative());
    REQUIRE(a.contains_zero());
  }

  SECTION("Interval touching zero from above")
  {
    Interval a(0.0, 5.0);
    REQUIRE_FALSE(a.is_positive());
    REQUIRE_FALSE(a.is_negative());
    REQUIRE(a.contains_zero());
  }

  SECTION("Interval touching zero from below")
  {
    Interval a(-5.0, 0.0);
    REQUIRE_FALSE(a.is_positive());
    REQUIRE_FALSE(a.is_negative());
    REQUIRE(a.contains_zero());
  }

  SECTION("Point interval at zero")
  {
    Interval a(0.0);
    REQUIRE_FALSE(a.is_positive());
    REQUIRE_FALSE(a.is_negative());
    REQUIRE(a.contains_zero());
  }
}

TEST_CASE("Interval width and midpoint")
{
  SECTION("Standard interval")
  {
    Interval a(2.0, 8.0);
    REQUIRE(a.width() == 6.0);
    REQUIRE(a.midpoint() == 5.0);
  }

  SECTION("Point interval")
  {
    Interval a(3.0);
    REQUIRE(a.width() == 0.0);
    REQUIRE(a.midpoint() == 3.0);
  }

  SECTION("Symmetric interval")
  {
    Interval a(-4.0, 4.0);
    REQUIRE(a.width() == 8.0);
    REQUIRE(a.midpoint() == 0.0);
  }

  SECTION("Negative interval")
  {
    Interval a(-10.0, -2.0);
    REQUIRE(a.width() == 8.0);
    REQUIRE(a.midpoint() == -6.0);
  }
}

//==============================================================================
// Arithmetic operator tests
//==============================================================================

TEST_CASE("Interval addition")
{
  Interval a(1.0, 3.0);
  Interval b(2.0, 5.0);
  Interval c = a + b;
  REQUIRE(c.lo == 3.0);
  REQUIRE(c.hi == 8.0);
}

TEST_CASE("Interval subtraction")
{
  Interval a(5.0, 10.0);
  Interval b(1.0, 3.0);
  Interval c = a - b;
  // [5,10] - [1,3] = [5-3, 10-1] = [2, 9]
  REQUIRE(c.lo == 2.0);
  REQUIRE(c.hi == 9.0);
}

TEST_CASE("Interval multiplication")
{
  SECTION("Both positive")
  {
    Interval a(2.0, 3.0);
    Interval b(4.0, 5.0);
    Interval c = a * b;
    REQUIRE(c.lo == 8.0);
    REQUIRE(c.hi == 15.0);
  }

  SECTION("Both negative")
  {
    Interval a(-3.0, -2.0);
    Interval b(-5.0, -4.0);
    Interval c = a * b;
    REQUIRE(c.lo == 8.0);
    REQUIRE(c.hi == 15.0);
  }

  SECTION("Mixed signs")
  {
    Interval a(-2.0, 3.0);
    Interval b(-1.0, 4.0);
    Interval c = a * b;
    // Products: 2, -8, -3, 12 -> min=-8, max=12
    REQUIRE(c.lo == -8.0);
    REQUIRE(c.hi == 12.0);
  }
}

TEST_CASE("Interval scalar operations")
{
  Interval a(2.0, 4.0);

  SECTION("Addition with scalar")
  {
    Interval b = a + 3.0;
    REQUIRE(b.lo == 5.0);
    REQUIRE(b.hi == 7.0);

    Interval c = 3.0 + a;
    REQUIRE(c.lo == 5.0);
    REQUIRE(c.hi == 7.0);
  }

  SECTION("Subtraction with scalar")
  {
    Interval b = a - 1.0;
    REQUIRE(b.lo == 1.0);
    REQUIRE(b.hi == 3.0);

    Interval c = 10.0 - a;
    // 10 - [2,4] = [10-4, 10-2] = [6, 8]
    REQUIRE(c.lo == 6.0);
    REQUIRE(c.hi == 8.0);
  }

  SECTION("Multiplication with positive scalar")
  {
    Interval b = a * 2.0;
    REQUIRE(b.lo == 4.0);
    REQUIRE(b.hi == 8.0);

    Interval c = 2.0 * a;
    REQUIRE(c.lo == 4.0);
    REQUIRE(c.hi == 8.0);
  }

  SECTION("Multiplication with negative scalar")
  {
    Interval b = a * (-2.0);
    // [2,4] * -2 = [-8, -4]
    REQUIRE(b.lo == -8.0);
    REQUIRE(b.hi == -4.0);
  }

  SECTION("Division by scalar")
  {
    Interval b = a / 2.0;
    REQUIRE(b.lo == 1.0);
    REQUIRE(b.hi == 2.0);
  }
}

TEST_CASE("Interval unary negation")
{
  Interval a(2.0, 5.0);
  Interval b = -a;
  REQUIRE(b.lo == -5.0);
  REQUIRE(b.hi == -2.0);
}

TEST_CASE("Interval compound assignment operators")
{
  SECTION("operator+=")
  {
    Interval a(1.0, 3.0);
    a += Interval(2.0, 4.0);
    REQUIRE(a.lo == 3.0);
    REQUIRE(a.hi == 7.0);
  }

  SECTION("operator-=")
  {
    Interval a(5.0, 10.0);
    a -= Interval(1.0, 2.0);
    REQUIRE(a.lo == 3.0);
    REQUIRE(a.hi == 9.0);
  }

  SECTION("operator*= with positive")
  {
    Interval a(2.0, 4.0);
    a *= 3.0;
    REQUIRE(a.lo == 6.0);
    REQUIRE(a.hi == 12.0);
  }

  SECTION("operator*= with negative")
  {
    Interval a(2.0, 4.0);
    a *= -3.0;
    REQUIRE(a.lo == -12.0);
    REQUIRE(a.hi == -6.0);
  }

  SECTION("operator/=")
  {
    Interval a(4.0, 8.0);
    a /= 2.0;
    REQUIRE(a.lo == 2.0);
    REQUIRE(a.hi == 4.0);
  }
}

//==============================================================================
// Mathematical function tests
//==============================================================================

TEST_CASE("Interval interval_sqr function")
{
  SECTION("Positive interval")
  {
    Interval a(2.0, 3.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 4.0);
    REQUIRE(b.hi == 9.0);
  }

  SECTION("Negative interval")
  {
    Interval a(-3.0, -2.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 4.0);
    REQUIRE(b.hi == 9.0);
  }

  SECTION("Interval spanning zero - symmetric")
  {
    Interval a(-2.0, 2.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 0.0);
    REQUIRE(b.hi == 4.0);
  }

  SECTION("Interval spanning zero - asymmetric positive dominant")
  {
    Interval a(-2.0, 5.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 0.0);
    REQUIRE(b.hi == 25.0);
  }

  SECTION("Interval spanning zero - asymmetric negative dominant")
  {
    Interval a(-5.0, 2.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 0.0);
    REQUIRE(b.hi == 25.0);
  }

  SECTION("Point interval")
  {
    Interval a(3.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 9.0);
    REQUIRE(b.hi == 9.0);
  }

  SECTION("Interval touching zero")
  {
    Interval a(0.0, 3.0);
    Interval b = interval_sqr(a);
    REQUIRE(b.lo == 0.0);
    REQUIRE(b.hi == 9.0);
  }
}

TEST_CASE("Interval interval_sqrt function")
{
  SECTION("Standard positive interval")
  {
    Interval a(4.0, 9.0);
    Interval b = interval_sqrt(a);
    REQUIRE(b.lo == 2.0);
    REQUIRE(b.hi == 3.0);
  }

  SECTION("Interval starting at zero")
  {
    Interval a(0.0, 16.0);
    Interval b = interval_sqrt(a);
    REQUIRE(b.lo == 0.0);
    REQUIRE(b.hi == 4.0);
  }

  SECTION("Point interval")
  {
    Interval a(25.0);
    Interval b = interval_sqrt(a);
    REQUIRE(b.lo == 5.0);
    REQUIRE(b.hi == 5.0);
  }

  SECTION("Interval with negative clipped to zero")
  {
    // Negative values should be clamped to zero
    Interval a(-1.0, 4.0);
    Interval b = interval_sqrt(a);
    REQUIRE(b.lo == 0.0);
    REQUIRE(b.hi == 2.0);
  }
}

TEST_CASE("Interval interval_hypot 2D function")
{
  SECTION("First quadrant - both positive")
  {
    Interval x(3.0, 4.0);
    Interval y(4.0, 5.0);
    Interval r = interval_hypot(x, y);
    REQUIRE_THAT(r.lo, WithinRel(5.0, 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(41.0), 1e-10));
  }

  SECTION("Box containing origin")
  {
    Interval x(-1.0, 1.0);
    Interval y(-1.0, 1.0);
    Interval r = interval_hypot(x, y);
    REQUIRE(r.lo == 0.0);
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(2.0), 1e-10));
  }

  SECTION("Third quadrant - both negative")
  {
    Interval x(-4.0, -3.0);
    Interval y(-5.0, -4.0);
    Interval r = interval_hypot(x, y);
    REQUIRE_THAT(r.lo, WithinRel(5.0, 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(41.0), 1e-10));
  }

  SECTION("X spans zero, Y positive")
  {
    Interval x(-2.0, 3.0);
    Interval y(4.0, 5.0);
    Interval r = interval_hypot(x, y);
    // Nearest point: (0, 4) -> distance 4
    // Farthest point: (3, 5) or (-2, 5) -> sqrt(34)
    REQUIRE_THAT(r.lo, WithinRel(4.0, 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(34.0), 1e-10));
  }

  SECTION("Point intervals")
  {
    Interval x(3.0);
    Interval y(4.0);
    Interval r = interval_hypot(x, y);
    REQUIRE_THAT(r.lo, WithinRel(5.0, 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(5.0, 1e-10));
  }
}

TEST_CASE("Interval interval_hypot 3D function")
{
  SECTION("All positive")
  {
    Interval x(1.0, 2.0);
    Interval y(2.0, 3.0);
    Interval z(2.0, 4.0);
    Interval r = interval_hypot(x, y, z);
    REQUIRE_THAT(r.lo, WithinRel(3.0, 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(29.0), 1e-10));
  }

  SECTION("Box containing origin")
  {
    Interval x(-1.0, 1.0);
    Interval y(-1.0, 1.0);
    Interval z(-1.0, 1.0);
    Interval r = interval_hypot(x, y, z);
    REQUIRE(r.lo == 0.0);
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(3.0), 1e-10));
  }

  SECTION("Mixed signs")
  {
    Interval x(-3.0, -1.0);
    Interval y(2.0, 4.0);
    Interval z(-2.0, 2.0);
    Interval r = interval_hypot(x, y, z);
    // Nearest: (-1, 2, 0) -> sqrt(5)
    // Farthest: (-3, 4, 2) or (-3, 4, -2) -> sqrt(29)
    REQUIRE_THAT(r.lo, WithinRel(std::sqrt(5.0), 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(std::sqrt(29.0), 1e-10));
  }

  SECTION("Point intervals")
  {
    Interval x(2.0);
    Interval y(3.0);
    Interval z(6.0);
    Interval r = interval_hypot(x, y, z);
    REQUIRE_THAT(r.lo, WithinRel(7.0, 1e-10));
    REQUIRE_THAT(r.hi, WithinRel(7.0, 1e-10));
  }
}

//==============================================================================
// Surface equation simulation tests
//==============================================================================

TEST_CASE("Interval arithmetic simulates cylinder equation")
{
  // Cylinder: (x - x0)^2 + (y - y0)^2 - R^2 = 0
  // Centered at (1, 2) with radius 3
  double x0 = 1.0, y0 = 2.0, R = 3.0;

  SECTION("Box entirely inside cylinder")
  {
    Interval x(0.5, 1.5);
    Interval y(1.5, 2.5);
    Interval result = interval_sqr(x - x0) + interval_sqr(y - y0) - R * R;
    REQUIRE(result.is_negative());
  }

  SECTION("Box entirely outside cylinder")
  {
    Interval x(10.0, 12.0);
    Interval y(10.0, 12.0);
    Interval result = interval_sqr(x - x0) + interval_sqr(y - y0) - R * R;
    REQUIRE(result.is_positive());
  }

  SECTION("Box straddling cylinder surface")
  {
    Interval x(3.5, 4.5); // Near the +x edge of cylinder
    Interval y(1.5, 2.5);
    Interval result = interval_sqr(x - x0) + interval_sqr(y - y0) - R * R;
    REQUIRE(result.contains_zero());
  }
}

TEST_CASE("Interval arithmetic simulates sphere equation")
{
  // Sphere: (x - x0)^2 + (y - y0)^2 + (z - z0)^2 - R^2 = 0
  double x0 = 0.0, y0 = 0.0, z0 = 0.0, R = 5.0;

  SECTION("Box at origin - inside")
  {
    Interval x(-1.0, 1.0);
    Interval y(-1.0, 1.0);
    Interval z(-1.0, 1.0);
    Interval result = interval_sqr(x - x0) + interval_sqr(y - y0) +
                      interval_sqr(z - z0) - R * R;
    REQUIRE(result.is_negative());
  }

  SECTION("Box far outside")
  {
    Interval x(10.0, 12.0);
    Interval y(10.0, 12.0);
    Interval z(10.0, 12.0);
    Interval result = interval_sqr(x - x0) + interval_sqr(y - y0) +
                      interval_sqr(z - z0) - R * R;
    REQUIRE(result.is_positive());
  }
}

TEST_CASE("Interval arithmetic simulates cone equation")
{
  // Z-cone: x^2 + y^2 - k^2 * z^2 = 0 (apex at origin, opening upward)
  double k_sq = 0.25; // 45-degree half-angle -> k = 0.5

  SECTION("Point on cone surface")
  {
    // At z=2, radius should be 1 (since r = k*z = 0.5*2 = 1)
    Interval x(0.9, 1.1);
    Interval y(-0.1, 0.1);
    Interval z(1.9, 2.1);
    Interval result = interval_sqr(x) + interval_sqr(y) - k_sq * interval_sqr(z);
    REQUIRE(result.contains_zero());
  }

  SECTION("Inside cone")
  {
    // At z=4, allowed radius is 2. Point at x=0.5, y=0 is inside
    Interval x(0.4, 0.6);
    Interval y(-0.1, 0.1);
    Interval z(3.9, 4.1);
    Interval result = interval_sqr(x) + interval_sqr(y) - k_sq * interval_sqr(z);
    REQUIRE(result.is_negative());
  }
}

TEST_CASE("Interval arithmetic simulates torus radial term")
{
  // Torus uses sqrt(y^2 + z^2) for the radial distance
  // Test that interval_hypot works correctly for this
  SECTION("Ring away from axis")
  {
    Interval y(4.0, 6.0);
    Interval z(-1.0, 1.0);
    Interval rho = interval_hypot(y, z);
    // Nearest point to axis: (4, 0) -> 4
    // Farthest: (6, 1) or (6, -1) -> sqrt(37)
    REQUIRE_THAT(rho.lo, WithinRel(4.0, 1e-10));
    REQUIRE_THAT(rho.hi, WithinRel(std::sqrt(37.0), 1e-10));
  }

  SECTION("Full torus equation: (rho - R)^2 / a^2 + x^2 / b^2 - 1")
  {
    // Major radius R=5, minor radii a=1, b=1
    double R = 5.0, a = 1.0, b = 1.0;
    Interval x(-0.5, 0.5);
    Interval y(4.0, 6.0);
    Interval z(-0.5, 0.5);

    Interval rho = interval_hypot(y, z);
    Interval result =
      interval_sqr(rho - R) / (a * a) + interval_sqr(x) / (b * b) - 1.0;
    // This box should straddle the torus surface
    REQUIRE(result.contains_zero());
  }
}

TEST_CASE("Interval arithmetic simulates quadric cross terms")
{
  // General quadric with cross terms:
  // Ax^2 + By^2 + Cz^2 + Dxy + Eyz + Fzx + Gx + Hy + Jz + K = 0
  double A = 1.0, B = 1.0, C = 1.0;
  double D = 0.5, E = 0.0, F = 0.0; // Only xy cross term
  double G = 0.0, H = 0.0, J = 0.0, K = -1.0;

  Interval x(0.0, 1.0);
  Interval y(0.0, 1.0);
  Interval z(0.0, 0.5);

  // Evaluate: x^2 + y^2 + z^2 + 0.5*x*y - 1
  Interval result = A * interval_sqr(x) + B * interval_sqr(y) +
                    C * interval_sqr(z) + D * x * y + E * y * z + F * z * x +
                    G * x + H * y + J * z + K;
  // Box spans from inside to outside the surface
  REQUIRE(result.contains_zero());
}

TEST_CASE("Interval arithmetic simulates plane equation")
{
  // Plane: Ax + By + Cz - D = 0
  double A = 1.0, B = 0.0, C = 0.0, D = 5.0; // Plane at x = 5

  SECTION("Box entirely below plane")
  {
    Interval x(0.0, 3.0);
    Interval y(-10.0, 10.0);
    Interval z(-10.0, 10.0);
    Interval result = A * x + B * y + C * z - D;
    REQUIRE(result.is_negative());
  }

  SECTION("Box entirely above plane")
  {
    Interval x(7.0, 10.0);
    Interval y(-10.0, 10.0);
    Interval z(-10.0, 10.0);
    Interval result = A * x + B * y + C * z - D;
    REQUIRE(result.is_positive());
  }

  SECTION("Box straddling plane")
  {
    Interval x(4.0, 6.0);
    Interval y(-10.0, 10.0);
    Interval z(-10.0, 10.0);
    Interval result = A * x + B * y + C * z - D;
    REQUIRE(result.contains_zero());
  }
}

//==============================================================================
// Edge case and robustness tests
//==============================================================================

TEST_CASE("Interval arithmetic with zero-width intervals")
{
  Interval a(3.0);
  Interval b(4.0);

  Interval sum = a + b;
  REQUIRE(sum.lo == 7.0);
  REQUIRE(sum.hi == 7.0);

  Interval prod = a * b;
  REQUIRE(prod.lo == 12.0);
  REQUIRE(prod.hi == 12.0);
}

TEST_CASE("Interval arithmetic with very small intervals")
{
  double eps = 1e-15;
  Interval a(1.0 - eps, 1.0 + eps);
  Interval b(2.0 - eps, 2.0 + eps);

  Interval sum = a + b;
  REQUIRE_THAT(sum.lo, WithinAbs(3.0, 3 * eps));
  REQUIRE_THAT(sum.hi, WithinAbs(3.0, 3 * eps));
}

TEST_CASE("Interval interval_sqr at boundary of sign change")
{
  // Interval just barely spanning zero
  Interval a(-1e-10, 1e-10);
  Interval b = interval_sqr(a);
  REQUIRE(b.lo == 0.0);
  REQUIRE_THAT(b.hi, WithinAbs(1e-20, 1e-25));
}

TEST_CASE("Interval interval_hypot with one degenerate dimension")
{
  // Thin slab in y direction
  Interval x(3.0, 4.0);
  Interval y(4.0, 4.0); // Zero width
  Interval r = interval_hypot(x, y);
  REQUIRE_THAT(r.lo, WithinRel(5.0, 1e-10));
  REQUIRE_THAT(r.hi, WithinRel(std::sqrt(32.0), 1e-10));
}

TEST_CASE("Chained interval operations")
{
  // Test that chaining works correctly: ((a + b) * c) - d
  Interval a(1.0, 2.0);
  Interval b(2.0, 3.0);
  Interval c(2.0, 2.0); // scalar
  Interval d(1.0, 2.0);

  Interval result = (a + b) * c - d;
  // a + b = [3, 5]
  // * 2 = [6, 10]
  // - [1, 2] = [6-2, 10-1] = [4, 9]
  REQUIRE(result.lo == 4.0);
  REQUIRE(result.hi == 9.0);
}

TEST_CASE("Interval constexpr operations")
{
  // Verify constexpr arithmetic works
  constexpr Interval a(1.0, 2.0);
  constexpr Interval b(3.0, 4.0);
  constexpr Interval sum = a + b;
  static_assert(sum.lo == 4.0, "constexpr addition lo failed");
  static_assert(sum.hi == 6.0, "constexpr addition hi failed");

  constexpr Interval sq = interval_sqr(a);
  static_assert(sq.lo == 1.0, "constexpr interval_sqr lo failed");
  static_assert(sq.hi == 4.0, "constexpr interval_sqr hi failed");

  REQUIRE(sum.lo == 4.0);
  REQUIRE(sq.hi == 4.0);
}