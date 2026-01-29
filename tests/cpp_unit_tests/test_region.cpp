#include <catch2/catch_test_macros.hpp>

#include "openmc/bounding_box.h"
#include "openmc/cell.h"
#include "openmc/surface.h"

#include <pugixml.hpp>

namespace {

// Helper class to set up and tear down test surfaces
class SurfaceFixture {
public:
  SurfaceFixture()
  {
    pugi::xml_document doc;
    pugi::xml_node surf_node = doc.append_child("surface");
    surf_node.set_name("surface");
    surf_node.append_attribute("id") = "0";
    surf_node.append_attribute("type") = "x-plane";
    surf_node.append_attribute("coeffs") = "1";

    for (int i = 1; i < 10; ++i) {
      surf_node.attribute("id") = i;
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceXPlane>(surf_node));
      openmc::model::surface_map[i] = i - 1;
    }
  }

  ~SurfaceFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

} // anonymous namespace

TEST_CASE("Test region simplification")
{
  SurfaceFixture fixture;

  SECTION("Original bug case from issue #3685")
  {
    // Input: "-1 2 (-3 4) | (-5 6)" was being incorrectly interpreted
    auto region = openmc::Region("(-1 2 (-3 4) | (-5 6))", 0);
    REQUIRE(region.str() == " ( ( -1 2 ( -3 4 ) ) | ( -5 6 ) )");
  }

  SECTION("Simple union - no extra parentheses needed")
  {
    auto region = openmc::Region("1 | 2", 0);
    REQUIRE(region.str() == " 1 | 2");
  }

  SECTION("Intersection then union")
  {
    // Intersection should have higher precedence, so (1 2) grouped
    auto region = openmc::Region("1 2 | 3", 0);
    REQUIRE(region.str() == " ( 1 2 ) | 3");
  }

  SECTION("Union then intersection")
  {
    // The (2 3) intersection should be grouped
    auto region = openmc::Region("1 | 2 3", 0);
    REQUIRE(region.str() == " 1 | ( 2 3 )");
  }

  SECTION("Nested parentheses preserved")
  {
    // These parentheses are meaningful and should be preserved
    auto region = openmc::Region("(1 | 2) (3 | 4)", 0);
    REQUIRE(region.str() == " ( 1 | 2 ) ( 3 | 4 )");
  }

  SECTION("Deep nesting")
  {
    auto region = openmc::Region("((1 2) | (3 4)) 5", 0);
    REQUIRE(region.str() == " ( ( 1 2 ) | ( 3 4 ) ) 5");
  }

  SECTION("Multiple unions")
  {
    auto region = openmc::Region("1 | 2 | 3", 0);
    REQUIRE(region.str() == " 1 | 2 | 3");
  }

  SECTION("Multiple intersections")
  {
    auto region = openmc::Region("1 2 3", 0);
    // Simple cell - no operators in output
    REQUIRE(region.str() == " 1 2 3");
  }

  SECTION("Complex mixed expression")
  {
    auto region = openmc::Region("1 2 | 3 4 | 5 6", 0);
    REQUIRE(region.str() == " ( 1 2 ) | ( 3 4 ) | ( 5 6 )");
  }
}

//==============================================================================
// Box Classification Tests
//==============================================================================

namespace {

// Helper class to set up surfaces for box classification tests
// Creates x-planes at x = 0, 1, 2, 3, ... for surfaces 1, 2, 3, 4, ...
class BoxClassificationFixture {
public:
  BoxClassificationFixture()
  {
    pugi::xml_document doc;
    pugi::xml_node surf_node = doc.append_child("surface");
    surf_node.append_attribute("id") = "1";
    surf_node.append_attribute("type") = "x-plane";
    surf_node.append_attribute("coeffs") = "0";

    // Create planes at x = 0, 1, 2, 3, ...
    for (int i = 1; i <= 10; ++i) {
      surf_node.attribute("id") = i;
      surf_node.attribute("coeffs") = std::to_string(i - 1).c_str();
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceXPlane>(surf_node));
      openmc::model::surface_map[i] = i - 1;
    }
  }

  ~BoxClassificationFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

} // anonymous namespace

TEST_CASE("Test classify_box simple region")
{
  BoxClassificationFixture fixture;

  // Region: x > 0 AND x < 2 (slab between x=0 and x=2)
  // Expression: "1 -3" means surface 1 positive (x > 0) AND surface 3 negative (x < 2)
  auto region = openmc::Region("1 -3", 0);
  REQUIRE(region.is_simple());

  SECTION("Box entirely inside")
  {
    // Box from x=0.5 to x=1.5 is entirely inside [0, 2]
    openmc::BoundingBox box({0.5, -1.0, -1.0}, {1.5, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::INSIDE);
  }

  SECTION("Box entirely outside - left")
  {
    // Box from x=-2 to x=-1 is entirely outside (left of x=0)
    openmc::BoundingBox box({-2.0, -1.0, -1.0}, {-1.0, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::OUTSIDE);
  }

  SECTION("Box entirely outside - right")
  {
    // Box from x=3 to x=4 is entirely outside (right of x=2)
    openmc::BoundingBox box({3.0, -1.0, -1.0}, {4.0, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::OUTSIDE);
  }

  SECTION("Box straddles left boundary")
  {
    // Box from x=-0.5 to x=0.5 straddles x=0
    openmc::BoundingBox box({-0.5, -1.0, -1.0}, {0.5, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::AMBIGUOUS);
  }
}

TEST_CASE("Test classify_box complex region with union")
{
  BoxClassificationFixture fixture;

  // Region: (x > 0 AND x < 1) OR (x > 2 AND x < 3)
  // Two disjoint slabs: [0,1] and [2,3]
  // Expression: "(1 -2) | (3 -4)"
  auto region = openmc::Region("(1 -2) | (3 -4)", 0);
  REQUIRE_FALSE(region.is_simple());

  SECTION("Box in first slab - INSIDE")
  {
    openmc::BoundingBox box({0.25, -1.0, -1.0}, {0.75, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::INSIDE);
  }

  SECTION("Box in second slab - INSIDE")
  {
    openmc::BoundingBox box({2.25, -1.0, -1.0}, {2.75, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::INSIDE);
  }

  SECTION("Box between slabs - OUTSIDE")
  {
    openmc::BoundingBox box({1.25, -1.0, -1.0}, {1.75, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::OUTSIDE);
  }

  SECTION("Box spanning both slabs - AMBIGUOUS")
  {
    // This box contains parts of both slabs and the gap
    openmc::BoundingBox box({0.5, -1.0, -1.0}, {2.5, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::AMBIGUOUS);
  }

  SECTION("Box straddling first slab boundary - AMBIGUOUS | OUTSIDE = AMBIGUOUS")
  {
    // Box straddles the left edge of the first slab
    // Left part is OUTSIDE, right part is INSIDE, so first slab gives AMBIGUOUS
    // Second slab is entirely OUTSIDE
    // AMBIGUOUS | OUTSIDE = AMBIGUOUS
    openmc::BoundingBox box({-0.25, -1.0, -1.0}, {0.25, 1.0, 1.0});
    REQUIRE(region.classify_box(box) == openmc::BoxClassification::AMBIGUOUS);
  }
}

TEST_CASE("Test classify_box union operator correctness")
{
  BoxClassificationFixture fixture;

  // This test specifically exercises the bug that was fixed:
  // When the left operand of a union is AMBIGUOUS, the result should be AMBIGUOUS
  // (unless the right operand is INSIDE).

  // Region: "1 | 2" means (x > 0) OR (x > 1)
  // For box at x = [-0.5, 0.5]:
  //   - Surface 1 (x > 0): AMBIGUOUS
  //   - Surface 2 (x > 1): OUTSIDE
  //   - Result: AMBIGUOUS | OUTSIDE = AMBIGUOUS
  auto region = openmc::Region("1 | 2", 0);

  openmc::BoundingBox box({-0.5, -1.0, -1.0}, {0.5, 1.0, 1.0});
  REQUIRE(region.classify_box(box) == openmc::BoxClassification::AMBIGUOUS);
}

TEST_CASE("Test classify_box intersection operator correctness")
{
  BoxClassificationFixture fixture;

  // Region: "1 2" means (x > 0) AND (x > 1)
  // For box at x = [0.5, 1.5]:
  //   - Surface 1 (x > 0): INSIDE
  //   - Surface 2 (x > 1): AMBIGUOUS
  //   - Result: INSIDE & AMBIGUOUS = AMBIGUOUS
  auto region = openmc::Region("1 2", 0);

  openmc::BoundingBox box({0.5, -1.0, -1.0}, {1.5, 1.0, 1.0});
  REQUIRE(region.classify_box(box) == openmc::BoxClassification::AMBIGUOUS);
}
