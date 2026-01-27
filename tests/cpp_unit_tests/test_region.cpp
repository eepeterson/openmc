#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/bounding_box.h"
#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/surface.h"

#include <pugixml.hpp>

#include <cmath>

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
// Octree Volume Calculation Tests
//==============================================================================

namespace {

// Helper class to set up geometry with sphere and box surfaces for volume tests
class VolumeTestFixture {
public:
  VolumeTestFixture()
  {
    pugi::xml_document doc;

    // Create sphere surface (id=1): x^2 + y^2 + z^2 - 1^2 = 0 (unit sphere)
    {
      pugi::xml_node sphere_node = doc.append_child("surface");
      sphere_node.append_attribute("id") = 1;
      sphere_node.append_attribute("type") = "sphere";
      sphere_node.append_attribute("coeffs") = "0 0 0 1"; // x0, y0, z0, r
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceSphere>(sphere_node));
      openmc::model::surface_map[1] = 0;
    }

    // Create 6 planes to form a box from -0.5 to 0.5 (id=2-7)
    // -x plane at x = -0.5 (id=2)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 2;
      plane_node.append_attribute("type") = "x-plane";
      plane_node.append_attribute("coeffs") = "-0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceXPlane>(plane_node));
      openmc::model::surface_map[2] = 1;
    }
    // +x plane at x = 0.5 (id=3)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 3;
      plane_node.append_attribute("type") = "x-plane";
      plane_node.append_attribute("coeffs") = "0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceXPlane>(plane_node));
      openmc::model::surface_map[3] = 2;
    }
    // -y plane at y = -0.5 (id=4)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 4;
      plane_node.append_attribute("type") = "y-plane";
      plane_node.append_attribute("coeffs") = "-0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceYPlane>(plane_node));
      openmc::model::surface_map[4] = 3;
    }
    // +y plane at y = 0.5 (id=5)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 5;
      plane_node.append_attribute("type") = "y-plane";
      plane_node.append_attribute("coeffs") = "0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceYPlane>(plane_node));
      openmc::model::surface_map[5] = 4;
    }
    // -z plane at z = -0.5 (id=6)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 6;
      plane_node.append_attribute("type") = "z-plane";
      plane_node.append_attribute("coeffs") = "-0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZPlane>(plane_node));
      openmc::model::surface_map[6] = 5;
    }
    // +z plane at z = 0.5 (id=7)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 7;
      plane_node.append_attribute("type") = "z-plane";
      plane_node.append_attribute("coeffs") = "0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZPlane>(plane_node));
      openmc::model::surface_map[7] = 6;
    }

    // Create cylinder along z-axis (id=8): radius 0.3
    {
      pugi::xml_node cyl_node = doc.append_child("surface");
      cyl_node.append_attribute("id") = 8;
      cyl_node.append_attribute("type") = "z-cylinder";
      cyl_node.append_attribute("coeffs") = "0 0 0.3"; // x0, y0, r
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZCylinder>(cyl_node));
      openmc::model::surface_map[8] = 7;
    }

    // Add a smaller sphere for shell tests (id=9): radius 0.5
    {
      pugi::xml_node sphere_node = doc.append_child("surface");
      sphere_node.append_attribute("id") = 9;
      sphere_node.append_attribute("type") = "sphere";
      sphere_node.append_attribute("coeffs") = "0 0 0 0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceSphere>(sphere_node));
      openmc::model::surface_map[9] = 8;
    }

    // Add an offset sphere for union tests (id=10): radius 0.5 at (3, 0, 0)
    {
      pugi::xml_node sphere_node = doc.append_child("surface");
      sphere_node.append_attribute("id") = 10;
      sphere_node.append_attribute("type") = "sphere";
      sphere_node.append_attribute("coeffs") = "3 0 0 0.5";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceSphere>(sphere_node));
      openmc::model::surface_map[10] = 9;
    }

    // Add a z-cone (id=11): apex at origin, R^2 = 1 (45-degree half-angle)
    // Equation: x^2 + y^2 - z^2 = 0
    {
      pugi::xml_node cone_node = doc.append_child("surface");
      cone_node.append_attribute("id") = 11;
      cone_node.append_attribute("type") = "z-cone";
      cone_node.append_attribute("coeffs") = "0 0 0 1"; // x0, y0, z0, R^2
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZCone>(cone_node));
      openmc::model::surface_map[11] = 10;
    }

    // Add z-planes for truncating cone (id=12, 13)
    // z = 0 plane (id=12)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 12;
      plane_node.append_attribute("type") = "z-plane";
      plane_node.append_attribute("coeffs") = "0";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZPlane>(plane_node));
      openmc::model::surface_map[12] = 11;
    }
    // z = 1 plane (id=13)
    {
      pugi::xml_node plane_node = doc.append_child("surface");
      plane_node.append_attribute("id") = 13;
      plane_node.append_attribute("type") = "z-plane";
      plane_node.append_attribute("coeffs") = "1";
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZPlane>(plane_node));
      openmc::model::surface_map[13] = 12;
    }

    // Add a z-torus (id=14): centered at origin
    // Major radius A = 2.0, minor radii B = C = 0.5 (circular cross-section)
    // Equation: z^2/B^2 + (sqrt(x^2+y^2) - A)^2/C^2 - 1 = 0
    {
      pugi::xml_node torus_node = doc.append_child("surface");
      torus_node.append_attribute("id") = 14;
      torus_node.append_attribute("type") = "z-torus";
      torus_node.append_attribute("coeffs") = "0 0 0 2.0 0.5 0.5"; // x0, y0, z0, A, B, C
      openmc::model::surfaces.push_back(
        std::make_unique<openmc::SurfaceZTorus>(torus_node));
      openmc::model::surface_map[14] = 13;
    }
  }

  ~VolumeTestFixture()
  {
    openmc::model::surfaces.clear();
    openmc::model::surface_map.clear();
  }
};

} // anonymous namespace

TEST_CASE("Octree volume calculation")
{
  VolumeTestFixture fixture;

  SECTION("Unit cube volume")
  {
    // Create a unit cube region: 2 -3 4 -5 6 -7
    // which means: x > -0.5 AND x < 0.5 AND y > -0.5 AND y < 0.5 AND z > -0.5 AND z < 0.5
    auto region = openmc::Region("2 -3 4 -5 6 -7", 0);
    openmc::BoundingBox root_box({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});

    // Analytical volume: 1.0
    double analytical_volume = 1.0;

    // Octree should get exact volume for box (all boxes classify cleanly)
    double octree_volume = region.volume_octree(root_box, 10);
    REQUIRE_THAT(octree_volume, Catch::Matchers::WithinRel(analytical_volume, 1e-10));
  }

  SECTION("Unit sphere volume")
  {
    // Create a unit sphere region: -1 (inside surface 1)
    auto region = openmc::Region("-1", 0);
    openmc::BoundingBox root_box({-1.5, -1.5, -1.5}, {1.5, 1.5, 1.5});

    // Analytical volume: (4/3) * pi * r^3 = (4/3) * pi * 1 = 4.18879...
    double analytical_volume = (4.0 / 3.0) * openmc::PI;

    // Octree volume should converge as depth increases
    double vol_d5 = region.volume_octree(root_box, 5);
    double vol_d8 = region.volume_octree(root_box, 8);
    double vol_d10 = region.volume_octree(root_box, 10);

    // Errors should decrease with increasing depth
    double err_d5 = std::abs(vol_d5 - analytical_volume) / analytical_volume;
    double err_d8 = std::abs(vol_d8 - analytical_volume) / analytical_volume;
    double err_d10 = std::abs(vol_d10 - analytical_volume) / analytical_volume;

    REQUIRE(err_d8 < err_d5);
    REQUIRE(err_d10 < err_d8);

    // At depth 10, should be within ~0.5%
    REQUIRE_THAT(vol_d10, Catch::Matchers::WithinRel(analytical_volume, 0.005));
  }

  SECTION("Cube inside sphere - intersection geometry")
  {
    // Create a cube intersected with sphere: (2 -3 4 -5 6 -7) -1
    // This is cube intersection with sphere interior
    // The cube corners are at sqrt(3 * 0.25) = 0.866 from origin, inside unit sphere
    auto region = openmc::Region("(2 -3 4 -5 6 -7) -1", 0);
    openmc::BoundingBox root_box({-1.5, -1.5, -1.5}, {1.5, 1.5, 1.5});

    // The unit cube is fully inside unit sphere, so volume = 1.0
    // But octree may have slight ambiguity at corners where cube is close to sphere
    double analytical_volume = 1.0;

    double octree_volume = region.volume_octree(root_box, 12);
    // Allow 0.5% tolerance since corners approach sphere surface
    REQUIRE_THAT(octree_volume, Catch::Matchers::WithinRel(analytical_volume, 0.005));
  }

  SECTION("Sphere with cylinder removed - difference geometry")
  {
    // Create sphere minus a cylinder: -1 8
    // This is sphere interior AND outside cylinder (carved cylinder)
    auto region = openmc::Region("-1 8", 0);
    openmc::BoundingBox root_box({-1.5, -1.5, -1.5}, {1.5, 1.5, 1.5});

    // Analytical: sphere volume - cylinder volume through sphere
    // For cylinder of radius 0.3 through unit sphere, the cylinder caps
    // are cut by the sphere surface. The exact volume is complex, but we
    // can verify the octree converges
    double sphere_volume = (4.0 / 3.0) * openmc::PI;

    // The cylinder height within the sphere is 2 * sqrt(1 - 0.3^2) = 2 * 0.9539...
    // Cylinder cap volume is complex; let's just verify convergence
    double vol_d6 = region.volume_octree(root_box, 6);
    double vol_d8 = region.volume_octree(root_box, 8);
    double vol_d10 = region.volume_octree(root_box, 10);

    // Volume should be less than sphere
    REQUIRE(vol_d10 < sphere_volume);
    REQUIRE(vol_d10 > 0);

    // Volumes should converge (differences decrease)
    double diff_68 = std::abs(vol_d8 - vol_d6);
    double diff_810 = std::abs(vol_d10 - vol_d8);
    REQUIRE(diff_810 < diff_68);
  }

  SECTION("Spherical shell - complex region")
  {
    // Shell region: -1 9 (inside outer sphere AND outside inner sphere)
    // Surface 1: unit sphere at origin
    // Surface 9: r=0.5 sphere at origin
    auto region = openmc::Region("-1 9", 0);
    openmc::BoundingBox root_box({-1.5, -1.5, -1.5}, {1.5, 1.5, 1.5});

    // Analytical: V = (4/3) * pi * (R^3 - r^3) = (4/3) * pi * (1 - 0.125)
    double analytical_volume = (4.0 / 3.0) * openmc::PI * (1.0 - 0.125);

    double octree_volume = region.volume_octree(root_box, 10);

    // Should be within ~0.5%
    REQUIRE_THAT(octree_volume, Catch::Matchers::WithinRel(analytical_volume, 0.005));
  }

  SECTION("Union of two separate regions")
  {
    // Union of original sphere (r=1 at origin) with small sphere (r=0.5 at x=3)
    // Surface 1: unit sphere at origin
    // Surface 10: r=0.5 sphere at (3, 0, 0)
    auto region = openmc::Region("-1 | -10", 0);
    openmc::BoundingBox root_box({-2.0, -2.0, -2.0}, {4.0, 2.0, 2.0});

    // Analytical: two non-overlapping spheres
    double vol_large = (4.0 / 3.0) * openmc::PI * 1.0;
    double vol_small = (4.0 / 3.0) * openmc::PI * 0.125;
    double analytical_volume = vol_large + vol_small;

    double octree_volume = region.volume_octree(root_box, 10);

    // Should be within ~1% (larger box makes convergence slower)
    REQUIRE_THAT(octree_volume, Catch::Matchers::WithinRel(analytical_volume, 0.01));
  }

  SECTION("Truncated cone volume")
  {
    // Truncated cone (frustum): inside cone surface, between z=0 and z=1
    // Surface 11: z-cone with apex at origin, R^2=1 (45-degree half-angle)
    // Surface 12: z=0 plane
    // Surface 13: z=1 plane
    // Region: inside cone AND z > 0 AND z < 1
    // For the cone surface, negative sense means inside (toward axis)
    auto region = openmc::Region("-11 12 -13", 0);
    openmc::BoundingBox root_box({-1.5, -1.5, -0.5}, {1.5, 1.5, 1.5});

    // Analytical volume of cone frustum from z=0 to z=1:
    // For a 45-degree cone, radius = z, so at z=0 r=0, at z=1 r=1
    // V = (1/3) * pi * h * (r1^2 + r1*r2 + r2^2)
    // With r1=0 (apex), r2=1 (at z=1), h=1:
    // V = (1/3) * pi * 1 * (0 + 0 + 1) = pi/3
    double analytical_volume = openmc::PI / 3.0;

    double octree_volume = region.volume_octree(root_box, 10);

    // Should be within ~1% (cone surface is more challenging)
    REQUIRE_THAT(octree_volume, Catch::Matchers::WithinRel(analytical_volume, 0.01));
  }

  SECTION("Torus volume")
  {
    // Torus: inside torus surface 14
    // Surface 14: z-torus with major radius A=2.0, minor radii B=C=0.5
    // Region: inside torus (negative sense)
    auto region = openmc::Region("-14", 0);
    openmc::BoundingBox root_box({-3.0, -3.0, -1.0}, {3.0, 3.0, 1.0});

    // Analytical volume of torus with circular cross-section:
    // V = 2 * pi^2 * R * r^2
    // Where R = major radius = 2.0, r = minor radius = 0.5
    // V = 2 * pi^2 * 2.0 * 0.25 = pi^2
    double analytical_volume = openmc::PI * openmc::PI;

    double octree_volume = region.volume_octree(root_box, 10);

    // Should be within ~1% (torus is a challenging surface)
    REQUIRE_THAT(octree_volume, Catch::Matchers::WithinRel(analytical_volume, 0.01));
  }

  SECTION("Cone intersected with sphere - convergence test")
  {
    // Cone inside sphere: tests intersection of cone and sphere
    // Region: inside cone (-11) AND inside sphere (-1) AND z > 0 (12)
    // This is a complex geometry - verify convergence rather than exact value
    auto region = openmc::Region("-11 -1 12", 0);
    openmc::BoundingBox root_box({-1.5, -1.5, -0.5}, {1.5, 1.5, 1.5});

    double vol_d6 = region.volume_octree(root_box, 6);
    double vol_d8 = region.volume_octree(root_box, 8);
    double vol_d10 = region.volume_octree(root_box, 10);

    // Volume should be positive and bounded by sphere volume
    double sphere_volume = (4.0 / 3.0) * openmc::PI;
    REQUIRE(vol_d10 > 0);
    REQUIRE(vol_d10 < sphere_volume);

    // Volumes should converge (differences decrease)
    double diff_68 = std::abs(vol_d8 - vol_d6);
    double diff_810 = std::abs(vol_d10 - vol_d8);
    REQUIRE(diff_810 < diff_68);
  }
}
