#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/sparse_matrix.h"

using namespace openmc;
using Catch::Matchers::WithinAbs;

// Helper: build a simple 3x3 matrix
//   [1  0  2]
//   [0  3  0]
//   [4  0  5]
// In CSC (column-major):
//   col 0: rows 0,2 vals 1,4
//   col 1: row 1    val 3
//   col 2: rows 0,2 vals 2,5
static CSCMatrix make_A()
{
  return CSCMatrix::from_triplets(3, {0, 1, 2, 0, 2}, {0, 1, 0, 2, 2},
    {1.0, 3.0, 4.0, 2.0, 5.0});
}

// Helper: build a 3x3 matrix with different pattern
//   [0  6  0]
//   [7  0  0]
//   [0  0  8]
static CSCMatrix make_B()
{
  return CSCMatrix::from_triplets(3, {1, 0, 2}, {0, 1, 2}, {7.0, 6.0, 8.0});
}

TEST_CASE("CSCMatrix scalar multiply")
{
  auto A = make_A();

  auto C = 2.0 * A;
  REQUIRE(C.n() == 3);
  REQUIRE(C.nnz() == 5);

  // Check values: should be doubled
  const auto& d = C.data();
  CHECK_THAT(d[0], WithinAbs(2.0, 1e-15));  // A(0,0) * 2
  CHECK_THAT(d[1], WithinAbs(8.0, 1e-15));  // A(2,0) * 2
  CHECK_THAT(d[2], WithinAbs(6.0, 1e-15));  // A(1,1) * 2
  CHECK_THAT(d[3], WithinAbs(4.0, 1e-15));  // A(0,2) * 2
  CHECK_THAT(d[4], WithinAbs(10.0, 1e-15)); // A(2,2) * 2

  // Pattern must be identical
  CHECK(C.pattern() == A.pattern());
}

TEST_CASE("CSCMatrix scalar multiply by zero")
{
  auto A = make_A();
  auto C = 0.0 * A;
  REQUIRE(C.nnz() == 5); // structural zeros remain
  for (int k = 0; k < C.nnz(); ++k) {
    CHECK_THAT(C.data()[k], WithinAbs(0.0, 1e-15));
  }
}

TEST_CASE("CSCMatrix scale in-place")
{
  auto A = make_A();
  A.scale(3.0);
  const auto& d = A.data();
  CHECK_THAT(d[0], WithinAbs(3.0, 1e-15));
  CHECK_THAT(d[1], WithinAbs(12.0, 1e-15));
  CHECK_THAT(d[2], WithinAbs(9.0, 1e-15));
  CHECK_THAT(d[3], WithinAbs(6.0, 1e-15));
  CHECK_THAT(d[4], WithinAbs(15.0, 1e-15));
}

TEST_CASE("CSCMatrix operator+= same pattern")
{
  auto A = make_A();
  auto A2 = make_A();
  A += A2;

  // Same pattern: fast path, values doubled
  CHECK(A.nnz() == 5);
  const auto& d = A.data();
  CHECK_THAT(d[0], WithinAbs(2.0, 1e-15));
  CHECK_THAT(d[1], WithinAbs(8.0, 1e-15));
  CHECK_THAT(d[2], WithinAbs(6.0, 1e-15));
  CHECK_THAT(d[3], WithinAbs(4.0, 1e-15));
  CHECK_THAT(d[4], WithinAbs(10.0, 1e-15));
}

TEST_CASE("CSCMatrix operator+= different pattern")
{
  auto A = make_A();
  auto B = make_B();
  A += B;

  // Union of patterns: nnz should be 5 + 3 - 1 (overlap at (2,2)) = 7
  CHECK(A.n() == 3);
  CHECK(A.nnz() == 7);

  // Verify combined values by checking known entries
  // Reconstruct into dense for checking
  const auto& indptr = A.indptr();
  const auto& indices = A.indices();
  const auto& data = A.data();

  double dense[3][3] = {};
  for (int col = 0; col < 3; ++col) {
    for (int k = indptr[col]; k < indptr[col + 1]; ++k) {
      dense[indices[k]][col] = data[k];
    }
  }

  // A + B:
  // [1  6  2]
  // [7  3  0]
  // [4  0  13]
  CHECK_THAT(dense[0][0], WithinAbs(1.0, 1e-15));
  CHECK_THAT(dense[0][1], WithinAbs(6.0, 1e-15));
  CHECK_THAT(dense[0][2], WithinAbs(2.0, 1e-15));
  CHECK_THAT(dense[1][0], WithinAbs(7.0, 1e-15));
  CHECK_THAT(dense[1][1], WithinAbs(3.0, 1e-15));
  CHECK_THAT(dense[1][2], WithinAbs(0.0, 1e-15));
  CHECK_THAT(dense[2][0], WithinAbs(4.0, 1e-15));
  CHECK_THAT(dense[2][1], WithinAbs(0.0, 1e-15));
  CHECK_THAT(dense[2][2], WithinAbs(13.0, 1e-15));
}

TEST_CASE("CSCMatrix combined: A_decay + s * A_rxn pattern")
{
  // This is the actual use case: A = A_decay + s * A_rxn
  auto A_decay = make_A();
  auto A_rxn = make_B();
  double s = 1.5;

  auto A = A_decay + s * A_rxn;
  CHECK(A.n() == 3);

  // Reconstruct dense
  const auto& indptr = A.indptr();
  const auto& indices = A.indices();
  const auto& data = A.data();

  double dense[3][3] = {};
  for (int col = 0; col < 3; ++col) {
    for (int k = indptr[col]; k < indptr[col + 1]; ++k) {
      dense[indices[k]][col] = data[k];
    }
  }

  // A_decay + 1.5 * A_rxn:
  // [1+0    0+9    2+0  ]     [1    9    2  ]
  // [0+10.5 3+0    0+0  ]  =  [10.5 3    0  ]
  // [4+0    0+0    5+12 ]     [4    0    17  ]
  CHECK_THAT(dense[0][0], WithinAbs(1.0, 1e-15));
  CHECK_THAT(dense[0][1], WithinAbs(9.0, 1e-15));
  CHECK_THAT(dense[0][2], WithinAbs(2.0, 1e-15));
  CHECK_THAT(dense[1][0], WithinAbs(10.5, 1e-15));
  CHECK_THAT(dense[1][1], WithinAbs(3.0, 1e-15));
  CHECK_THAT(dense[2][0], WithinAbs(4.0, 1e-15));
  CHECK_THAT(dense[2][2], WithinAbs(17.0, 1e-15));
}

TEST_CASE("CSCMatrix empty matrix arithmetic")
{
  CSCMatrix empty;
  CHECK(empty.n() == 0);
  CHECK(empty.nnz() == 0);

  // scale empty
  empty.scale(5.0);
  CHECK(empty.nnz() == 0);

  // scalar * empty
  auto C = 3.0 * empty;
  CHECK(C.nnz() == 0);
}

TEST_CASE("CSCPattern construction and accessors")
{
  // 3x3 pattern:
  //   col 0: rows 0, 2
  //   col 1: row 1
  //   col 2: rows 0, 2
  vector<int> indptr = {0, 2, 3, 5};
  vector<int> indices = {0, 2, 1, 0, 2};
  CSCPattern pat(3, indptr, indices);

  CHECK(pat.n() == 3);
  CHECK(pat.nnz() == 5);
  CHECK(pat.indptr() == indptr);
  CHECK(pat.indices() == indices);
}

TEST_CASE("CSCPattern default constructor")
{
  CSCPattern pat;
  CHECK(pat.n() == 0);
  CHECK(pat.nnz() == 0);
}

TEST_CASE("CSCPattern operator==")
{
  vector<int> indptr = {0, 2, 3, 5};
  vector<int> indices = {0, 2, 1, 0, 2};
  CSCPattern a(3, indptr, indices);
  CSCPattern b(3, indptr, indices);

  CHECK(a == b);
  CHECK_FALSE(a != b);

  // Different dimension
  CSCPattern c(4, {0, 0, 0, 0, 0}, {});
  CHECK(a != c);

  // Same dimension but different pattern
  CSCPattern d(3, {0, 1, 2, 3}, {0, 1, 2});
  CHECK(a != d);
}

TEST_CASE("CSCPattern with_diagonal inserts missing diagonals")
{
  // Pattern with no diagonal entries:
  //   col 0: row 2
  //   col 1: (empty)
  //   col 2: row 0
  CSCPattern pat(3, {0, 1, 1, 2}, {2, 0});
  auto wd = pat.with_diagonal();

  CHECK(wd.n() == 3);
  // Original 2 entries + 3 missing diagonals = 5
  CHECK(wd.nnz() == 5);

  // Verify diagonals are present and rows stay sorted
  const auto& ip = wd.indptr();
  const auto& ix = wd.indices();

  // col 0: rows 0(new diag), 2
  CHECK(ip[0] == 0);
  CHECK(ix[0] == 0);
  CHECK(ix[1] == 2);
  // col 1: row 1(new diag)
  CHECK(ip[1] == 2);
  CHECK(ix[2] == 1);
  // col 2: rows 0, 2(new diag)
  CHECK(ip[2] == 3);
  CHECK(ix[3] == 0);
  CHECK(ix[4] == 2);
  CHECK(ip[3] == 5);
}

TEST_CASE("CSCPattern with_diagonal preserves existing diagonals")
{
  // Full diagonal already present
  CSCPattern pat(3, {0, 2, 4, 6}, {0, 1, 1, 2, 0, 2});
  auto wd = pat.with_diagonal();

  CHECK(wd.n() == 3);
  CHECK(wd.nnz() == 6); // unchanged
  CHECK(wd == pat);
}

TEST_CASE("CSCPattern with_diagonal partial diagonals")
{
  // 3x3 pattern: col 0 has diag, col 1 missing, col 2 has diag
  //   col 0: rows 0, 2
  //   col 1: row 0
  //   col 2: rows 1, 2
  CSCPattern pat(3, {0, 2, 3, 5}, {0, 2, 0, 1, 2});
  auto wd = pat.with_diagonal();

  CHECK(wd.nnz() == 6); // 5 + 1 missing diagonal for col 1

  // col 1 should now have rows 0, 1 (diagonal inserted in sorted order)
  const auto& ip = wd.indptr();
  const auto& ix = wd.indices();
  CHECK(ix[ip[1]] == 0);
  CHECK(ix[ip[1] + 1] == 1);
}

TEST_CASE("CSCMatrix construction and accessors")
{
  // 3x3 matrix:
  //   [1  0  2]
  //   [0  3  0]
  //   [4  0  5]
  vector<int> indptr = {0, 2, 3, 5};
  vector<int> indices = {0, 2, 1, 0, 2};
  vector<double> data = {1.0, 4.0, 3.0, 2.0, 5.0};
  CSCMatrix mat(3, indptr, indices, data);

  CHECK(mat.n() == 3);
  CHECK(mat.nnz() == 5);
  CHECK(mat.indptr() == indptr);
  CHECK(mat.indices() == indices);
  CHECK(mat.data() == data);

  // Verify pattern matches equivalent standalone CSCPattern
  CSCPattern pat(3, indptr, indices);
  CHECK(mat.pattern() == pat);
}

TEST_CASE("CSCMatrix default constructor")
{
  CSCMatrix mat;
  CHECK(mat.n() == 0);
  CHECK(mat.nnz() == 0);
}
