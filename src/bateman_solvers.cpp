//! \file bateman_solvers.cpp
//! \brief Implementation of Bateman equation solvers

#include "openmc/bateman_solvers.h"

#include <algorithm> // for fill, max, swap
#include <cmath>     // for abs

#include "openmc/error.h"

namespace openmc {

//==============================================================================
// CRAM coefficient tables
//
// Coefficients from M. Pusa, "Higher-Order Chebyshev Rational Approximation
// Method and Application to Burnup Equations," Nucl. Sci. Eng., 182:3,
// 297-318 (2016). Values match openmc/deplete/cram.py exactly.
//==============================================================================

namespace {

// --- CRAM16 coefficients (8 poles) ---

const std::complex<double> cram16_theta[] = {
  {+3.509103608414918e+0, +8.436198985884374e+0},
  {+5.948152268951177e+0, +3.587457362018322e+0},
  {-5.264971343442647e+0, +1.622022147316793e+1},
  {+1.419375897185666e+0, +1.092536348449672e+1},
  {+6.416177699099435e+0, +1.194122393370139e+0},
  {+4.993174737717997e+0, +5.996881713603942e+0},
  {-1.413928462488886e+0, +1.349772569889275e+1},
  {-1.084391707869699e+1, +1.927744616718165e+1},
};

const std::complex<double> cram16_alpha[] = {
  {+5.464930576870210e+3, -3.797983575308356e+4},
  {+9.045112476907548e+1, -1.115537522430261e+3},
  {+2.344818070467641e+2, -4.228020157070496e+2},
  {+9.453304067358312e+1, -2.951294291446048e+2},
  {+7.283792954673409e+2, -1.205646080220011e+5},
  {+3.648229059594851e+1, -1.155509621409682e+2},
  {+2.547321630156819e+1, -2.639500283021502e+1},
  {+2.394538338734709e+1, -5.650522971778156e+0},
};

const double cram16_alpha0 = 2.124853710495224e-16;

// --- CRAM48 coefficients (24 poles) ---

const std::complex<double> cram48_theta[] = {
  {-4.465731934165702e+1, +6.233225190695437e+1},
  {-5.284616241568964e+0, +4.057499381311059e+1},
  {-8.867715667624458e+0, +4.325515754166724e+1},
  {+3.493013124279215e+0, +3.281615453173585e+1},
  {+1.564102508858634e+1, +1.558061616372237e+1},
  {+1.742097597385893e+1, +1.076629305714420e+1},
  {-2.834466755180654e+1, +5.492841024648724e+1},
  {+1.661569367939544e+1, +1.316994930024688e+1},
  {+8.011836167974721e+0, +2.780232111309410e+1},
  {-2.056267541998229e+0, +3.794824788914354e+1},
  {+1.449208170441839e+1, +1.799988210051809e+1},
  {+1.853807176907916e+1, +5.974332563100539e+0},
  {+9.932562704505182e+0, +2.532823409972962e+1},
  {-2.244223871767187e+1, +5.179633600312162e+1},
  {+8.590014121680897e-1, +3.536456194294350e+1},
  {-1.286192925744479e+1, +4.600304902833652e+1},
  {+1.164596909542055e+1, +2.287153304140217e+1},
  {+1.806076684783089e+1, +8.368200580099821e+0},
  {+5.870672154659249e+0, +3.029700159040121e+1},
  {-3.542938819659747e+1, +5.834381701800013e+1},
  {+1.901323489060250e+1, +1.194282058271408e+0},
  {+1.885508331552577e+1, +3.583428564427879e+0},
  {-1.734689708174982e+1, +4.883941101108207e+1},
  {+1.316284237125190e+1, +2.042951874827759e+1},
};

const std::complex<double> cram48_alpha[] = {
  {+6.387380733878774e+2, -6.743912502859256e+2},
  {+1.909896179065730e+2, -3.973203432721332e+2},
  {+4.236195226571914e+2, -2.041233768918671e+3},
  {+4.645770595258726e+2, -1.652917287299683e+3},
  {+7.765163276752433e+2, -1.783617639907328e+4},
  {+1.907115136768522e+3, -5.887068595142284e+4},
  {+2.909892685603256e+3, -9.953255345514560e+3},
  {+1.944772206620450e+2, -1.427131226068449e+3},
  {+1.382799786972332e+5, -3.256885197214938e+6},
  {+5.628442079602433e+3, -2.924284515884309e+4},
  {+2.151681283794220e+2, -1.121774011188224e+3},
  {+1.324720240514420e+3, -6.370088443140973e+4},
  {+1.617548476343347e+4, -1.008798413156542e+6},
  {+1.112729040439685e+2, -8.837109731680418e+1},
  {+1.074624783191125e+2, -1.457246116408180e+2},
  {+8.835727765158191e+1, -6.388286188419360e+1},
  {+9.354078136054179e+1, -2.195424319460237e+2},
  {+9.418142823531573e+1, -6.719055740098035e+2},
  {+1.040012390717851e+2, -1.693747595553868e+2},
  {+6.861882624343235e+1, -1.177598523430493e+1},
  {+8.766654491283722e+1, -4.596464999363902e+3},
  {+1.056007619389650e+2, -1.738294585524067e+3},
  {+7.738987569039419e+1, -4.311715386228984e+1},
  {+1.041366366475571e+2, -2.777743732451969e+2},
};

const double cram48_alpha0 = 2.258038182743983e-47;

} // anonymous namespace

//==============================================================================
// IPFCramSolver implementation
//==============================================================================

IPFCramSolver::IPFCramSolver(Order order)
{
  if (order == Order::cram16) {
    n_poles_ = 8;
    alpha_.assign(cram16_alpha, cram16_alpha + n_poles_);
    theta_.assign(cram16_theta, cram16_theta + n_poles_);
    alpha0_ = cram16_alpha0;
  } else {
    n_poles_ = 24;
    alpha_.assign(cram48_alpha, cram48_alpha + n_poles_);
    theta_.assign(cram48_theta, cram48_theta + n_poles_);
    alpha0_ = cram48_alpha0;
  }
}

vector<double> IPFCramSolver::solve(
  const CSCMatrix& A, const vector<double>& n0, double dt)
{
  int n = A.n();

  // Check if we need to redo symbolic factorization
  CSCPattern candidate = A.pattern().with_diagonal();
  if (!(candidate == solve_pattern_)) {
    solve_pattern_ = std::move(candidate);
    symbolic_factorize(solve_pattern_);
  }

  // Initialize result: y = alpha0 * n0
  vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = alpha0_ * n0[i];
  }

  // For each pole pair: solve (A*dt - theta_j*I) x = n0, accumulate
  for (int p = 0; p < n_poles_; ++p) {
    numeric_factorize(A, dt, theta_[p]);
    triangular_solve(n0, x_);

    // y += 2 * Re(alpha_j * x_j)
    for (int i = 0; i < n; ++i) {
      y[i] += 2.0 * std::real(alpha_[p] * x_[i]);
    }
  }

  return y;
}

//==============================================================================
// Symbolic factorization
//
// Computes the elimination tree and determines the nonzero structure of
// L and U factors for a given sparsity pattern. This is the expensive
// analysis step that is cached across calls with the same pattern.
//==============================================================================

void IPFCramSolver::symbolic_factorize(const CSCPattern& pattern)
{
  int n = pattern.n();
  const auto& indptr = pattern.indptr();
  const auto& indices = pattern.indices();

  // --- Step 1: Compute elimination tree using row-merge algorithm ---
  // For each column j, etree_[j] = min row index > j in L[:,j]
  // which corresponds to the parent of j in the elimination tree.
  etree_.assign(n, -1);
  vector<int> ancestor(n, -1);

  for (int j = 0; j < n; ++j) {
    for (int idx = indptr[j]; idx < indptr[j + 1]; ++idx) {
      int i = indices[idx];
      if (i >= j)
        continue; // only look at upper part (row < col)

      // Walk from i to root of current tree, finding j's parent
      int r = i;
      while (ancestor[r] != -1 && ancestor[r] != j) {
        int next = ancestor[r];
        ancestor[r] = j; // path compression
        r = next;
      }
      if (ancestor[r] == -1) {
        ancestor[r] = j;
        etree_[r] = j;
      }
    }
  }

  // --- Step 2: Compute column counts for L and U ---
  // L nonzeros per column: for column j of L, the nonzeros below the diagonal
  // include rows reachable through the elimination tree from A's column j.
  // U nonzeros per row (stored by column in CSC): entries above diagonal.

  // Use symbolic Cholesky-like analysis adapted for unsymmetric LU.
  // For left-looking LU without pivoting, L[:,j] has nonzeros in rows
  // that are reachable from the pattern of A[:,j] through the elimination tree.

  // Compute the row structure of each column of L and U
  vector<vector<int>> l_cols(n); // rows of L[:,j] (below diagonal, excluding j)
  vector<vector<int>> u_cols(n); // rows of U[:,j] (above diagonal, including j)

  vector<bool> marked(n, false);
  vector<int> stack;

  for (int j = 0; j < n; ++j) {
    // Mark all rows reachable from A[:,j] through the elimination tree
    stack.clear();

    for (int idx = indptr[j]; idx < indptr[j + 1]; ++idx) {
      int i = indices[idx];
      if (i == j) {
        // Diagonal always present in U
        continue;
      }

      // Walk up the elimination tree from i until we hit j or an
      // already-marked node
      int r = i;
      int stack_start = stack.size();
      while (r != -1 && r < j && !marked[r]) {
        stack.push_back(r);
        marked[r] = true;
        r = etree_[r];
      }

      // All nodes on the path contribute to U[r,j] (r < j) or L[i,j] (i > j)
      if (i > j && !marked[i]) {
        marked[i] = true;
        stack.push_back(i);
      }
    }

    // Separate into L and U contributions
    for (int r : stack) {
      if (r < j) {
        u_cols[j].push_back(r); // U[r,j]
      } else {
        l_cols[j].push_back(r); // L[r,j]
      }
      marked[r] = false; // reset for next column
    }

    // Sort rows within each column
    std::sort(u_cols[j].begin(), u_cols[j].end());
    std::sort(l_cols[j].begin(), l_cols[j].end());

    // Diagonal always in U
    u_cols[j].push_back(j);
  }

  // --- Step 3: Build L and U CSCPatterns ---
  {
    vector<int> l_indptr(n + 1, 0);
    vector<int> l_indices;
    for (int j = 0; j < n; ++j) {
      l_indptr[j] = static_cast<int>(l_indices.size());
      for (int r : l_cols[j]) {
        l_indices.push_back(r);
      }
    }
    l_indptr[n] = static_cast<int>(l_indices.size());
    l_pattern_ = CSCPattern(n, std::move(l_indptr), std::move(l_indices));
  }

  {
    vector<int> u_indptr(n + 1, 0);
    vector<int> u_indices;
    for (int j = 0; j < n; ++j) {
      u_indptr[j] = static_cast<int>(u_indices.size());
      for (int r : u_cols[j]) {
        u_indices.push_back(r);
      }
    }
    u_indptr[n] = static_cast<int>(u_indices.size());
    u_pattern_ = CSCPattern(n, std::move(u_indptr), std::move(u_indices));
  }

  // --- Step 4: Allocate numeric workspace ---
  l_data_.resize(l_pattern_.nnz());
  u_data_.resize(u_pattern_.nnz());
  piv_.resize(n);
  inv_piv_.resize(n);
  work_.resize(n);
  x_.resize(n);
}

//==============================================================================
// Numeric factorization
//
// Left-looking column LU factorization with partial pivoting.
// Forms the shifted matrix M = A*dt - theta*I on-the-fly (no temporary
// complex matrix is constructed).
//==============================================================================

void IPFCramSolver::numeric_factorize(
  const CSCMatrix& A, double dt, std::complex<double> theta)
{
  int n = A.n();
  const auto& a_indptr = A.indptr();
  const auto& a_indices = A.indices();
  const auto& a_data = A.data();

  const auto& sp_indptr = solve_pattern_.indptr();
  const auto& sp_indices = solve_pattern_.indices();

  const auto& l_indptr = l_pattern_.indptr();
  const auto& l_indices = l_pattern_.indices();
  const auto& u_indptr = u_pattern_.indptr();
  const auto& u_indices = u_pattern_.indices();

  // Initialize pivot as identity
  for (int i = 0; i < n; ++i) {
    piv_[i] = i;
    inv_piv_[i] = i;
  }

  // Zero workspace
  std::fill(work_.begin(), work_.end(), std::complex<double>(0.0, 0.0));

  for (int j = 0; j < n; ++j) {
    // --- Scatter M[:,j] = A[:,j]*dt - theta*delta(i,j) into workspace ---
    // Use the solve pattern (A's pattern + forced diagonal) to determine
    // which entries to scatter. Match against A's actual entries.
    {
      int a_pos = a_indptr[j];
      int a_end = a_indptr[j + 1];

      for (int sp_pos = sp_indptr[j]; sp_pos < sp_indptr[j + 1]; ++sp_pos) {
        int row = sp_indices[sp_pos];
        std::complex<double> val(0.0, 0.0);

        // Check if A has this entry
        if (a_pos < a_end && a_indices[a_pos] == row) {
          val = dt * a_data[a_pos];
          ++a_pos;
        }

        // Subtract theta on diagonal
        if (row == j) {
          val -= theta;
        }

        work_[row] = val;
      }
    }

    // --- Apply previous L columns (left-looking update) ---
    // For each k < j where U[k,j] is nonzero, subtract L[:,k] * U[k,j]
    for (int u_pos = u_indptr[j]; u_pos < u_indptr[j + 1]; ++u_pos) {
      int k = u_indices[u_pos];
      if (k >= j)
        break; // only process k < j

      // U[k,j] = work[piv[k]] (pivoted row k)
      std::complex<double> ukj = work_[piv_[k]];
      u_data_[u_pos] = ukj;

      if (std::abs(ukj) == 0.0)
        continue;

      // Subtract L[:,k] * U[k,j] from work
      for (int l_pos = l_indptr[k]; l_pos < l_indptr[k + 1]; ++l_pos) {
        int row = l_indices[l_pos];
        work_[piv_[row]] -= l_data_[l_pos] * ukj;
      }
    }

    // --- Partial pivoting: find max magnitude below diagonal ---
    {
      double max_abs = 0.0;
      int max_row = j;
      for (int l_pos = l_indptr[j]; l_pos < l_indptr[j + 1]; ++l_pos) {
        int row = l_indices[l_pos];
        double abs_val = std::abs(work_[piv_[row]]);
        if (abs_val > max_abs) {
          max_abs = abs_val;
          max_row = row;
        }
      }
      // Also consider diagonal
      double diag_abs = std::abs(work_[piv_[j]]);
      if (diag_abs >= max_abs) {
        max_row = j;
      }

      // Swap pivot entries if needed
      if (max_row != j) {
        std::swap(piv_[j], piv_[max_row]);
        inv_piv_[piv_[j]] = j;
        inv_piv_[piv_[max_row]] = max_row;
      }
    }

    // --- Store U[j,j] (diagonal of U, after pivoting) ---
    {
      // Find the diagonal position in u_indices for column j
      // It's the last entry since we put diagonal last during symbolic
      int u_diag_pos = u_indptr[j + 1] - 1;
      u_data_[u_diag_pos] = work_[piv_[j]];
    }

    // --- Compute and store L[:,j] = work / U[j,j] ---
    {
      std::complex<double> ujj = work_[piv_[j]];
      if (std::abs(ujj) == 0.0) {
        fatal_error("Zero pivot encountered in sparse LU factorization "
                    "during CRAM solve");
      }

      for (int l_pos = l_indptr[j]; l_pos < l_indptr[j + 1]; ++l_pos) {
        int row = l_indices[l_pos];
        l_data_[l_pos] = work_[piv_[row]] / ujj;
      }
    }

    // --- Clear workspace for this column ---
    for (int sp_pos = sp_indptr[j]; sp_pos < sp_indptr[j + 1]; ++sp_pos) {
      work_[sp_indices[sp_pos]] = {0.0, 0.0};
    }
    // Also clear any fill-in positions
    for (int l_pos = l_indptr[j]; l_pos < l_indptr[j + 1]; ++l_pos) {
      work_[piv_[l_indices[l_pos]]] = {0.0, 0.0};
    }
  }
}

//==============================================================================
// Triangular solve
//
// Solve LUx = Pb where P is the row permutation from pivoting.
// First solve Ly = Pb (forward substitution), then Ux = y (back substitution).
//==============================================================================

void IPFCramSolver::triangular_solve(
  const vector<double>& b, vector<std::complex<double>>& x) const
{
  int n = solve_pattern_.n();
  const auto& l_indptr = l_pattern_.indptr();
  const auto& l_indices = l_pattern_.indices();
  const auto& u_indptr = u_pattern_.indptr();
  const auto& u_indices = u_pattern_.indices();

  // Forward substitution: Ly = Pb
  // y is stored in x temporarily
  for (int j = 0; j < n; ++j) {
    x[j] = std::complex<double>(b[piv_[j]], 0.0);
  }

  for (int j = 0; j < n; ++j) {
    // x[j] is already correct (L has unit diagonal)
    // Update subsequent rows
    for (int l_pos = l_indptr[j]; l_pos < l_indptr[j + 1]; ++l_pos) {
      int row = l_indices[l_pos];
      x[row] -= l_data_[l_pos] * x[j];
    }
  }

  // Back substitution: Ux = y
  for (int j = n - 1; j >= 0; --j) {
    // Divide by U[j,j] (last entry in U's column j)
    int u_diag_pos = u_indptr[j + 1] - 1;
    x[j] /= u_data_[u_diag_pos];

    // Update earlier rows
    for (int u_pos = u_indptr[j]; u_pos < u_indptr[j + 1] - 1; ++u_pos) {
      int row = u_indices[u_pos];
      x[row] -= u_data_[u_pos] * x[j];
    }
  }
}

} // namespace openmc
