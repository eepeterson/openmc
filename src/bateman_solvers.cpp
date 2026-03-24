//! \file bateman_solvers.cpp
//! \brief Implementation of Bateman equation solvers

#include "openmc/bateman_solvers.h"

#include <algorithm> // for sort
#include <complex>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "openmc/error.h"

namespace openmc {

namespace {

// Fast complex reciprocal: 1/(a+bi) = (a-bi) / (a² + b²)
// Avoids GCC's __divdc3 which includes NaN/Inf handling we don't need.
// CRAM poles guarantee the diagonal is always well-conditioned.
inline std::complex<double> fast_crecip(std::complex<double> z)
{
  double a = z.real();
  double b = z.imag();
  double denom = a * a + b * b;
  return {a / denom, -b / denom};
}

// Fast complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
// Avoids GCC's __muldc3 which includes NaN recovery we don't need.
inline std::complex<double> fast_cmul(
  std::complex<double> x, std::complex<double> y)
{
  double a = x.real(), b = x.imag();
  double c = y.real(), d = y.imag();
  return {a * c - b * d, a * d + b * c};
}

} // anonymous namespace

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

  // Symbolic factorization: compute L/U sparsity patterns for this matrix.
  // Reused across all pole solves below.
  CSCPattern pattern = A.pattern().with_diagonal();
  symbolic_factorize(pattern);

  // IPF CRAM iteration:
  //   y_0 = n0
  //   y_{k+1} = y_k + 2*Re(alpha_k * (A*dt - theta_k*I)^{-1} * y_k)
  //   result = alpha0 * y_final
  vector<double> y(n0.begin(), n0.end());

  for (int p = 0; p < n_poles_; ++p) {
    numeric_factorize(A, pattern, dt, theta_[p]);
    triangular_solve(y, x_);

    // y += 2 * Re(alpha_p * x_p)
    for (int i = 0; i < n; ++i) {
      auto ax = fast_cmul(alpha_[p], x_[i]);
      y[i] += 2.0 * ax.real();
    }
  }

  // Final scaling
  for (int i = 0; i < n; ++i) {
    y[i] *= alpha0_;
  }

  return y;
}

//==============================================================================
// Symbolic factorization
//
// Computes the exact L/U sparsity patterns for left-looking column LU
// factorization without pivoting. Pivoting is unnecessary because the
// transmutation matrix A is Metzler (non-negative off-diagonal entries)
// and each CRAM pole theta has nonzero imaginary part. For M = A*dt - theta*I
// with A Metzler, unpivoted Gaussian elimination produces pivots u_jj
// satisfying |u_jj| >= |Im(theta)| >= 1.194. This guarantees non-singular
// factorization, and since no row swaps occur the L/U patterns are
// deterministic and identical across all poles.
//
// Algorithm: symbolic left-looking factorization with worklist-based fill
// propagation. For each column j, start with the structural nonzeros of
// the input pattern, then propagate fill through previously computed L
// column patterns. Any row k < j that becomes nonzero (a U entry) triggers
// examination of L[:,k]'s rows, which may create additional fill.
//==============================================================================

void IPFCramSolver::symbolic_factorize(const CSCPattern& pattern)
{
  int n = pattern.n();
  const auto& indptr = pattern.indptr();
  const auto& indices = pattern.indices();

  // Build L and U fill patterns column by column
  vector<vector<int>> l_cols(n);
  vector<bool> marked(n, false);
  vector<int> u_work, l_work;

  // Temporary storage for U column patterns (needed for CSC construction)
  vector<vector<int>> u_cols(n);

  for (int j = 0; j < n; ++j) {
    u_work.clear();
    l_work.clear();

    // Scatter: mark off-diagonal nonzero rows of column j
    for (int p = indptr[j]; p < indptr[j + 1]; ++p) {
      int i = indices[p];
      if (i == j)
        continue;
      if (!marked[i]) {
        marked[i] = true;
        if (i < j)
          u_work.push_back(i);
        else
          l_work.push_back(i);
      }
    }

    // Propagate fill through L columns of above-diagonal entries.
    // u_work grows as new above-diagonal rows are discovered via fill.
    for (size_t idx = 0; idx < u_work.size(); ++idx) {
      int k = u_work[idx];
      for (int row : l_cols[k]) {
        if (row == j)
          continue; // diagonal, skip
        if (!marked[row]) {
          marked[row] = true;
          if (row < j)
            u_work.push_back(row);
          else
            l_work.push_back(row);
        }
      }
    }

    // Sort row indices and store
    std::sort(u_work.begin(), u_work.end());
    std::sort(l_work.begin(), l_work.end());
    u_cols[j] = u_work;
    l_cols[j] = l_work;

    // Clear marked flags
    for (int k : u_work)
      marked[k] = false;
    for (int i : l_work)
      marked[i] = false;
  }

  // Build L CSC-style index arrays
  l_indptr_.resize(n + 1);
  l_rowidx_.clear();
  for (int j = 0; j < n; ++j) {
    l_indptr_[j] = static_cast<int>(l_rowidx_.size());
    for (int r : l_cols[j])
      l_rowidx_.push_back(r);
  }
  l_indptr_[n] = static_cast<int>(l_rowidx_.size());

  // Build U CSC-style index arrays (diagonal stored as last entry per column)
  u_indptr_.resize(n + 1);
  u_rowidx_.clear();
  for (int j = 0; j < n; ++j) {
    u_indptr_[j] = static_cast<int>(u_rowidx_.size());
    for (int r : u_cols[j])
      u_rowidx_.push_back(r);
    u_rowidx_.push_back(j); // diagonal last
  }
  u_indptr_[n] = static_cast<int>(u_rowidx_.size());

  // Allocate numeric workspace
  l_data_.resize(l_rowidx_.size());
  u_data_.resize(u_rowidx_.size());
  u_diag_.resize(n);
  work_.resize(n);
  x_.resize(n);
}

//==============================================================================
// Numeric factorization
//
// Left-looking column LU factorization without pivoting.
// Forms the shifted matrix M = A*dt - theta*I on-the-fly.
//
// For each column j, the above-diagonal U rows (stored in ascending order
// in u_rowidx_) serve as the left-looking schedule: each k < j with
// U[k,j] != 0 triggers a rank-1 update w -= L[:,k] * U[k,j]. Processing
// in ascending order naturally performs the forward substitution that
// resolves fill-in dependencies between earlier columns.
//==============================================================================

void IPFCramSolver::numeric_factorize(const CSCMatrix& A,
  const CSCPattern& pattern, double dt, std::complex<double> theta)
{
  int n = A.n();
  const auto& a_indptr = A.indptr();
  const auto& a_indices = A.indices();
  const auto& a_data = A.data();

  const auto& sp_indptr = pattern.indptr();
  const auto& sp_indices = pattern.indices();

  for (int j = 0; j < n; ++j) {

    // --- Step 1: Scatter M[:,j] = A[:,j]*dt - theta*I[:,j] ---
    {
      int a_pos = a_indptr[j];
      int a_end = a_indptr[j + 1];

      for (int sp_pos = sp_indptr[j]; sp_pos < sp_indptr[j + 1]; ++sp_pos) {
        int row = sp_indices[sp_pos];
        std::complex<double> val(0.0, 0.0);

        if (a_pos < a_end && a_indices[a_pos] == row) {
          val = dt * a_data[a_pos];
          ++a_pos;
        }

        if (row == j) {
          val -= theta;
        }

        work_[row] = val;
      }
    }

    // --- Step 2: Left-looking updates ---
    // Process U[:,j] rows in ascending order (forward substitution).
    // Each k < j with U[k,j] != 0 subtracts L[:,k] * U[k,j].
    for (int up = u_indptr_[j]; up < u_indptr_[j + 1] - 1; ++up) {
      int k = u_rowidx_[up];
      std::complex<double> ukj = work_[k];
      u_data_[up] = ukj;

      for (int lp = l_indptr_[k]; lp < l_indptr_[k + 1]; ++lp) {
        work_[l_rowidx_[lp]] -= fast_cmul(l_data_[lp], ukj);
      }
    }

    // --- Step 3: Extract diagonal and L column ---
    std::complex<double> inv_ujj = fast_crecip(work_[j]);
    u_diag_[j] = inv_ujj;
    u_data_[u_indptr_[j + 1] - 1] = work_[j];
    for (int lp = l_indptr_[j]; lp < l_indptr_[j + 1]; ++lp) {
      l_data_[lp] = fast_cmul(work_[l_rowidx_[lp]], inv_ujj);
    }

    // --- Step 4: Clear workspace ---
    // Clear all positions in the predicted fill pattern (U + L + diagonal)
    for (int up = u_indptr_[j]; up < u_indptr_[j + 1]; ++up) {
      work_[u_rowidx_[up]] = {0.0, 0.0};
    }
    for (int lp = l_indptr_[j]; lp < l_indptr_[j + 1]; ++lp) {
      work_[l_rowidx_[lp]] = {0.0, 0.0};
    }
  }
}

//==============================================================================
// Triangular solve
//
// Solve LUx = b without permutation (no pivoting means identity perm).
// Forward substitution solves Lz = b, back substitution solves Ux = z.
//==============================================================================

void IPFCramSolver::triangular_solve(
  const vector<double>& b, vector<std::complex<double>>& x) const
{
  int n = static_cast<int>(u_diag_.size());

  // Copy real RHS into complex vector
  for (int j = 0; j < n; ++j) {
    x[j] = std::complex<double>(b[j], 0.0);
  }

  // Forward substitution: Lz = b (L is unit lower triangular)
  for (int j = 0; j < n; ++j) {
    for (int lp = l_indptr_[j]; lp < l_indptr_[j + 1]; ++lp) {
      x[l_rowidx_[lp]] -= fast_cmul(l_data_[lp], x[j]);
    }
  }

  // Back substitution: Ux = z
  // u_diag_ stores reciprocals (1/U[j,j]) to avoid complex division
  for (int j = n - 1; j >= 0; --j) {
    x[j] = fast_cmul(x[j], u_diag_[j]);

    for (int up = u_indptr_[j]; up < u_indptr_[j + 1] - 1; ++up) {
      x[u_rowidx_[up]] -= fast_cmul(u_data_[up], x[j]);
    }
  }
}

//==============================================================================
// Decay-specific preparation
//
// Computes a topological permutation of the decay matrix so that the
// permuted matrix is strictly lower-triangular (plus diagonal). This is
// always possible for radioactive decay: parents decay into daughters,
// never the reverse. The permutation, inverse permutation, and permuted
// lower-triangular structure are cached for reuse across solve_decay()
// calls with the same matrix pattern.
//
// Uses Kahn's algorithm (BFS-based topological sort) on the column→row
// edges of the CSC matrix. Ties are broken by processing lower-index
// nodes first (via a min-heap), giving a deterministic ordering.
//==============================================================================

void IPFCramSolver::prepare_decay(const CSCMatrix& A_decay)
{
  int n = A_decay.n();
  const auto& indptr = A_decay.indptr();
  const auto& indices = A_decay.indices();
  const auto& data = A_decay.data();

  // Build in-degree counts from the off-diagonal structure.
  // In the CSC decay matrix, a nonzero at (row, col) with row != col means
  // col decays into row, i.e. col is a predecessor of row.
  // In-degree of a node = number of predecessors feeding into it.
  vector<int> in_degree(n, 0);
  for (int col = 0; col < n; ++col) {
    for (int p = indptr[col]; p < indptr[col + 1]; ++p) {
      int row = indices[p];
      if (row != col) {
        ++in_degree[row];
      }
    }
  }

  // Kahn's algorithm with min-heap for deterministic ordering
  std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
  for (int i = 0; i < n; ++i) {
    if (in_degree[i] == 0) {
      pq.push(i);
    }
  }

  decay_perm_.clear();
  decay_perm_.reserve(n);
  while (!pq.empty()) {
    int node = pq.top();
    pq.pop();
    decay_perm_.push_back(node);

    // "Remove" this node: decrement in-degree of its successors
    for (int p = indptr[node]; p < indptr[node + 1]; ++p) {
      int row = indices[p];
      if (row != node) {
        if (--in_degree[row] == 0) {
          pq.push(row);
        }
      }
    }
  }

  // Build inverse permutation
  decay_inv_perm_.resize(n);
  for (int i = 0; i < n; ++i) {
    decay_inv_perm_[decay_perm_[i]] = i;
  }

  // Build the permuted lower-triangular structure.
  // After permutation, all off-diagonal entries should satisfy new_row >
  // new_col (lower-triangular). We store these in CSC column-major order
  // with sorted row indices, plus a separate diagonal array.
  decay_diag_.assign(n, 0.0);

  // First pass: count entries per new column
  vector<int> col_counts(n, 0);
  for (int old_col = 0; old_col < n; ++old_col) {
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int old_row = indices[p];
      int new_col = decay_inv_perm_[old_col];
      int new_row = decay_inv_perm_[old_row];
      if (new_row == new_col) {
        decay_diag_[new_col] = data[p];
      } else {
        ++col_counts[new_col];
      }
    }
  }

  // Build column pointers
  decay_lt_indptr_.resize(n + 1);
  decay_lt_indptr_[0] = 0;
  for (int j = 0; j < n; ++j) {
    decay_lt_indptr_[j + 1] = decay_lt_indptr_[j] + col_counts[j];
  }

  // Second pass: fill row indices and values
  int lt_nnz = decay_lt_indptr_[n];
  decay_lt_rowidx_.resize(lt_nnz);
  decay_lt_data_.resize(lt_nnz);
  vector<int> col_pos(n, 0); // current write position per column

  for (int old_col = 0; old_col < n; ++old_col) {
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int old_row = indices[p];
      int new_col = decay_inv_perm_[old_col];
      int new_row = decay_inv_perm_[old_row];
      if (new_row != new_col) {
        int pos = decay_lt_indptr_[new_col] + col_pos[new_col]++;
        decay_lt_rowidx_[pos] = new_row;
        decay_lt_data_[pos] = data[p];
      }
    }
  }

  // Sort row indices within each column (and reorder data to match)
  for (int j = 0; j < n; ++j) {
    int start = decay_lt_indptr_[j];
    int end = decay_lt_indptr_[j + 1];
    int len = end - start;
    if (len <= 1)
      continue;

    // Build index permutation for this column's entries
    vector<int> order(len);
    for (int i = 0; i < len; ++i)
      order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return decay_lt_rowidx_[start + a] < decay_lt_rowidx_[start + b];
    });

    // Apply permutation
    vector<int> tmp_idx(len);
    vector<double> tmp_val(len);
    for (int i = 0; i < len; ++i) {
      tmp_idx[i] = decay_lt_rowidx_[start + order[i]];
      tmp_val[i] = decay_lt_data_[start + order[i]];
    }
    for (int i = 0; i < len; ++i) {
      decay_lt_rowidx_[start + i] = tmp_idx[i];
      decay_lt_data_[start + i] = tmp_val[i];
    }
  }

  // Cache the pattern for change detection
  decay_pattern_ = A_decay.pattern();

  // Ensure workspace is large enough
  if (static_cast<int>(x_.size()) < n) {
    x_.resize(n);
  }
}

//==============================================================================
// Fast decay solve
//
// Exploits the lower-triangular structure of pure-decay matrices.
// After topological permutation, (A'*dt - theta*I) is lower-triangular,
// so each CRAM pole requires only a single O(nnz) forward substitution
// — no LU factorization. This is dramatically faster than the general
// solve() for decay-only steps.
//==============================================================================

vector<double> IPFCramSolver::solve_decay(
  const CSCMatrix& A_decay, const vector<double>& n0, double dt)
{
  int n = A_decay.n();

  // Recompute permutation if pattern changed
  if (A_decay.pattern() != decay_pattern_) {
    prepare_decay(A_decay);
  }

  // Permute n0 into topological order
  vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = n0[decay_perm_[i]];
  }

  // IPF CRAM iteration in permuted space
  for (int p = 0; p < n_poles_; ++p) {
    auto theta_p = theta_[p];

    // Copy permuted y into complex RHS
    for (int j = 0; j < n; ++j) {
      x_[j] = std::complex<double>(y[j], 0.0);
    }

    // Forward substitution on (A'*dt - theta*I) x = y
    // A' is lower-triangular, so the shifted matrix is also lower-triangular.
    // Process columns 0..n-1 in order: solve for x[j], then update rows below.
    for (int j = 0; j < n; ++j) {
      // Divide by diagonal: (A'[j,j]*dt - theta)
      x_[j] = fast_cmul(x_[j],
        fast_crecip({decay_diag_[j] * dt - theta_p.real(), -theta_p.imag()}));

      // Subtract contribution from column j to rows below
      for (int lp = decay_lt_indptr_[j]; lp < decay_lt_indptr_[j + 1]; ++lp) {
        int row = decay_lt_rowidx_[lp];
        double a_val = decay_lt_data_[lp] * dt;
        // x[row] -= A'[row,j]*dt * x[j]
        x_[row] -=
          std::complex<double>(a_val * x_[j].real(), a_val * x_[j].imag());
      }
    }

    // y += 2 * Re(alpha_p * x_p)
    for (int i = 0; i < n; ++i) {
      auto ax = fast_cmul(alpha_[p], x_[i]);
      y[i] += 2.0 * ax.real();
    }
  }

  // Scale by alpha0 and permute back to original order
  vector<double> result(n);
  for (int i = 0; i < n; ++i) {
    result[decay_perm_[i]] = y[i] * alpha0_;
  }

  return result;
}

//==============================================================================
// Batch solve with OpenMP parallelism
//==============================================================================

void cram_solve_batch(int n_systems, int order, const int* dimensions,
  const int* indptr_offsets, const int* all_indptr, const int* indices_offsets,
  const int* all_indices, const double* all_data, const int* n0_offsets,
  const double* all_n0, double dt, bool decay_only, double* all_results)
{
  auto cram_order =
    (order == 16) ? IPFCramSolver::Order::cram16 : IPFCramSolver::Order::cram48;

  // Create one solver per thread (each has its own mutable workspace)
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  vector<IPFCramSolver> solvers;
  solvers.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    solvers.emplace_back(cram_order);
  }

#pragma omp parallel for schedule(dynamic) if(n_systems > 1)
  for (int i = 0; i < n_systems; ++i) {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    auto& solver = solvers[tid];

    int n = dimensions[i];

    // Reconstruct CSC matrix from packed arrays
    int ip_start = indptr_offsets[i];
    int ix_start = indices_offsets[i];
    int ip_len = n + 1;
    int nnz = all_indptr[ip_start + n] - all_indptr[ip_start];

    vector<int> indptr(all_indptr + ip_start, all_indptr + ip_start + ip_len);
    // Shift indptr so it starts at 0 (it was stored with absolute offsets)
    int base = indptr[0];
    if (base != 0) {
      for (auto& v : indptr)
        v -= base;
    }

    vector<int> indices(all_indices + ix_start, all_indices + ix_start + nnz);
    vector<double> data(all_data + ix_start, all_data + ix_start + nnz);

    CSCPattern pattern(n, std::move(indptr), std::move(indices));
    CSCMatrix A(std::move(pattern), std::move(data));

    // Get initial composition
    int n0_start = n0_offsets[i];
    vector<double> n0(all_n0 + n0_start, all_n0 + n0_start + n);

    // Solve
    vector<double> result;
    if (decay_only) {
      result = solver.solve_decay(A, n0, dt);
    } else {
      result = solver.solve(A, n0, dt);
    }

    // Copy result to output
    std::copy(result.begin(), result.end(), all_results + n0_start);
  }
}

} // namespace openmc

//==============================================================================
// C API
//==============================================================================

using namespace openmc;

extern "C" int openmc_cram_solve(int n, const int* indptr,
  const int* indices, const double* data, const double* n0, double dt,
  int order, double* result)
{
  try {
    auto cram_order = (order == 16) ? IPFCramSolver::Order::cram16
                                    : IPFCramSolver::Order::cram48;
    IPFCramSolver solver(cram_order);

    // Construct CSCMatrix from raw arrays (copies into solver-owned storage)
    vector<int> ip(indptr, indptr + n + 1);
    vector<int> ix(indices, indices + indptr[n]);
    vector<double> d(data, data + indptr[n]);
    CSCPattern pattern(n, std::move(ip), std::move(ix));
    CSCMatrix A(std::move(pattern), std::move(d));

    vector<double> n0_vec(n0, n0 + n);
    vector<double> y = solver.solve(A, n0_vec, dt);
    std::copy(y.begin(), y.end(), result);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_cram_solve_batch(int n_systems, int order,
  const int* dimensions, const int* indptr_offsets, const int* all_indptr,
  const int* indices_offsets, const int* all_indices, const double* all_data,
  const int* n0_offsets, const double* all_n0, double dt, bool decay_only,
  double* all_results)
{
  try {
    cram_solve_batch(n_systems, order, dimensions, indptr_offsets, all_indptr,
      indices_offsets, all_indices, all_data, n0_offsets, all_n0, dt,
      decay_only, all_results);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}
