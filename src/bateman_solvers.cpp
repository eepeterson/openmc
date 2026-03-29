//! \file bateman_solvers.cpp
//! \brief Implementation of Bateman equation solvers

#include "openmc/bateman_solvers.h"

#include <algorithm> // for sort
#include <complex>
#include <numeric> // for iota

#include "openmc/capi.h"
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
// IPFCramSolver::compute_reachability
//
// Computes the structural reachability (transitive closure) for each column
// of a decay matrix under a topological permutation. The result is invariant
// for a given chain topology and can be reused across time steps.
//==============================================================================

void IPFCramSolver::compute_reachability(const CSCMatrix& A,
  const vector<int>& perm, vector<int>& reach_indptr,
  vector<int>& reach_indices)
{
  int n = A.n();
  const auto& indptr = A.indptr();
  const auto& indices = A.indices();
  const auto& data = A.data();

  // Build inverse permutation
  vector<int> inv_perm(n);
  for (int i = 0; i < n; ++i) {
    inv_perm[perm[i]] = i;
  }

  // Build lower-triangular column structure (pattern only).
  // Skip explicit zeros to avoid phantom edges.
  vector<int> lt_indptr(n + 1, 0);
  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int new_row = inv_perm[indices[p]];
      if (new_row != new_col && data[p] != 0.0) {
        ++lt_indptr[new_col + 1];
      }
    }
  }
  for (int j = 0; j < n; ++j) {
    lt_indptr[j + 1] += lt_indptr[j];
  }

  int lt_nnz = lt_indptr[n];
  vector<int> lt_rowidx(lt_nnz);
  vector<int> col_pos(n, 0);
  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int new_row = inv_perm[indices[p]];
      if (new_row != new_col && data[p] != 0.0) {
        lt_rowidx[lt_indptr[new_col] + col_pos[new_col]++] = new_row;
      }
    }
  }

  // Sort row indices within each column
  for (int j = 0; j < n; ++j) {
    std::sort(lt_rowidx.begin() + lt_indptr[j],
      lt_rowidx.begin() + lt_indptr[j + 1]);
  }

  // Compute reach via memoized transitive closure (leaves first).
  // reach[j] = sorted indices of all nodes reachable from j (excluding j).
  vector<vector<int>> reach(n);
  vector<int> merge_buf;

  for (int j = n - 1; j >= 0; --j) {
    int n_children = lt_indptr[j + 1] - lt_indptr[j];
    if (n_children == 0)
      continue;

    if (n_children == 1) {
      int c = lt_rowidx[lt_indptr[j]];
      auto& rc = reach[c];
      reach[j].resize(1 + rc.size());
      auto it = std::lower_bound(rc.begin(), rc.end(), c);
      size_t pos = it - rc.begin();
      std::copy(rc.begin(), it, reach[j].begin());
      reach[j][pos] = c;
      std::copy(it, rc.end(), reach[j].begin() + pos + 1);
    } else {
      int c0 = lt_rowidx[lt_indptr[j]];
      auto& rc0 = reach[c0];
      merge_buf.clear();
      merge_buf.reserve(rc0.size() + 1);
      auto it0 = std::lower_bound(rc0.begin(), rc0.end(), c0);
      merge_buf.insert(merge_buf.end(), rc0.begin(), it0);
      merge_buf.push_back(c0);
      merge_buf.insert(merge_buf.end(), it0, rc0.end());

      for (int lp = lt_indptr[j] + 1; lp < lt_indptr[j + 1]; ++lp) {
        int c = lt_rowidx[lp];
        auto& rc = reach[c];
        vector<int> child_set;
        child_set.reserve(rc.size() + 1);
        auto itc = std::lower_bound(rc.begin(), rc.end(), c);
        child_set.insert(child_set.end(), rc.begin(), itc);
        child_set.push_back(c);
        child_set.insert(child_set.end(), itc, rc.end());

        vector<int> merged;
        merged.reserve(merge_buf.size() + child_set.size());
        std::set_union(merge_buf.begin(), merge_buf.end(),
          child_set.begin(), child_set.end(), std::back_inserter(merged));
        merge_buf = std::move(merged);
      }

      reach[j] = std::move(merge_buf);
    }
  }

  // Flatten to CSC-like format
  reach_indptr.resize(n + 1);
  reach_indptr[0] = 0;
  for (int j = 0; j < n; ++j) {
    reach_indptr[j + 1] =
      reach_indptr[j] + static_cast<int>(reach[j].size());
  }
  int total = reach_indptr[n];
  reach_indices.resize(total);
  for (int j = 0; j < n; ++j) {
    std::copy(reach[j].begin(), reach[j].end(),
      reach_indices.begin() + reach_indptr[j]);
  }
}

//==============================================================================
// Sparse triangular solve for basis vector RHS
//
// Solves LUx = e_j using structural reachability in the L-factor graph.
// The reach vector contains the sorted column indices reachable from j
// in the directed graph of L (edge k->i iff L[i,k] != 0 structurally).
//==============================================================================
// Matrix exponential via IPF CRAM — decay-only variant
//
// Exploits topological permutation to lower-triangular form for pure-decay
// matrices. Forward substitution only (no U factor). Sparsity is preserved
// across all IPF poles because the nonzero pattern of each column is the
// DAG transitive closure, which is invariant under forward substitution.
//
// Algorithm:
//   1. Permute A into lower-triangular form, extract diagonal + off-diag
//   2. Use precomputed reach[j] (transitive closure, flat CSC-like arrays)
//   3. Store M in compressed form: only reach[j] entries per column j
//   4. For each pole k:
//      - For each column j: sparse forward sub over {j} ∪ reach[j]
//      - Accumulate F_k update into compressed M
//   5. Scale by alpha0, unpermute, build sparse CSC
//==============================================================================

CSCMatrix IPFCramSolver::expm(
  const CSCMatrix& A, double dt, double drop_tol,
  const vector<int>& perm, const int* reach_indptr,
  const int* reach_indices)
{
  int n = A.n();
  const auto& indptr = A.indptr();
  const auto& indices = A.indices();
  const auto& data = A.data();

  // Build inverse permutation
  vector<int> inv_perm(n);
  for (int i = 0; i < n; ++i) {
    inv_perm[perm[i]] = i;
  }

  // Scatter A into permuted lower-triangular structure.
  // Skip explicit zeros in the CSC data array to avoid phantom edges
  // in the reachability graph.
  diag_.assign(n, 0.0);

  vector<int> col_counts(n, 0);
  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int new_row = inv_perm[indices[p]];
      if (new_row == new_col) {
        diag_[new_col] = data[p];
      } else if (data[p] != 0.0) {
        ++col_counts[new_col];
      }
    }
  }

  lt_indptr_.resize(n + 1);
  lt_indptr_[0] = 0;
  for (int j = 0; j < n; ++j) {
    lt_indptr_[j + 1] = lt_indptr_[j] + col_counts[j];
  }

  int lt_nnz = lt_indptr_[n];
  lt_rowidx_.resize(lt_nnz);
  lt_data_.resize(lt_nnz);
  vector<int> col_pos(n, 0);

  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int new_row = inv_perm[indices[p]];
      if (new_row != new_col && data[p] != 0.0) {
        int pos = lt_indptr_[new_col] + col_pos[new_col]++;
        lt_rowidx_[pos] = new_row;
        lt_data_[pos] = data[p];
      }
    }
  }

  // Sort row indices within each column
  for (int j = 0; j < n; ++j) {
    int start = lt_indptr_[j];
    int end = lt_indptr_[j + 1];
    int len = end - start;
    if (len <= 1)
      continue;

    vector<int> order(len);
    for (int i = 0; i < len; ++i)
      order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return lt_rowidx_[start + a] < lt_rowidx_[start + b];
    });

    vector<int> tmp_idx(len);
    vector<double> tmp_val(len);
    for (int i = 0; i < len; ++i) {
      tmp_idx[i] = lt_rowidx_[start + order[i]];
      tmp_val[i] = lt_data_[start + order[i]];
    }
    for (int i = 0; i < len; ++i) {
      lt_rowidx_[start + i] = tmp_idx[i];
      lt_data_[start + i] = tmp_val[i];
    }
  }

  // Compressed M column offsets derived from precomputed reach:
  // col_offset(j) = j + reach_indptr[j], since each column stores
  // 1 (diagonal) + reach_size entries. Total = n + reach_indptr[n].
  int total_nnz = n + reach_indptr[n];

  // Compressed M: M_vals[j + reach_indptr[j]] stores M[j,j], followed by
  // M[reach_indices[reach_indptr[j]+0], j], M[reach_indices[...+1], j], etc.
  // Initialize to identity: M[j,j] = 1.0
  vector<double> M_vals(total_nnz, 0.0);
  for (int j = 0; j < n; ++j) {
    M_vals[j + reach_indptr[j]] = 1.0;
  }

  // Dense complex workspace (reuse x_ member)
  x_.resize(n);

  // IPF iteration: M = F_K * ... * F_1 * I
  for (int p = 0; p < n_poles_; ++p) {
    auto theta_p = theta_[p];
    auto alpha_p = alpha_[p];

    for (int j = 0; j < n; ++j) {
      int off = j + reach_indptr[j];
      int roff = reach_indptr[j];
      int rlen = reach_indptr[j + 1] - roff;

      // Scatter compressed M column into dense workspace
      x_[j] = {M_vals[off], 0.0};
      for (int k = 0; k < rlen; ++k) {
        x_[reach_indices[roff + k]] = {M_vals[off + 1 + k], 0.0};
      }

      // Sparse forward substitution over {j} ∪ reach[j]
      // Process column j first
      x_[j] = fast_cmul(x_[j],
        fast_crecip({diag_[j] * dt - theta_p.real(), -theta_p.imag()}));
      for (int lp = lt_indptr_[j]; lp < lt_indptr_[j + 1]; ++lp) {
        double a_val = lt_data_[lp] * dt;
        x_[lt_rowidx_[lp]] -=
          std::complex<double>(a_val * x_[j].real(), a_val * x_[j].imag());
      }
      // Process reach columns in order
      for (int ki = 0; ki < rlen; ++ki) {
        int c = reach_indices[roff + ki];
        x_[c] = fast_cmul(x_[c],
          fast_crecip({diag_[c] * dt - theta_p.real(), -theta_p.imag()}));
        for (int lp = lt_indptr_[c]; lp < lt_indptr_[c + 1]; ++lp) {
          double a_val = lt_data_[lp] * dt;
          x_[lt_rowidx_[lp]] -=
            std::complex<double>(a_val * x_[c].real(), a_val * x_[c].imag());
        }
      }

      // Gather back: M[:,j] += 2*Re(alpha_p * x), then clear workspace
      auto ax = fast_cmul(alpha_p, x_[j]);
      M_vals[off] += 2.0 * ax.real();
      x_[j] = {0.0, 0.0};
      for (int k = 0; k < rlen; ++k) {
        ax = fast_cmul(alpha_p, x_[reach_indices[roff + k]]);
        M_vals[off + 1 + k] += 2.0 * ax.real();
        x_[reach_indices[roff + k]] = {0.0, 0.0};
      }
    }
  }

  // Scale by alpha0 and build CSC output directly in original (unpermuted)
  // space. Skip negative entries: exp(At) >= 0 for Metzler matrices, so any
  // negative value is a CRAM approximation artifact.
  //
  // Two-pass approach avoids the O(nnz log nnz) sort in from_triplets.
  // Pass 1: count entries per original column to build indptr.
  vector<int> out_indptr(n + 1, 0);
  for (int j = 0; j < n; ++j) {
    int orig_col = perm[j];
    int off = j + reach_indptr[j];
    int roff = reach_indptr[j];
    int rlen = reach_indptr[j + 1] - roff;
    if (M_vals[off] * alpha0_ > drop_tol)
      ++out_indptr[orig_col + 1];
    for (int k = 0; k < rlen; ++k) {
      if (M_vals[off + 1 + k] * alpha0_ > drop_tol)
        ++out_indptr[orig_col + 1];
    }
  }
  for (int c = 0; c < n; ++c) {
    out_indptr[c + 1] += out_indptr[c];
  }

  int out_nnz = out_indptr[n];
  vector<int> out_indices(out_nnz);
  vector<double> out_data(out_nnz);

  // Pass 2: scatter entries into CSC arrays.
  vector<int> col_pos2(n, 0);
  for (int j = 0; j < n; ++j) {
    int orig_col = perm[j];
    int off = j + reach_indptr[j];
    int roff = reach_indptr[j];
    int rlen = reach_indptr[j + 1] - roff;
    int base = out_indptr[orig_col];

    // Diagonal entry: row = perm[j] = orig_col
    double val = M_vals[off] * alpha0_;
    if (val > drop_tol) {
      int pos = base + col_pos2[orig_col]++;
      out_indices[pos] = orig_col;
      out_data[pos] = val;
    }

    // Reach entries
    for (int k = 0; k < rlen; ++k) {
      val = M_vals[off + 1 + k] * alpha0_;
      if (val > drop_tol) {
        int pos = base + col_pos2[orig_col]++;
        out_indices[pos] = perm[reach_indices[roff + k]];
        out_data[pos] = val;
      }
    }
  }

  // Sort row indices within each column (CSC format requires sorted rows).
  // Use insertion sort for short columns (optimal for ~5 entries), fall back
  // to std::sort for longer columns (e.g. spontaneous fission nuclides with
  // hundreds of decay products).
  for (int c = 0; c < n; ++c) {
    int start = out_indptr[c];
    int end = out_indptr[c + 1];
    int len = end - start;
    if (len <= 1)
      continue;
    if (len <= 16) {
      // Insertion sort — optimal for tiny arrays
      for (int i = start + 1; i < end; ++i) {
        int row = out_indices[i];
        double val = out_data[i];
        int j = i - 1;
        while (j >= start && out_indices[j] > row) {
          out_indices[j + 1] = out_indices[j];
          out_data[j + 1] = out_data[j];
          --j;
        }
        out_indices[j + 1] = row;
        out_data[j + 1] = val;
      }
    } else {
      // O(n log n) sort for larger columns
      vector<int> order(len);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
        [&](int a, int b) {
          return out_indices[start + a] < out_indices[start + b];
        });
      vector<int> tmp_idx(len);
      vector<double> tmp_val(len);
      for (int i = 0; i < len; ++i) {
        tmp_idx[i] = out_indices[start + order[i]];
        tmp_val[i] = out_data[start + order[i]];
      }
      std::copy(tmp_idx.begin(), tmp_idx.end(), out_indices.begin() + start);
      std::copy(tmp_val.begin(), tmp_val.end(), out_data.begin() + start);
    }
  }

  CSCPattern pattern(n, std::move(out_indptr), std::move(out_indices));
  return CSCMatrix(std::move(pattern), std::move(out_data));
}

// Overload without precomputed reachability — computes it internally.
CSCMatrix IPFCramSolver::expm(
  const CSCMatrix& A, double dt, double drop_tol,
  const vector<int>& perm)
{
  vector<int> ri, rx;
  compute_reachability(A, perm, ri, rx);
  return expm(A, dt, drop_tol, perm, ri.data(), rx.data());
}

//==============================================================================
// IPFCramSolver::solve (decay variant)
//
// Optimized CRAM solver for pure-decay (lower-triangular) matrices.
// The topological permutation is provided by the caller. The matrix values
// are scattered into a permuted lower-triangular structure, and each CRAM
// pole is solved by forward substitution only.
//==============================================================================

vector<double> IPFCramSolver::solve(
  const CSCMatrix& A, const vector<double>& n0,
  double dt, const vector<int>& perm)
{
  int n = A.n();
  const auto& indptr = A.indptr();
  const auto& indices = A.indices();
  const auto& data = A.data();

  // Build inverse permutation
  vector<int> inv_perm(n);
  for (int i = 0; i < n; ++i) {
    inv_perm[perm[i]] = i;
  }

  // Scatter A into permuted lower-triangular structure.
  // After permutation, off-diagonal entries satisfy new_row > new_col.
  diag_.assign(n, 0.0);

  // First pass: count off-diagonal entries per permuted column
  vector<int> col_counts(n, 0);
  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int new_row = inv_perm[indices[p]];
      if (new_row == new_col) {
        diag_[new_col] = data[p];
      } else {
        ++col_counts[new_col];
      }
    }
  }

  // Build column pointers
  lt_indptr_.resize(n + 1);
  lt_indptr_[0] = 0;
  for (int j = 0; j < n; ++j) {
    lt_indptr_[j + 1] = lt_indptr_[j] + col_counts[j];
  }

  // Second pass: fill row indices and values
  int lt_nnz = lt_indptr_[n];
  lt_rowidx_.resize(lt_nnz);
  lt_data_.resize(lt_nnz);
  vector<int> col_pos(n, 0);

  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr[old_col]; p < indptr[old_col + 1]; ++p) {
      int new_row = inv_perm[indices[p]];
      if (new_row != new_col) {
        int pos = lt_indptr_[new_col] + col_pos[new_col]++;
        lt_rowidx_[pos] = new_row;
        lt_data_[pos] = data[p];
      }
    }
  }

  // Sort row indices within each column
  for (int j = 0; j < n; ++j) {
    int start = lt_indptr_[j];
    int end = lt_indptr_[j + 1];
    int len = end - start;
    if (len <= 1)
      continue;

    vector<int> order(len);
    for (int i = 0; i < len; ++i)
      order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return lt_rowidx_[start + a] < lt_rowidx_[start + b];
    });

    vector<int> tmp_idx(len);
    vector<double> tmp_val(len);
    for (int i = 0; i < len; ++i) {
      tmp_idx[i] = lt_rowidx_[start + order[i]];
      tmp_val[i] = lt_data_[start + order[i]];
    }
    for (int i = 0; i < len; ++i) {
      lt_rowidx_[start + i] = tmp_idx[i];
      lt_data_[start + i] = tmp_val[i];
    }
  }

  // Permute n0 into topological order
  vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = n0[perm[i]];
  }

  // IPF CRAM iteration in permuted space
  x_.resize(n);
  for (int p = 0; p < n_poles_; ++p) {
    auto theta_p = theta_[p];

    // Copy permuted y into complex RHS
    for (int j = 0; j < n; ++j) {
      x_[j] = std::complex<double>(y[j], 0.0);
    }

    // Forward substitution on (A'*dt - theta*I) x = y
    for (int j = 0; j < n; ++j) {
      x_[j] = fast_cmul(x_[j],
        fast_crecip({diag_[j] * dt - theta_p.real(), -theta_p.imag()}));

      for (int lp = lt_indptr_[j]; lp < lt_indptr_[j + 1]; ++lp) {
        int row = lt_rowidx_[lp];
        double a_val = lt_data_[lp] * dt;
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
    result[perm[i]] = y[i] * alpha0_;
  }

  return result;
}

} // namespace openmc

//==============================================================================
// C API
//==============================================================================

using namespace openmc;

extern "C" int openmc_cram_solve_batch(int n_materials, const int* dims,
  const int* all_indptr, const int* all_indices, const double* all_data,
  const int* nnz_per_mat, const double* all_n0, double dt, int order,
  const int* perm, double* all_results)
{
  try {
    if (order != 16 && order != 48) {
      set_errmsg(fmt::format(
        "CRAM order must be 16 or 48, got {}", order));
      return OPENMC_E_INVALID_ARGUMENT;
    }

    auto cram_order = (order == 16) ? IPFCramSolver::Order::cram16
                                    : IPFCramSolver::Order::cram48;

    // Precompute offsets into the concatenated arrays
    vector<int> indptr_offset(n_materials + 1, 0);
    vector<int> nnz_offset(n_materials + 1, 0);
    vector<int> n0_offset(n_materials + 1, 0);
    for (int m = 0; m < n_materials; ++m) {
      indptr_offset[m + 1] = indptr_offset[m] + dims[m] + 1;
      nnz_offset[m + 1] = nnz_offset[m] + nnz_per_mat[m];
      n0_offset[m + 1] = n0_offset[m] + dims[m];
    }

    // Track first error across threads
    int err_code = 0;
    std::string err_msg;

    #pragma omp parallel
    {
      IPFCramSolver solver(cram_order);

      #pragma omp for schedule(dynamic)
      for (int m = 0; m < n_materials; ++m) {
        // Skip remaining work if another thread hit an error
        if (err_code != 0) continue;

        try {
          int nm = dims[m];
          int nnz = nnz_per_mat[m];
          const int* ip = all_indptr + indptr_offset[m];
          const int* ix = all_indices + nnz_offset[m];
          const double* d = all_data + nnz_offset[m];
          const double* n0_m = all_n0 + n0_offset[m];
          double* res_m = all_results + n0_offset[m];

          CSCPattern pattern(
            nm, vector<int>(ip, ip + nm + 1), vector<int>(ix, ix + nnz));
          CSCMatrix A(std::move(pattern), vector<double>(d, d + nnz));
          vector<double> n0_vec(n0_m, n0_m + nm);

          vector<double> y;
          if (perm != nullptr) {
            vector<int> perm_vec(perm, perm + nm);
            y = solver.solve(A, n0_vec, dt, perm_vec);
          } else {
            y = solver.solve(A, n0_vec, dt);
          }

          std::copy(y.begin(), y.end(), res_m);
        } catch (const std::exception& e) {
          #pragma omp critical
          {
            if (err_code == 0) {
              err_code = OPENMC_E_UNASSIGNED;
              err_msg = fmt::format(
                "Error solving material {}: {}", m, e.what());
            }
          }
        }
      }
    }

    if (err_code != 0) {
      set_errmsg(err_msg);
      return err_code;
    }
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_cram_solve(int n, const int* indptr,
  const int* indices, const double* data, const double* n0, double dt,
  int order, const int* perm, double* result)
{
  try {
    if (order != 16 && order != 48) {
      set_errmsg(fmt::format(
        "CRAM order must be 16 or 48, got {}", order));
      return OPENMC_E_INVALID_ARGUMENT;
    }

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
    vector<double> y;

    if (perm != nullptr) {
      vector<int> perm_vec(perm, perm + n);
      y = solver.solve(A, n0_vec, dt, perm_vec);
    } else {
      y = solver.solve(A, n0_vec, dt);
    }

    std::copy(y.begin(), y.end(), result);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_decay_reachability(int n, const int* indptr,
  const int* indices, const double* data, const int* perm,
  int* reach_indptr, int* reach_indices, int* total_reach)
{
  try {
    vector<int> ip(indptr, indptr + n + 1);
    vector<int> ix(indices, indices + indptr[n]);
    vector<double> d(data, data + indptr[n]);
    CSCPattern pattern(n, std::move(ip), std::move(ix));
    CSCMatrix A(std::move(pattern), std::move(d));

    vector<int> perm_vec(perm, perm + n);
    vector<int> ri, rx;
    IPFCramSolver::compute_reachability(A, perm_vec, ri, rx);

    *total_reach = static_cast<int>(rx.size());

    if (reach_indptr != nullptr) {
      std::copy(ri.begin(), ri.end(), reach_indptr);
    }
    if (reach_indices != nullptr) {
      std::copy(rx.begin(), rx.end(), reach_indices);
    }
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}

extern "C" int openmc_cram_expm(int n, const int* indptr,
  const int* indices, const double* data, double dt, int order,
  double drop_tol, const int* perm, const int* reach_indptr,
  const int* reach_indices, int* out_indptr, int* out_indices,
  double* out_data, int* out_nnz)
{
  try {
    if (order != 16 && order != 48) {
      set_errmsg(fmt::format(
        "CRAM order must be 16 or 48, got {}", order));
      return OPENMC_E_INVALID_ARGUMENT;
    }

    if (perm == nullptr) {
      set_errmsg("perm must be provided for expm (decay-only solver)");
      return OPENMC_E_INVALID_ARGUMENT;
    }

    auto cram_order = (order == 16) ? IPFCramSolver::Order::cram16
                                    : IPFCramSolver::Order::cram48;
    IPFCramSolver solver(cram_order);

    // Construct CSCMatrix from raw arrays
    vector<int> ip(indptr, indptr + n + 1);
    vector<int> ix(indices, indices + indptr[n]);
    vector<double> d(data, data + indptr[n]);
    CSCPattern pattern(n, std::move(ip), std::move(ix));
    CSCMatrix A(std::move(pattern), std::move(d));

    vector<int> perm_vec(perm, perm + n);

    CSCMatrix result;
    if (reach_indptr != nullptr && reach_indices != nullptr) {
      result = solver.expm(A, dt, drop_tol, perm_vec,
        reach_indptr, reach_indices);
    } else {
      result = solver.expm(A, dt, drop_tol, perm_vec);
    }

    *out_nnz = result.nnz();

    // If out_indices is NULL, this is a query call — just return nnz
    if (out_indices == nullptr) {
      // Also fill out_indptr if provided (always n+1 entries)
      if (out_indptr != nullptr) {
        std::copy(
          result.indptr().begin(), result.indptr().end(), out_indptr);
      }
      return 0;
    }

    // Fill output arrays
    std::copy(result.indptr().begin(), result.indptr().end(), out_indptr);
    std::copy(result.indices().begin(), result.indices().end(), out_indices);
    std::copy(result.data().begin(), result.data().end(), out_data);
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_UNASSIGNED;
  }
  return 0;
}
