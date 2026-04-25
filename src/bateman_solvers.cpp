//! \file bateman_solvers.cpp
//! \brief Implementation of Bateman equation solvers

#include "openmc/bateman_solvers.h"

#include <algorithm> // for sort
#include <complex>
#include <numeric>   // for iota

#include "openmc/capi.h"
#include "openmc/chain.h"
#include "openmc/error.h"

namespace openmc {

namespace {

// Fast complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
// Avoids GCC's __muldc3 which includes NaN recovery we don't need.
inline std::complex<double> fast_cmul(
  std::complex<double> x, std::complex<double> y)
{
  double a = x.real(), b = x.imag();
  double c = y.real(), d = y.imag();
  return {a * c - b * d, a * d + b * c};
}

// Fast complex reciprocal: 1/(a+bi) = (a-bi) / (a^2 + b^2)
inline std::complex<double> fast_crecip(std::complex<double> z)
{
  double a = z.real();
  double b = z.imag();
  double denom = a * a + b * b;
  return {a / denom, -b / denom};
}

} // anonymous namespace

//==============================================================================
// LU factorization for the CRAM shifted operator
//==============================================================================

void numeric_factorize_cram(const CSCMatrix& A, double dt,
  std::complex<double> theta, const SymbolicLUFactorization& symbolic,
  NumericLUFactorization& numeric, vector<std::complex<double>>& work)
{
  int n = A.n();
  const auto& a_indptr = A.indptr();
  const auto& a_indices = A.indices();
  const auto& a_data = A.data();

  const auto& sp_indptr = symbolic.pattern.indptr();
  const auto& sp_indices = symbolic.pattern.indices();
  const auto& l_indptr = symbolic.l_pattern.indptr();
  const auto& l_rowidx = symbolic.l_pattern.indices();
  const auto& u_indptr = symbolic.u_pattern.indptr();
  const auto& u_rowidx = symbolic.u_pattern.indices();

  if (static_cast<int>(work.size()) != n) {
    work.resize(n);
  }
  numeric.l_data.resize(symbolic.l_pattern.nnz());
  numeric.u_data.resize(symbolic.u_pattern.nnz());
  numeric.u_diag_inv.resize(n);

  for (int j = 0; j < n; ++j) {

    // Scatter the shifted operator column: M[:, j] = dt * A[:, j] - theta * I[:, j]
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

        work[row] = val;
      }
    }

    // Left-looking updates.
    for (int up = u_indptr[j]; up < u_indptr[j + 1] - 1; ++up) {
      int k = u_rowidx[up];
      std::complex<double> ukj = work[k];
      numeric.u_data[up] = ukj;

      for (int lp = l_indptr[k]; lp < l_indptr[k + 1]; ++lp) {
        work[l_rowidx[lp]] -= fast_cmul(numeric.l_data[lp], ukj);
      }
    }

    std::complex<double> inv_ujj = fast_crecip(work[j]);
    numeric.u_diag_inv[j] = inv_ujj;
    numeric.u_data[u_indptr[j + 1] - 1] = work[j];
    for (int lp = l_indptr[j]; lp < l_indptr[j + 1]; ++lp) {
      numeric.l_data[lp] = fast_cmul(work[l_rowidx[lp]], inv_ujj);
    }

    // Reset the scatter buffer to zero for the next column. The union of
    // U[:, j] and L[:, j] row indices is a superset of every row written by
    // the scatter and left-looking updates above, so these two loops suffice.
    for (int up = u_indptr[j]; up < u_indptr[j + 1]; ++up) {
      work[u_rowidx[up]] = {0.0, 0.0};
    }
    for (int lp = l_indptr[j]; lp < l_indptr[j + 1]; ++lp) {
      work[l_rowidx[lp]] = {0.0, 0.0};
    }
    // Invariant: the union of U[:, j] rows and L[:, j] rows is a superset of
    // every row touched by the scatter and left-looking updates above, so
    // these two loops fully reset `work` to zero before the next column.
  }
}

void triangular_solve_lu(const vector<double>& b,
  const SymbolicLUFactorization& symbolic,
  const NumericLUFactorization& numeric, vector<std::complex<double>>& x)
{
  int n = symbolic.pattern.n();
  const auto& l_indptr = symbolic.l_pattern.indptr();
  const auto& l_rowidx = symbolic.l_pattern.indices();
  const auto& u_indptr = symbolic.u_pattern.indptr();
  const auto& u_rowidx = symbolic.u_pattern.indices();

  if (static_cast<int>(x.size()) != n) {
    x.resize(n);
  }

  for (int j = 0; j < n; ++j) {
    x[j] = std::complex<double>(b[j], 0.0);
  }

  for (int j = 0; j < n; ++j) {
    for (int lp = l_indptr[j]; lp < l_indptr[j + 1]; ++lp) {
      x[l_rowidx[lp]] -= fast_cmul(numeric.l_data[lp], x[j]);
    }
  }

  for (int j = n - 1; j >= 0; --j) {
    x[j] = fast_cmul(x[j], numeric.u_diag_inv[j]);

    for (int up = u_indptr[j]; up < u_indptr[j + 1] - 1; ++up) {
      x[u_rowidx[up]] -= fast_cmul(numeric.u_data[up], x[j]);
    }
  }
}

//==============================================================================
// CRAM coefficient tables
//
// Coefficients from M. Pusa, "Higher-Order Chebyshev Rational Approximation
// Method and Application to Burnup Equations," Nucl. Sci. Eng., 182:3,
// 297-318 (2016). Values match openmc/deplete/cram.py exactly.
//==============================================================================

namespace {

// --- CRAM16 coefficients (8 poles) ---

constexpr std::complex<double> cram16_theta[] = {
  {+3.509103608414918e+0, +8.436198985884374e+0},
  {+5.948152268951177e+0, +3.587457362018322e+0},
  {-5.264971343442647e+0, +1.622022147316793e+1},
  {+1.419375897185666e+0, +1.092536348449672e+1},
  {+6.416177699099435e+0, +1.194122393370139e+0},
  {+4.993174737717997e+0, +5.996881713603942e+0},
  {-1.413928462488886e+0, +1.349772569889275e+1},
  {-1.084391707869699e+1, +1.927744616718165e+1},
};

constexpr std::complex<double> cram16_alpha[] = {
  {+5.464930576870210e+3, -3.797983575308356e+4},
  {+9.045112476907548e+1, -1.115537522430261e+3},
  {+2.344818070467641e+2, -4.228020157070496e+2},
  {+9.453304067358312e+1, -2.951294291446048e+2},
  {+7.283792954673409e+2, -1.205646080220011e+5},
  {+3.648229059594851e+1, -1.155509621409682e+2},
  {+2.547321630156819e+1, -2.639500283021502e+1},
  {+2.394538338734709e+1, -5.650522971778156e+0},
};

constexpr double cram16_alpha0 = 2.124853710495224e-16;

// --- CRAM48 coefficients (24 poles) ---

constexpr std::complex<double> cram48_theta[] = {
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

constexpr std::complex<double> cram48_alpha[] = {
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

constexpr double cram48_alpha0 = 2.258038182743983e-47;

} // anonymous namespace

//==============================================================================
// IPFCramSolver implementation
//==============================================================================

IPFCramSolver::IPFCramSolver(int order)
{
  if (order == 16) {
    n_poles_ = 8;
    alpha_ = cram16_alpha;
    theta_ = cram16_theta;
    alpha0_ = cram16_alpha0;
  } else if (order == 48) {
    n_poles_ = 24;
    alpha_ = cram48_alpha;
    theta_ = cram48_theta;
    alpha0_ = cram48_alpha0;
  } else {
    throw std::invalid_argument {
      fmt::format("CRAM order must be 16 or 48, got {}.", order)};
  }
}

vector<double> IPFCramSolver::solve(
  const CSCMatrix& A, const vector<double>& n0, double dt, int substeps)
{
  if (substeps <= 0) {
    throw std::invalid_argument {
      fmt::format("substeps must be positive, got {}.", substeps)};
  }

  int n = A.n();
  double step_dt = dt / substeps;

  // Symbolic factorization: compute L/U sparsity patterns for this matrix.
  // Reused across all pole solves below.
  auto symbolic = symbolic_factorize(A.pattern().with_diagonal());
  vector<std::complex<double>> work(n);
  vector<std::complex<double>> x(n);

  // IPF CRAM iteration:
  //   y_0 = n0
  //   y_{k+1} = y_k + 2*Re(alpha_k * (A*dt - theta_k*I)^{-1} * y_k)
  //   result = alpha0 * y_final
  vector<double> y(n0.begin(), n0.end());

  if (substeps > 1) {
    vector<NumericLUFactorization> cached_pole_factorizations(n_poles_);
    for (int p = 0; p < n_poles_; ++p) {
      numeric_factorize_cram(
        A, step_dt, theta_[p], symbolic, cached_pole_factorizations[p], work);
    }

    for (int step = 0; step < substeps; ++step) {
      for (int p = 0; p < n_poles_; ++p) {
        triangular_solve_lu(y, symbolic, cached_pole_factorizations[p], x);

        // y += 2 * Re(alpha_p * x_p)
        for (int i = 0; i < n; ++i) {
          auto ax = fast_cmul(alpha_[p], x[i]);
          y[i] += 2.0 * ax.real();
        }
      }

      for (int i = 0; i < n; ++i) {
        y[i] *= alpha0_;
      }
    }
  } else {
    // Reuse a single numeric container while recomputing each pole on demand.
    NumericLUFactorization current_pole_factorization;
    for (int p = 0; p < n_poles_; ++p) {
      numeric_factorize_cram(
        A, step_dt, theta_[p], symbolic, current_pole_factorization, work);
      triangular_solve_lu(y, symbolic, current_pole_factorization, x);

      // y += 2 * Re(alpha_p * x_p)
      for (int i = 0; i < n; ++i) {
        auto ax = fast_cmul(alpha_[p], x[i]);
        y[i] += 2.0 * ax.real();
      }
    }

    for (int i = 0; i < n; ++i) {
      y[i] *= alpha0_;
    }
  }

  return y;
}

//==============================================================================
// IPFCramDecaySolver implementation
//==============================================================================

IPFCramDecaySolver::IPFCramDecaySolver(
  const DepletionChain& chain, int order)
  : chain_(chain),
    perm_(chain.decay_perm()),
    diag_(chain.decay_diag()),
    lt_indptr_(chain.decay_lt_indptr()),
    lt_rowidx_(chain.decay_lt_rowidx()),
    lt_data_(chain.decay_lt_data()),
    reach_indptr_(chain.decay_reach_indptr()),
    reach_indices_(chain.decay_reach_indices())
{
  if (order == 16) {
    n_poles_ = 8;
    alpha_ = cram16_alpha;
    theta_ = cram16_theta;
    alpha0_ = cram16_alpha0;
  } else if (order == 48) {
    n_poles_ = 24;
    alpha_ = cram48_alpha;
    theta_ = cram48_theta;
    alpha0_ = cram48_alpha0;
  } else {
    throw std::invalid_argument {
      fmt::format("CRAM order must be 16 or 48, got {}.", order)};
  }
}

vector<double> IPFCramDecaySolver::solve_step(
  const vector<double>& n0, double dt) const
{
  int n = chain_.size();

  // Permute n0 into topological order: y[topo_idx] = n0[orig_idx].
  vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = n0[perm_[i]];
  }

  // Each pole solve fuses forward substitution and accumulation into a
  // single pass over the precomputed permuted lower-triangular structure.
  vector<std::complex<double>> x(n);
  for (int p = 0; p < n_poles_; ++p) {
    auto alpha_p = alpha_[p];
    auto theta_p = theta_[p];

    for (int j = 0; j < n; ++j) {
      x[j] = std::complex<double>(y[j], 0.0);
    }

    for (int j = 0; j < n; ++j) {
      // Diagonal solve: x[j] /= (diag*dt - theta).
      x[j] = fast_cmul(x[j], fast_crecip(std::complex<double> {
                                diag_[j] * dt - theta_p.real(),
                                -theta_p.imag()}));

      // Accumulate immediately — x[j] won't change again for this pole.
      y[j] += 2.0 * fast_cmul(alpha_p, x[j]).real();

      // Scatter contributions to later rows.
      for (int lp = lt_indptr_[j]; lp < lt_indptr_[j + 1]; ++lp) {
        double a = lt_data_[lp] * dt;
        x[lt_rowidx_[lp]] -=
          std::complex<double>(a * x[j].real(), a * x[j].imag());
      }
    }
  }

  // Scale by alpha0 and unpermute.
  vector<double> result(n);
  for (int i = 0; i < n; ++i) {
    result[perm_[i]] = y[i] * alpha0_;
  }
  return result;
}

CSCMatrix IPFCramDecaySolver::build_expm(double dt_sub) const
{
  int n = chain_.size();
  const int* reach_indptr = reach_indptr_.data();
  const int* reach_indices = reach_indices_.data();

  // Compressed M storage: column j stores M[j,j] followed by entries for
  // each row in reach[j]. Total size = n + reach_indptr[n].
  int total = n + reach_indptr[n];
  vector<double> M_vals(total, 0.0);

  // Initialize M = I (only diagonal entries).
  for (int j = 0; j < n; ++j) {
    M_vals[j + reach_indptr[j]] = 1.0;
  }

  vector<std::complex<double>> x(n, std::complex<double> {0.0, 0.0});

  // IPF iteration: for each pole, run column-by-column forward substitution
  // restricted to {j} ∪ reach[j].
  for (int p = 0; p < n_poles_; ++p) {
    auto theta_p = theta_[p];
    auto alpha_p = alpha_[p];

    for (int j = 0; j < n; ++j) {
      int off = j + reach_indptr[j];
      int roff = reach_indptr[j];
      int rlen = reach_indptr[j + 1] - roff;

      // Scatter compressed M column into dense complex workspace.
      x[j] = {M_vals[off], 0.0};
      for (int k = 0; k < rlen; ++k) {
        x[reach_indices[roff + k]] = {M_vals[off + 1 + k], 0.0};
      }

      // Forward substitution over {j} ∪ reach[j]. Process j first.
      x[j] = fast_cmul(x[j], fast_crecip(std::complex<double> {
                               diag_[j] * dt_sub - theta_p.real(),
                               -theta_p.imag()}));
      for (int lp = lt_indptr_[j]; lp < lt_indptr_[j + 1]; ++lp) {
        double a = lt_data_[lp] * dt_sub;
        x[lt_rowidx_[lp]] -=
          std::complex<double>(a * x[j].real(), a * x[j].imag());
      }
      for (int ki = 0; ki < rlen; ++ki) {
        int c = reach_indices[roff + ki];
        x[c] = fast_cmul(x[c], fast_crecip(std::complex<double> {
                                 diag_[c] * dt_sub - theta_p.real(),
                                 -theta_p.imag()}));
        for (int lp = lt_indptr_[c]; lp < lt_indptr_[c + 1]; ++lp) {
          double a = lt_data_[lp] * dt_sub;
          x[lt_rowidx_[lp]] -=
            std::complex<double>(a * x[c].real(), a * x[c].imag());
        }
      }

      // Gather and accumulate: M[:,j] += 2 * Re(alpha * x). Clear workspace.
      auto ax = fast_cmul(alpha_p, x[j]);
      M_vals[off] += 2.0 * ax.real();
      x[j] = {0.0, 0.0};
      for (int k = 0; k < rlen; ++k) {
        int r = reach_indices[roff + k];
        ax = fast_cmul(alpha_p, x[r]);
        M_vals[off + 1 + k] += 2.0 * ax.real();
        x[r] = {0.0, 0.0};
      }
    }
  }

  // Build CSC output in original (unpermuted) column space, scaling by
  // alpha0 and dropping non-positive values (Metzler-matrix exponentials
  // have non-negative entries; negatives are CRAM artifacts).
  vector<int> out_indptr(n + 1, 0);
  for (int j = 0; j < n; ++j) {
    int orig_col = perm_[j];
    int off = j + reach_indptr[j];
    int rlen = reach_indptr[j + 1] - reach_indptr[j];
    if (M_vals[off] * alpha0_ > 0.0)
      ++out_indptr[orig_col + 1];
    for (int k = 0; k < rlen; ++k) {
      if (M_vals[off + 1 + k] * alpha0_ > 0.0)
        ++out_indptr[orig_col + 1];
    }
  }
  for (int c = 0; c < n; ++c) {
    out_indptr[c + 1] += out_indptr[c];
  }

  int out_nnz = out_indptr[n];
  vector<int> out_indices(out_nnz);
  vector<double> out_data(out_nnz);

  vector<int> col_pos(n, 0);
  for (int j = 0; j < n; ++j) {
    int orig_col = perm_[j];
    int off = j + reach_indptr[j];
    int roff = reach_indptr[j];
    int rlen = reach_indptr[j + 1] - roff;
    int base = out_indptr[orig_col];

    double v = M_vals[off] * alpha0_;
    if (v > 0.0) {
      int pos = base + col_pos[orig_col]++;
      out_indices[pos] = orig_col;
      out_data[pos] = v;
    }
    for (int k = 0; k < rlen; ++k) {
      v = M_vals[off + 1 + k] * alpha0_;
      if (v > 0.0) {
        int pos = base + col_pos[orig_col]++;
        out_indices[pos] = perm_[reach_indices[roff + k]];
        out_data[pos] = v;
      }
    }
  }

  // Sort row indices within each column. Insertion sort for short columns,
  // index-permutation sort for longer ones.
  for (int c = 0; c < n; ++c) {
    int start = out_indptr[c];
    int end = out_indptr[c + 1];
    int len = end - start;
    if (len <= 1)
      continue;
    if (len <= 16) {
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
      vector<int> order(len);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
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

vector<double> IPFCramDecaySolver::apply_expm(
  const CSCMatrix& M, const vector<double>& n0)
{
  int n = M.n();
  const auto& indptr = M.indptr();
  const auto& indices = M.indices();
  const auto& data = M.data();

  vector<double> result(n, 0.0);
  for (int j = 0; j < n; ++j) {
    double xj = n0[j];
    if (xj == 0.0)
      continue;
    for (int p = indptr[j]; p < indptr[j + 1]; ++p) {
      result[indices[p]] += data[p] * xj;
    }
  }
  return result;
}

const CSCMatrix& IPFCramDecaySolver::cached_expm(double dt_sub)
{
  for (auto it = expm_cache_.begin(); it != expm_cache_.end(); ++it) {
    if (it->dt_sub == dt_sub) {
      // Move-to-front for LRU.
      ExpmEntry hit = std::move(*it);
      expm_cache_.erase(it);
      expm_cache_.insert(expm_cache_.begin(), std::move(hit));
      return expm_cache_.front().M;
    }
  }
  if (static_cast<int>(expm_cache_.size()) >= max_cache_size_) {
    expm_cache_.pop_back();
  }
  expm_cache_.insert(
    expm_cache_.begin(), ExpmEntry {dt_sub, build_expm(dt_sub)});
  return expm_cache_.front().M;
}

vector<double> IPFCramDecaySolver::solve(
  const CSCMatrix& A, const vector<double>& n0, double dt, int substeps)
{
  if (substeps <= 0) {
    throw std::invalid_argument {
      fmt::format("substeps must be positive, got {}.", substeps)};
  }
  int n = chain_.size();
  if (static_cast<int>(n0.size()) != n) {
    throw std::invalid_argument {
      fmt::format("IPFCramDecaySolver: n0 size ({}) != chain size ({}).",
        n0.size(), n)};
  }
  // A is intentionally ignored (the chain owns the cached decay matrix).
  (void)A;

  if (substeps == 1) {
    return solve_step(n0, dt);
  }

  // Build/lookup M = exp(decay * dt/substeps), then apply substeps times.
  double dt_sub = dt / substeps;
  const CSCMatrix& M = cached_expm(dt_sub);

  vector<double> y = n0;
  for (int s = 0; s < substeps; ++s) {
    y = apply_expm(M, y);
  }
  return y;
}

} // namespace openmc

//==============================================================================
// C API
//==============================================================================

using namespace openmc;

extern "C" int openmc_cram_solve(int n, const int* indptr, const int* indices,
  const double* data, const double* n0, double dt, int order, int substeps,
  int is_decay, double* result)
{
  if (!n0 || !result) {
    set_errmsg("openmc_cram_solve: null pointer argument");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (!is_decay && (!indptr || !indices || !data)) {
    set_errmsg("openmc_cram_solve: null pointer argument");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  if (n < 0) {
    set_errmsg(fmt::format("matrix dimension must be non-negative, got {}", n));
    return OPENMC_E_INVALID_ARGUMENT;
  }

  try {
    vector<double> n0_vec(n0, n0 + n);
    vector<double> y;

    if (is_decay) {
      if (!data::depletion_chain) {
        set_errmsg(
          "openmc_cram_solve: is_decay requested but no depletion chain "
          "is loaded; call openmc_load_depletion_chain() first.");
        return OPENMC_E_DATA;
      }
      if (n != data::depletion_chain->size()) {
        set_errmsg(fmt::format(
          "openmc_cram_solve: n ({}) != depletion chain size ({}).",
          n, data::depletion_chain->size()));
        return OPENMC_E_INVALID_ARGUMENT;
      }
      // Cache the decay solver; rebuild if chain or order changed.
      static std::unique_ptr<IPFCramDecaySolver> cached_solver;
      static const DepletionChain* cached_chain = nullptr;
      static int cached_order = -1;
      if (!cached_solver || cached_chain != data::depletion_chain.get() ||
          cached_order != order) {
        cached_solver = std::make_unique<IPFCramDecaySolver>(
          *data::depletion_chain, order);
        cached_chain = data::depletion_chain.get();
        cached_order = order;
      }
      // A is unused by IPFCramDecaySolver; pass an empty matrix.
      CSCMatrix A_unused {CSCPattern(n)};
      y = cached_solver->solve(A_unused, n0_vec, dt, substeps);
    } else {
      int nnz = indptr[n];
      CSCMatrix A(n, vector<int>(indptr, indptr + n + 1),
        vector<int>(indices, indices + nnz), vector<double>(data, data + nnz));
      IPFCramSolver solver(order);
      y = solver.solve(A, n0_vec, dt, substeps);
    }
    std::copy(y.begin(), y.end(), result);
  } catch (const std::invalid_argument& e) {
    set_errmsg(e.what());
    return OPENMC_E_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    set_errmsg(e.what());
    return OPENMC_E_DATA;
  }
  return 0;
}

extern "C" int openmc_load_depletion_chain(const char* filename)
{
  if (!filename) {
    set_errmsg("openmc_load_depletion_chain: null filename");
    return OPENMC_E_INVALID_ARGUMENT;
  }
  try {
    data::depletion_chain = make_unique<DepletionChain>();
    data::depletion_chain->load_xml(filename);
    data::chain_nuclide_map = data::depletion_chain->nuclide_map();
  } catch (const std::exception& e) {
    data::depletion_chain.reset();
    data::chain_nuclide_map.clear();
    set_errmsg(e.what());
    return OPENMC_E_DATA;
  }
  return 0;
}
