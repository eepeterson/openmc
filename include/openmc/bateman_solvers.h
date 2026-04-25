//! \file bateman_solvers.h
//! \brief Solvers for the Bateman depletion equations

#ifndef OPENMC_BATEMAN_SOLVERS_H
#define OPENMC_BATEMAN_SOLVERS_H

#include <complex>

#include "openmc/sparse_matrix.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Numeric LU factorization values for a CRAM shifted linear system
//!
//! Stores the complex-valued L/U entries for (dt*A - theta*I) corresponding
//! to a previously computed SymbolicLUFactorization. The U diagonal is stored
//! both in the CSC values array and as precomputed reciprocals for faster back
//! substitution.
//==============================================================================

struct NumericLUFactorization {
  vector<std::complex<double>> l_data;
  vector<std::complex<double>> u_data;
  vector<std::complex<double>> u_diag_inv;
};

//! Numeric left-looking LU factorization of the CRAM shifted operator
//! (dt*A - theta*I) without explicitly forming it. Uses no pivoting; this is
//! safe because A is Metzler (non-negative off-diagonal) and |Im(theta)| >=
//! 1.194 for every CRAM pole, so every pivot satisfies |u_jj| >= |Im(theta)|.
void numeric_factorize_cram(const CSCMatrix& A, double dt,
  std::complex<double> theta, const SymbolicLUFactorization& symbolic,
  NumericLUFactorization& numeric, vector<std::complex<double>>& work);

//! Solve LUx = b using a symbolic LU pattern and matching numeric values.
void triangular_solve_lu(const vector<double>& b,
  const SymbolicLUFactorization& symbolic,
  const NumericLUFactorization& numeric, vector<std::complex<double>>& x);

//==============================================================================
//! Abstract base class for Bateman equation solvers
//!
//! Solves dN/dt = A*N over an interval dt, given initial composition N(0).
//==============================================================================

class BatemanSolver {
public:
  virtual ~BatemanSolver() = default;

  //! Solve the Bateman equations
  //! \param A   Sparse transmutation matrix (n x n)
  //! \param n0  Initial atom densities [n]
  //! \param dt  Time interval [s]
  //! \param substeps Number of substeps to use within dt
  //! \return    Final atom densities [n]
  virtual vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt,
    int substeps = 1) = 0;
};

//==============================================================================
//! IPF CRAM solver for general transmutation matrices
//!
//! Implements the Incomplete Partial Fraction form of the Chebyshev
//! Rational Approximation Method (CRAM), as described in:
//!   M. Pusa, "Higher-Order Chebyshev Rational Approximation Method and
//!   Application to Burnup Equations," Nucl. Sci. Eng., 182:3, 297-318 (2016).
//!
//! The numeric factorization uses left-looking column LU without
//! pivoting. Each call to solve() performs a symbolic factorization
//! to compute L/U sparsity patterns, then reuses those patterns for
//! all pole solves within that call. Pivoting is unnecessary
//! because the transmutation matrix is Metzler (non-negative off-diagonal)
//! and the CRAM poles have nonzero imaginary parts (|Im(theta)| >= 1.194).
//! For any real Metzler matrix R and complex shift theta with Im(theta) != 0,
//! unpivoted Gaussian elimination on (R - theta*I) produces pivots u_jj
//! satisfying |u_jj| >= |Im(theta)|, guaranteeing non-singular factorization.
//! Since pivoting is not needed, the L/U sparsity patterns are deterministic
//! and identical across all poles, allowing a single symbolic phase to serve
//! all 24 (CRAM48) or 8 (CRAM16) linear solves.
//==============================================================================

class IPFCramSolver : public BatemanSolver {
public:
  explicit IPFCramSolver(int order = 48);

  //! Solve using full LU factorization (general transmutation matrix).
  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt,
    int substeps = 1) override;

private:
  // --- CRAM coefficients (non-owning views of the static tables) ---
  int n_poles_;                               //!< Number of poles (k/2)
  const std::complex<double>* alpha_;         //!< Residues [n_poles]
  const std::complex<double>* theta_;         //!< Poles [n_poles]
  double alpha0_;                             //!< Limit at infinity
};

class DepletionChain;

//==============================================================================
//! IPF CRAM solver specialized for pure-decay (acyclic) matrices
//!
//! When the depletion matrix has no transmutation rates (source rate = 0),
//! the dependency graph between nuclides is a directed acyclic graph and the
//! matrix can be permuted into strictly lower-triangular form. This solver
//! exploits that structure in two ways:
//!
//! - For substeps == 1, each CRAM pole solve fuses forward substitution and
//!   accumulation into a single pass over the precomputed permuted lower-
//!   triangular structure stored on the chain.
//! - For substeps > 1, the matrix exponential M = exp(A * dt/substeps) is
//!   built once column-by-column via CRAM (exploiting precomputed reachability
//!   to keep each column solve sparse) and applied substeps times via
//!   sparse matrix-vector multiply.
//==============================================================================

class IPFCramDecaySolver : public BatemanSolver {
public:
  IPFCramDecaySolver(const DepletionChain& chain, int order = 48);

  //! Solve the decay Bateman equations.
  //! \param A   Ignored: the chain owns the cached decay matrix. Its pattern
  //!            is asserted to match in debug builds; the argument exists for
  //!            BatemanSolver polymorphism.
  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt,
    int substeps = 1) override;

private:
  //! Per-step solve via fused forward-sub/accumulate (one pass per pole).
  vector<double> solve_step(const vector<double>& n0, double dt) const;

  //! Build M = exp(decay * dt_sub) using CRAM column-by-column, exploiting
  //! the chain's reachability arrays.
  CSCMatrix build_expm(double dt_sub) const;

  //! Apply a cached expm matrix M to n0 (sparse SpMV).
  static vector<double> apply_expm(
    const CSCMatrix& M, const vector<double>& n0);

  //! Look up M = exp(decay * dt_sub) in the LRU cache, building if missing.
  const CSCMatrix& cached_expm(double dt_sub);

  // Chain references (chain owns these)
  const DepletionChain& chain_;
  const vector<int>& perm_;
  const vector<double>& diag_;
  const vector<int>& lt_indptr_;
  const vector<int>& lt_rowidx_;
  const vector<double>& lt_data_;
  const vector<int>& reach_indptr_;
  const vector<int>& reach_indices_;

  // CRAM coefficients
  int n_poles_;
  const std::complex<double>* alpha_;
  const std::complex<double>* theta_;
  double alpha0_;

  // LRU expm cache (small linear scan; capacity 4)
  struct ExpmEntry {
    double dt_sub;
    CSCMatrix M;
  };
  vector<ExpmEntry> expm_cache_;
  static constexpr int max_cache_size_ = 4;
};

} // namespace openmc

#endif // OPENMC_BATEMAN_SOLVERS_H
