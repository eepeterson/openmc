//! \file bateman_solvers.h
//! \brief Solvers for the Bateman depletion equations

#ifndef OPENMC_BATEMAN_SOLVERS_H
#define OPENMC_BATEMAN_SOLVERS_H

#include <complex>

#include "openmc/memory.h"
#include "openmc/sparse_matrix.h"
#include "openmc/vector.h"

namespace openmc {

class DepletionChain; // Forward declaration

//! CRAM approximation order
enum class CramOrder { cram16, cram48 };

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
  //! \return    Final atom densities [n]
  virtual vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) = 0;
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
  explicit IPFCramSolver(CramOrder order = CramOrder::cram48);

  //! Solve using full LU factorization (general transmutation matrix).
  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) override;

private:
  // --- CRAM coefficients ---
  int n_poles_;                        //!< Number of poles (k/2)
  vector<std::complex<double>> alpha_; //!< Residues [n_poles]
  vector<std::complex<double>> theta_; //!< Poles [n_poles]
  double alpha0_;                      //!< Limit at infinity

  // --- Symbolic factorization state ---

  //! L factor structure (CSC, unit lower triangular, diagonal not stored).
  //! Row indices within each column are sorted in ascending order.
  vector<int> l_indptr_; //!< Column pointers [n+1]
  vector<int> l_rowidx_; //!< Row indices [l_nnz]

  //! U factor structure (CSC, including diagonal as last entry per column).
  //! Above-diagonal row indices are sorted in ascending order, followed by
  //! the diagonal entry. The ascending order of the above-diagonal rows
  //! enables correct left-looking forward substitution.
  vector<int> u_indptr_; //!< Column pointers [n+1]
  vector<int> u_rowidx_; //!< Row indices [u_nnz]

  // --- Numeric factorization workspace ---
  vector<std::complex<double>> l_data_; //!< L factor values [l_nnz]
  vector<std::complex<double>> u_data_; //!< U factor values [u_nnz]
  vector<std::complex<double>> u_diag_; //!< U diagonal values [n]
  vector<std::complex<double>> work_;   //!< Dense workspace [n]
  vector<std::complex<double>> x_;      //!< Complex solve result [n]

  // --- Private methods ---
  void symbolic_factorize(const CSCPattern& pattern);

  void numeric_factorize(const CSCMatrix& A, const CSCPattern& pattern,
    double dt, std::complex<double> theta);

  void triangular_solve(
    const vector<double>& b, vector<std::complex<double>>& x) const;
};

//==============================================================================
//! IPF CRAM decay solver using forward substitution
//!
//! Specialized solver for pure-decay (lower-triangular) matrices.
//! Exploits the fact that radioactive decay chains have a DAG structure:
//! a topological permutation makes the decay matrix strictly
//! lower-triangular. After permutation, each CRAM pole requires only
//! O(nnz) forward substitution instead of a full LU factorization.
//!
//! The permutation is provided at construction time and A.permute(perm)
//! is called at each solve to extract the lower-triangular structure.
//==============================================================================

class IPFCramDecaySolver : public BatemanSolver {
public:
  //! Construct a decay solver from a depletion chain.
  //! Reads the topological permutation from the chain.
  //! \param order CRAM approximation order
  //! \param chain Depletion chain with precomputed decay DAG properties
  IPFCramDecaySolver(CramOrder order, const DepletionChain& chain);

  //! Solve a pure-decay system via forward substitution.
  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) override;

private:
  // --- CRAM coefficients ---
  int n_poles_;
  vector<std::complex<double>> alpha_;
  vector<std::complex<double>> theta_;
  double alpha0_;

  // --- Structural data ---
  vector<int> perm_; //!< Topological permutation

  // --- Workspace (reused across calls) ---
  vector<double> diag_;            //!< Diagonal values [n]
  vector<std::complex<double>> x_; //!< Complex solve workspace [n]
};

//==============================================================================
//! IPF CRAM decay solver via cached matrix exponential
//!
//! Computes the full sparse matrix exponential exp(A*dt) once per
//! distinct dt, then applies it via sparse matrix-vector multiply.
//! This amortizes the O(n * total_reach * n_poles) exponential build
//! across many materials that share the same decay matrix and timestep.
//!
//! The permutation and reachability data are provided at construction.
//! The cached exponential is invalidated when dt changes.
//==============================================================================

class IPFCramExpmDecaySolver : public BatemanSolver {
public:
  //! Construct an expm decay solver from a depletion chain.
  //! Reads the topological permutation and reachability from the chain.
  //! \param order CRAM approximation order
  //! \param chain Depletion chain with precomputed decay DAG properties
  IPFCramExpmDecaySolver(CramOrder order, const DepletionChain& chain);

  //! Solve by applying cached exp(A*dt) to n0.
  //! Rebuilds the exponential if dt has changed since the last call.
  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) override;

  //! Build the sparse matrix exponential exp(A*dt).
  //! \param A        Sparse decay matrix (n x n)
  //! \param dt       Time interval [s]
  //! \param drop_tol Entries with |value| < drop_tol are dropped
  //! \return         Sparse matrix exponential exp(A*dt) as CSCMatrix
  CSCMatrix build_expm(
    const CSCMatrix& A, double dt, double drop_tol = 0.0);

private:
  // --- CRAM coefficients ---
  int n_poles_;
  vector<std::complex<double>> alpha_;
  vector<std::complex<double>> theta_;
  double alpha0_;

  // --- Structural data ---
  vector<int> perm_;          //!< Topological permutation
  vector<int> reach_indptr_;  //!< Reach column pointers [n+1]
  vector<int> reach_indices_; //!< Reach row indices

  // --- Cached matrix exponential ---
  CSCMatrix M_;              //!< Cached exp(A*dt)
  double cached_dt_ {-1.0}; //!< dt used to build M_, or -1 if uncached

  // --- Workspace ---
  vector<double> diag_;            //!< Diagonal values [n]
  vector<std::complex<double>> x_; //!< Complex workspace [n]
};

} // namespace openmc

#endif // OPENMC_BATEMAN_SOLVERS_H
