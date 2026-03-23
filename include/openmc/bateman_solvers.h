//! \file bateman_solvers.h
//! \brief Solvers for the Bateman depletion equations

#ifndef OPENMC_BATEMAN_SOLVERS_H
#define OPENMC_BATEMAN_SOLVERS_H

#include <complex>

#include "openmc/memory.h"
#include "openmc/sparse_matrix.h"
#include "openmc/vector.h"

namespace openmc {

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
//! IPF CRAM solver for the Bateman equations
//!
//! Implements the Incomplete Partial Fraction form of the Chebyshev
//! Rational Approximation Method (CRAM), as described in:
//!   M. Pusa, "Higher-Order Chebyshev Rational Approximation Method and
//!   Application to Burnup Equations," Nucl. Sci. Eng., 182:3, 297-318 (2016).
//!
//! The solver caches L/U sparsity patterns so that repeated calls with the
//! same matrix pattern skip symbolic analysis. The numeric factorization
//! uses left-looking column LU without pivoting: the complex diagonal shift
//! from each CRAM pole guarantees |Im(theta)| >= 1.194, ensuring the
//! diagonal dominance needed for stable unpivoted factorization. This
//! makes the L/U patterns identical across all poles, allowing a single
//! symbolic phase to serve all 24 (CRAM48) or 8 (CRAM16) linear solves.
//==============================================================================

class IPFCramSolver : public BatemanSolver {
public:
  //! CRAM approximation order
  enum class Order { cram16, cram48 };

  explicit IPFCramSolver(Order order = Order::cram48);

  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) override;

private:
  // --- CRAM coefficients ---
  int n_poles_;                        //!< Number of poles (k/2)
  vector<std::complex<double>> alpha_; //!< Residues [n_poles]
  vector<std::complex<double>> theta_; //!< Poles [n_poles]
  double alpha0_;                      //!< Limit at infinity

  // --- Cached symbolic factorization state ---

  //! Sparsity pattern used for the current factorization (A's pattern with
  //! forced diagonal entries). Compared against incoming matrices to detect
  //! when re-analysis is needed.
  CSCPattern solve_pattern_;

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

  // --- Numeric factorization workspace (reused across poles) ---
  vector<std::complex<double>> l_data_; //!< L factor values [l_nnz]
  vector<std::complex<double>> u_data_; //!< U factor values [u_nnz]
  vector<std::complex<double>> u_diag_; //!< U diagonal values [n]

  // --- Solve workspace ---
  vector<std::complex<double>> work_; //!< Dense workspace [n]
  vector<std::complex<double>> x_;    //!< Complex solve result [n]

  // --- Private methods ---

  //! Compute L/U sparsity patterns for the given matrix structure.
  //! Uses a symbolic left-looking factorization with worklist-based fill
  //! propagation through previously computed L column patterns.
  void symbolic_factorize(const CSCPattern& pattern);

  //! Numerically factorize the shifted complex matrix (A*dt - theta*I).
  //! Uses left-looking column LU without pivoting.
  //! \param A     Real transmutation matrix
  //! \param dt    Time step
  //! \param theta Complex pole (shift)
  void numeric_factorize(
    const CSCMatrix& A, double dt, std::complex<double> theta);

  //! Solve the triangular system LUx = b using the current factorization.
  //! \param b  Right-hand side (real-valued initial composition)
  //! \param x  Solution vector (complex-valued)
  void triangular_solve(
    const vector<double>& b, vector<std::complex<double>>& x) const;
};

} // namespace openmc

#endif // OPENMC_BATEMAN_SOLVERS_H
