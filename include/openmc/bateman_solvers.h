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
//! Implements the Incomplete Partial Factorization form of the Chebyshev
//! Rational Approximation Method (CRAM), as described in:
//!   M. Pusa, "Higher-Order Chebyshev Rational Approximation Method and
//!   Application to Burnup Equations," Nucl. Sci. Eng., 182:3, 297-318 (2016).
//!
//! The solver caches sparse LU symbolic analysis so that repeated calls with
//! the same sparsity pattern reuse the elimination tree and nonzero structure.
//! Partial pivoting is used for numerical stability with fissile chains.
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

  //! Elimination tree: parent[j] is the parent of column j in the
  //! elimination tree, or -1 if j is a root.
  vector<int> etree_;

  //! Sparsity patterns of L and U factors, computed during symbolic analysis.
  //! L is unit lower triangular; U is upper triangular.
  CSCPattern l_pattern_;
  CSCPattern u_pattern_;

  //! Column post-ordering for efficient elimination tree traversal
  vector<int> post_order_;

  // --- Numeric factorization workspace (reused across poles) ---
  vector<std::complex<double>> l_data_; //!< L factor values [l_nnz]
  vector<std::complex<double>> u_data_; //!< U factor values [u_nnz]
  vector<int> piv_;                     //!< Row permutation [n]
  vector<int> inv_piv_;                 //!< Inverse row permutation [n]

  // --- Solve workspace ---
  vector<std::complex<double>> work_; //!< Dense workspace [n]
  vector<std::complex<double>> x_;    //!< Complex solve result [n]

  // --- Private methods ---

  //! Perform symbolic analysis of the sparsity pattern to determine the
  //! elimination tree and the nonzero structures of L and U.
  void symbolic_factorize(const CSCPattern& pattern);

  //! Numerically factorize the shifted complex matrix (A*dt - theta*I).
  //! Uses the pre-computed symbolic structure. Includes partial pivoting.
  //! \param A     Real transmutation matrix
  //! \param dt    Time step
  //! \param theta Complex pole (shift)
  void numeric_factorize(
    const CSCMatrix& A, double dt, std::complex<double> theta);

  //! Solve the triangular system LUx = Pb using the current factorization.
  //! \param b  Right-hand side (real-valued initial composition)
  //! \param x  Solution vector (complex-valued)
  void triangular_solve(
    const vector<double>& b, vector<std::complex<double>>& x) const;
};

} // namespace openmc

#endif // OPENMC_BATEMAN_SOLVERS_H
