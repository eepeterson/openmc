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
  //! CRAM approximation order
  enum class Order { cram16, cram48 };

  explicit IPFCramSolver(Order order = Order::cram48);

  //! Solve using full LU factorization (general transmutation matrix).
  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) override;

  //! Solve a pure-decay system using triangular forward substitution.
  //!
  //! Exploits the fact that radioactive decay chains have a DAG structure:
  //! a topological permutation makes the decay matrix strictly
  //! lower-triangular. After permutation, each CRAM pole requires only
  //! O(nnz) forward substitution instead of a full LU factorization.
  //! At solve-time, the matrix values are scattered into the permuted
  //! lower-triangular structure, so changing decay constants between calls
  //! is fully supported without reconstruction.
  //!
  //! \param A    Sparse decay matrix (n x n)
  //! \param n0   Initial atom densities [n]
  //! \param dt   Time interval [s]
  //! \param perm Topological permutation: perm[new_idx] = old_idx.
  //!             Must reorder the decay matrix into lower-triangular form.
  //! \return     Final atom densities [n]
  vector<double> solve(const CSCMatrix& A, const vector<double>& n0,
    double dt, const vector<int>& perm);

  //! Compute the matrix exponential of a pure-decay matrix.
  //!
  //! Exploits topological permutation to lower-triangular form.
  //! Each CRAM pole requires only forward substitution, and the IPF
  //! iteration preserves sparsity across all poles: column j of exp(A*dt)
  //! has nonzeros only at positions reachable from j in the decay DAG
  //! (the transitive closure). For typical decay chains, this yields
  //! dramatically sparser results (~0.5% vs ~25% for full burnup) and
  //! a proportional speedup.
  //!
  //! \param A        Sparse decay matrix (n x n)
  //! \param dt       Time interval [s]
  //! \param drop_tol Entries with |value| < drop_tol are dropped (default 0)
  //! \param perm     Topological permutation: perm[new_idx] = old_idx.
  //!                 Must reorder the decay matrix into lower-triangular form.
  //! \return         Sparse matrix exponential exp(A*dt) as CSCMatrix
  CSCMatrix expm(const CSCMatrix& A, double dt, double drop_tol,
    const vector<int>& perm);

  //! Compute matrix exponential with precomputed reachability.
  //!
  //! Same as expm() but skips the internal reachability computation,
  //! using the provided flat arrays instead. This enables the caller to
  //! compute reachability once per chain and reuse it across time steps.
  //!
  //! \param A              Sparse decay matrix (n x n)
  //! \param dt             Time interval [s]
  //! \param drop_tol       Drop threshold for output entries
  //! \param perm           Topological permutation: perm[new_idx] = old_idx
  //! \param reach_indptr   Precomputed reach column pointers [n+1]
  //! \param reach_indices  Precomputed reach row indices
  //! \return               Sparse matrix exponential exp(A*dt) as CSCMatrix
  CSCMatrix expm(const CSCMatrix& A, double dt, double drop_tol,
    const vector<int>& perm, const int* reach_indptr,
    const int* reach_indices);

  //! Compute structural reachability for a pure-decay matrix.
  //!
  //! Delegates to CSCPattern::reachability after extracting the pattern.
  //! Provided for backward compatibility.
  //!
  //! \param A              Sparse decay matrix
  //! \param perm           Topological permutation: perm[new_idx] = old_idx
  //! \param reach_indptr   Output column pointers [n+1]
  //! \param reach_indices  Output row indices [total_reach]
  static void compute_reachability(const CSCMatrix& A,
    const vector<int>& perm, vector<int>& reach_indptr,
    vector<int>& reach_indices)
  {
    A.pattern().reachability(perm, reach_indptr, reach_indices);
  }

private:
  // --- CRAM coefficients ---
  int n_poles_;                        //!< Number of poles (k/2)
  vector<std::complex<double>> alpha_; //!< Residues [n_poles]
  vector<std::complex<double>> theta_; //!< Poles [n_poles]
  double alpha0_;                      //!< Limit at infinity

  // --- General solver: symbolic factorization state ---

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

  // --- General solver: numeric factorization workspace ---
  vector<std::complex<double>> l_data_; //!< L factor values [l_nnz]
  vector<std::complex<double>> u_data_; //!< U factor values [u_nnz]
  vector<std::complex<double>> u_diag_; //!< U diagonal values [n]
  vector<std::complex<double>> work_;   //!< Dense workspace [n]

  // --- Decay solver: permuted lower-triangular workspace ---
  vector<int> lt_indptr_;    //!< Column pointers [n+1]
  vector<int> lt_rowidx_;    //!< Row indices (below-diagonal only)
  vector<double> lt_data_;   //!< Values (below-diagonal only)
  vector<double> diag_;      //!< Diagonal values [n]

  // --- Shared workspace ---
  vector<std::complex<double>> x_; //!< Complex solve result [n]

  // --- Decay solver private methods ---

  //! Scatter a CSC matrix into permuted lower-triangular form.
  //!
  //! Populates diag_, lt_indptr_, lt_rowidx_, lt_data_ from the input
  //! matrix using the given topological permutation. Row indices within
  //! each column are sorted in ascending order after scatter.
  //!
  //! \param A    Sparse matrix (n x n)
  //! \param perm Topological permutation: perm[new_idx] = old_idx
  void scatter_to_lower_triangular(
    const CSCMatrix& A, const vector<int>& perm);

  // --- General solver private methods ---

  //! Compute L/U sparsity patterns for the given matrix structure.
  //! Uses a symbolic left-looking factorization with worklist-based fill
  //! propagation through previously computed L column patterns.
  void symbolic_factorize(const CSCPattern& pattern);

  //! Numerically factorize the shifted complex matrix (A*dt - theta*I).
  //! Uses left-looking column LU without pivoting.
  //! \param A       Real transmutation matrix
  //! \param pattern Input sparsity pattern (A with forced diagonal)
  //! \param dt      Time step
  //! \param theta   Complex pole (shift)
  void numeric_factorize(const CSCMatrix& A, const CSCPattern& pattern,
    double dt, std::complex<double> theta);

  //! Solve the triangular system LUx = b using the current factorization.
  //! \param b  Right-hand side (real-valued initial composition)
  //! \param x  Solution vector (complex-valued)
  void triangular_solve(
    const vector<double>& b, vector<std::complex<double>>& x) const;

};

} // namespace openmc

#endif // OPENMC_BATEMAN_SOLVERS_H
