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

  //! Solve a pure-decay Bateman system.
  //!
  //! The decay matrix must have a DAG structure (no cycles), which is
  //! always true for radioactive decay chains. The default implementation
  //! falls back to the general solve(); subclasses may override with a
  //! faster algorithm that exploits the lower-triangular structure.
  //!
  //! \param A_decay  Sparse decay-only transmutation matrix (n x n)
  //! \param n0       Initial atom densities [n]
  //! \param dt       Time interval [s]
  //! \return         Final atom densities [n]
  virtual vector<double> solve_decay(
    const CSCMatrix& A_decay, const vector<double>& n0, double dt)
  {
    return solve(A_decay, n0, dt);
  }
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

  vector<double> solve(
    const CSCMatrix& A, const vector<double>& n0, double dt) override;

  //! Fast pure-decay solver exploiting lower-triangular structure.
  //!
  //! On first call (or when the matrix pattern changes), computes a
  //! topological permutation that makes the decay matrix lower-triangular.
  //! Each CRAM pole then requires only a forward substitution — no LU
  //! factorization. The permutation and permuted matrix structure are
  //! cached for subsequent calls with the same pattern.
  vector<double> solve_decay(
    const CSCMatrix& A_decay, const vector<double>& n0, double dt) override;

private:
  // --- CRAM coefficients ---
  int n_poles_;                        //!< Number of poles (k/2)
  vector<std::complex<double>> alpha_; //!< Residues [n_poles]
  vector<std::complex<double>> theta_; //!< Poles [n_poles]
  double alpha0_;                      //!< Limit at infinity

  // --- Symbolic factorization state (recomputed each solve call) ---

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

  // --- Cached decay-specific state ---

  //! Sparsity pattern of the last decay matrix (used to detect changes)
  CSCPattern decay_pattern_;

  //! Topological permutation: perm_[new_idx] = old_idx.
  //! Reorders the decay matrix into strictly lower-triangular form.
  vector<int> decay_perm_;

  //! Inverse permutation: inv_perm_[old_idx] = new_idx.
  vector<int> decay_inv_perm_;

  //! Column pointers for the permuted lower-triangular decay matrix.
  //! Only below-diagonal entries are stored (diagonal tracked separately).
  vector<int> decay_lt_indptr_;

  //! Row indices for the permuted lower-triangular decay matrix.
  vector<int> decay_lt_rowidx_;

  //! Values for the permuted lower-triangular decay matrix.
  vector<double> decay_lt_data_;

  //! Diagonal values of the permuted decay matrix.
  vector<double> decay_diag_;

  // --- Private methods ---

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

  //! Compute topological permutation and extract lower-triangular structure
  //! from a decay matrix. Caches permutation and structure for reuse.
  //! \param A_decay  Decay-only transmutation matrix
  void prepare_decay(const CSCMatrix& A_decay);
};

//==============================================================================
// Batch solve interface (for calling from C API with OpenMP parallelism)
//==============================================================================

//! Solve multiple independent Bateman systems in parallel using OpenMP.
//!
//! Each system is described by a CSC matrix packed into contiguous arrays.
//! The systems may have different dimensions (though in practice depletion
//! matrices from the same chain are all the same size). Each OpenMP thread
//! owns its own IPFCramSolver instance to avoid data races on mutable
//! factorization workspace.
//!
//! \param n_systems     Number of independent systems to solve
//! \param order         CRAM order: 16 or 48
//! \param dimensions    Matrix dimension for each system [n_systems]
//! \param indptr_offsets Offset into all_indptr for each system [n_systems+1]
//! \param all_indptr    Concatenated column pointer arrays
//! \param indices_offsets Offset into all_indices/all_data for each system
//! [n_systems+1] \param all_indices   Concatenated row index arrays \param
//! all_data      Concatenated value arrays \param n0_offsets    Offset into
//! all_n0/all_results for each system [n_systems+1] \param all_n0 Concatenated
//! initial composition vectors \param dt            Time step [s] (same for all
//! systems) \param decay_only    If true, use solve_decay() for all systems
//! \param all_results   Output: concatenated result vectors [sum of dimensions]
void cram_solve_batch(int n_systems, int order, const int* dimensions,
  const int* indptr_offsets, const int* all_indptr, const int* indices_offsets,
  const int* all_indices, const double* all_data, const int* n0_offsets,
  const double* all_n0, double dt, bool decay_only, double* all_results);

} // namespace openmc

#endif // OPENMC_BATEMAN_SOLVERS_H
