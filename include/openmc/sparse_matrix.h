//! \file sparse_matrix.h
//! \brief Compressed Sparse Column (CSC) sparsity pattern and matrix classes

#ifndef OPENMC_SPARSE_MATRIX_H
#define OPENMC_SPARSE_MATRIX_H

#include <algorithm> // for sort, adjacent_find
#include <complex>   // for complex
#include <utility>   // for move

#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! CSC sparsity pattern (structure only, no values)
//!
//! Stores the compressed column structure of a square sparse matrix:
//! column pointers and row indices. Rows within each column are sorted in
//! ascending order.
//==============================================================================

class CSCPattern {
public:
  // Constructors
  CSCPattern() = default;
  CSCPattern(int n, vector<int> indptr, vector<int> indices)
    : n_(n), indptr_(std::move(indptr)), indices_(std::move(indices))
  {}

  //! Construct from coordinate (COO) triplets.
  //! Duplicate (row, col) pairs are allowed — only the structural pattern
  //! is retained (duplicates are collapsed to a single entry).
  //! \param n Matrix dimension (square n x n)
  //! \param rows Row indices
  //! \param cols Column indices
  static CSCPattern from_triplets(
    int n, const vector<int>& rows, const vector<int>& cols);

  // Accessors
  int n() const { return n_; }
  int nnz() const { return static_cast<int>(indices_.size()); }
  const vector<int>& indptr() const { return indptr_; }
  const vector<int>& indices() const { return indices_; }

  //! Return a new pattern with rows and columns permuted.
  //! \param perm Permutation vector: new_index -> old_index
  CSCPattern permute(const vector<int>& perm) const;

  //! Return a new pattern with all diagonal entries forced present.
  //! Existing entries (including any diagonals already present) are preserved.
  CSCPattern with_diagonal() const;

  //! Structural equality check. Two patterns are equal iff they have the same
  //! dimension and identical column pointers. For matrices derived from the
  //! same depletion chain, identical column structure guarantees identical row
  //! indices, so checking indices_ is not necessary.
  bool operator==(const CSCPattern& other) const;
  bool operator!=(const CSCPattern& other) const { return !(*this == other); }

private:
  int n_ {0};           //!< Matrix dimension
  vector<int> indptr_;  //!< Column pointers [n+1]
  vector<int> indices_; //!< Row indices [nnz], sorted within each column
};

//==============================================================================
//! CSC sparse matrix with real (double) values
//!
//! Associates a double-precision value with each structural nonzero defined
//! by the underlying CSCPattern.
//==============================================================================

class CSCMatrix {
public:
  // Constructors
  CSCMatrix() = default;
  CSCMatrix(CSCPattern pattern, vector<double> data)
    : pattern_(std::move(pattern)), data_(std::move(data))
  {}

  //! Construct from coordinate (COO) triplets.
  //! Duplicate (row, col) pairs are summed.
  //! \param n Matrix dimension (square n x n)
  //! \param rows Row indices
  //! \param cols Column indices
  //! \param vals Values
  static CSCMatrix from_triplets(int n, const vector<int>& rows,
    const vector<int>& cols, const vector<double>& vals);

  // Accessors
  int n() const { return pattern_.n(); }
  int nnz() const { return pattern_.nnz(); }
  const CSCPattern& pattern() const { return pattern_; }
  const vector<int>& indptr() const { return pattern_.indptr(); }
  const vector<int>& indices() const { return pattern_.indices(); }
  const vector<double>& data() const { return data_; }

  //! Return a new matrix with rows and columns permuted.
  //! \param perm Permutation vector: new_index -> old_index
  CSCMatrix permute(const vector<int>& perm) const;

private:
  CSCPattern pattern_;  //!< Structural pattern
  vector<double> data_; //!< Values [nnz]
};

//==============================================================================
//! CSC sparse matrix with complex values
//!
//! Pairs a CSCPattern with complex double-precision values. Used internally
//! by the CRAM solver for shifted linear systems.
//==============================================================================

class ComplexCSCMatrix {
public:
  // Constructors
  ComplexCSCMatrix() = default;
  ComplexCSCMatrix(CSCPattern pattern, vector<std::complex<double>> data)
    : pattern_(std::move(pattern)), data_(std::move(data))
  {}

  //! Construct from a real CSCMatrix with given pattern (which may have
  //! additional structural entries not in A, e.g. forced diagonal).
  //! Real values from A are copied to matching positions; extra positions
  //! in the target pattern are set to zero.
  //! \param target_pattern Sparsity pattern for the result (superset of A)
  //! \param A Source real matrix
  //! \param scale Scalar multiplier applied to all values from A
  static ComplexCSCMatrix from_real(
    const CSCPattern& target_pattern, const CSCMatrix& A, double scale = 1.0);

  // Accessors
  int n() const { return pattern_.n(); }
  int nnz() const { return pattern_.nnz(); }
  const CSCPattern& pattern() const { return pattern_; }
  const vector<int>& indptr() const { return pattern_.indptr(); }
  const vector<int>& indices() const { return pattern_.indices(); }
  const vector<std::complex<double>>& data() const { return data_; }
  vector<std::complex<double>>& data() { return data_; }

private:
  CSCPattern pattern_;                //!< Structural pattern
  vector<std::complex<double>> data_; //!< Values [nnz]
};

} // namespace openmc

#endif // OPENMC_SPARSE_MATRIX_H
