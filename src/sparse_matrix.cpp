//! \file sparse_matrix.cpp
//! \brief Implementation of CSCPattern and CSCMatrix

#include "openmc/sparse_matrix.h"

#include <algorithm>  // for sort, fill
#include <functional> // for plus
#include <numeric>    // for iota
#include <utility>    // for pair

#include <fmt/core.h>

#include "openmc/error.h"

namespace openmc {

//==============================================================================
// CSCPattern implementation
//==============================================================================

CSCPattern CSCPattern::from_triplets(
  int n, const vector<int>& rows, const vector<int>& cols)
{
  int nt = rows.size();

  // Sort triplets by (col, row)
  vector<int> order(nt);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return cols[a] < cols[b] || (cols[a] == cols[b] && rows[a] < rows[b]);
  });

  // Build CSC arrays, collapsing duplicate (row, col) pairs
  vector<int> indptr(n + 1, 0);
  vector<int> indices;
  indices.reserve(nt);

  int prev_col = -1;
  int prev_row = -1;
  for (int k = 0; k < nt; ++k) {
    int i = rows[order[k]];
    int j = cols[order[k]];

    // Skip duplicate entries
    if (j == prev_col && i == prev_row)
      continue;

    indices.push_back(i);

    // Fill column pointers for any skipped columns
    for (int c = prev_col + 1; c <= j; ++c) {
      indptr[c] = static_cast<int>(indices.size()) - 1;
    }
    prev_col = j;
    prev_row = i;
  }
  // Fill remaining column pointers
  for (int c = prev_col + 1; c <= n; ++c) {
    indptr[c] = static_cast<int>(indices.size());
  }

  return CSCPattern(n, std::move(indptr), std::move(indices));
}

CSCPattern CSCPattern::permute(const vector<int>& perm) const
{
  // perm[new_index] = old_index
  // Build inverse: inv_perm[old_index] = new_index
  int n = n_;
  vector<int> inv_perm(n);
  for (int i = 0; i < n; ++i) {
    inv_perm[perm[i]] = i;
  }

  // Collect permuted triplets
  vector<int> new_rows, new_cols;
  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int idx = indptr_[old_col]; idx < indptr_[old_col + 1]; ++idx) {
      int old_row = indices_[idx];
      int new_row = inv_perm[old_row];
      new_rows.push_back(new_row);
      new_cols.push_back(new_col);
    }
  }

  return CSCPattern::from_triplets(n, new_rows, new_cols);
}

bool CSCPattern::operator==(const CSCPattern& other) const
{
  return n_ == other.n_ && indptr_ == other.indptr_ &&
         indices_ == other.indices_;
}

CSCPattern CSCPattern::with_diagonal() const
{
  // First pass: count entries per column, noting missing diagonals
  int extra = 0;
  for (int col = 0; col < n_; ++col) {
    bool has_diag = false;
    for (int idx = indptr_[col]; idx < indptr_[col + 1]; ++idx) {
      if (indices_[idx] == col) {
        has_diag = true;
        break;
      }
    }
    if (!has_diag)
      ++extra;
  }

  if (extra == 0) {
    return CSCPattern(n_, vector<int>(indptr_), vector<int>(indices_));
  }

  // Build new CSC directly, inserting diagonal entries in sorted position
  int new_nnz = nnz() + extra;
  vector<int> new_indptr(n_ + 1);
  vector<int> new_indices(new_nnz);

  int dst = 0;
  for (int col = 0; col < n_; ++col) {
    new_indptr[col] = dst;
    bool has_diag = false;
    bool diag_inserted = false;
    for (int idx = indptr_[col]; idx < indptr_[col + 1]; ++idx) {
      int row = indices_[idx];
      if (!has_diag && !diag_inserted && row > col) {
        new_indices[dst++] = col;
        diag_inserted = true;
      }
      if (row == col)
        has_diag = true;
      new_indices[dst++] = row;
    }
    if (!has_diag && !diag_inserted) {
      new_indices[dst++] = col;
    }
  }
  new_indptr[n_] = dst;

  return CSCPattern(n_, std::move(new_indptr), std::move(new_indices));
}

//==============================================================================
// CSCMatrix implementation
//==============================================================================

CSCMatrix CSCMatrix::from_triplets(int n, const vector<int>& rows,
  const vector<int>& cols, const vector<double>& vals)
{
  int nt = rows.size();

  // Sort triplets by (col, row)
  vector<int> order(nt);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return cols[a] < cols[b] || (cols[a] == cols[b] && rows[a] < rows[b]);
  });

  // Build CSC arrays, summing duplicate (row, col) pairs
  vector<int> indptr(n + 1, 0);
  vector<int> indices;
  vector<double> data;
  indices.reserve(nt);
  data.reserve(nt);

  int prev_col = -1;
  int prev_row = -1;
  for (int k = 0; k < nt; ++k) {
    int i = rows[order[k]];
    int j = cols[order[k]];
    double v = vals[order[k]];

    if (j == prev_col && i == prev_row) {
      // Sum duplicate entries
      data.back() += v;
      continue;
    }

    indices.push_back(i);
    data.push_back(v);

    // Fill column pointers for any skipped columns
    for (int c = prev_col + 1; c <= j; ++c) {
      indptr[c] = static_cast<int>(indices.size()) - 1;
    }
    prev_col = j;
    prev_row = i;
  }
  // Fill remaining column pointers
  for (int c = prev_col + 1; c <= n; ++c) {
    indptr[c] = static_cast<int>(indices.size());
  }

  CSCPattern pattern(n, std::move(indptr), std::move(indices));
  return CSCMatrix(std::move(pattern), std::move(data));
}

CSCMatrix CSCMatrix::permute(const vector<int>& perm) const
{
  // perm[new_index] = old_index
  int n = pattern_.n();
  vector<int> inv_perm(n);
  for (int i = 0; i < n; ++i) {
    inv_perm[perm[i]] = i;
  }

  // Collect permuted triplets with values
  vector<int> new_rows, new_cols;
  vector<double> new_vals;
  const auto& indptr = pattern_.indptr();
  const auto& indices = pattern_.indices();

  for (int old_col = 0; old_col < n; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int idx = indptr[old_col]; idx < indptr[old_col + 1]; ++idx) {
      int old_row = indices[idx];
      int new_row = inv_perm[old_row];
      new_rows.push_back(new_row);
      new_cols.push_back(new_col);
      new_vals.push_back(data_[idx]);
    }
  }

  return CSCMatrix::from_triplets(n, new_rows, new_cols, new_vals);
}

CSCMatrix CSCMatrix::operator+(const CSCMatrix& other) const
{
  int n = pattern_.n();
  if (other.n() != n) {
    fatal_error(fmt::format(
      "Cannot add CSC matrices with different dimensions ({} vs {})", n,
      other.n()));
  }

  const auto& a_indptr = indptr();
  const auto& a_indices = indices();
  const auto& b_indptr = other.indptr();
  const auto& b_indices = other.indices();

  // Merge sorted row indices column-by-column
  vector<int> new_indptr(n + 1);
  vector<int> new_indices;
  vector<double> new_data;
  new_indices.reserve(nnz() + other.nnz());
  new_data.reserve(nnz() + other.nnz());

  for (int col = 0; col < n; ++col) {
    new_indptr[col] = static_cast<int>(new_indices.size());
    int a_start = a_indptr[col], a_end = a_indptr[col + 1];
    int b_start = b_indptr[col], b_end = b_indptr[col + 1];
    int ai = a_start, bi = b_start;

    while (ai < a_end && bi < b_end) {
      if (a_indices[ai] < b_indices[bi]) {
        new_indices.push_back(a_indices[ai]);
        new_data.push_back(data_[ai]);
        ++ai;
      } else if (a_indices[ai] > b_indices[bi]) {
        new_indices.push_back(b_indices[bi]);
        new_data.push_back(other.data_[bi]);
        ++bi;
      } else {
        // Same row: sum values
        new_indices.push_back(a_indices[ai]);
        new_data.push_back(data_[ai] + other.data_[bi]);
        ++ai;
        ++bi;
      }
    }
    while (ai < a_end) {
      new_indices.push_back(a_indices[ai]);
      new_data.push_back(data_[ai]);
      ++ai;
    }
    while (bi < b_end) {
      new_indices.push_back(b_indices[bi]);
      new_data.push_back(other.data_[bi]);
      ++bi;
    }
  }
  new_indptr[n] = static_cast<int>(new_indices.size());

  CSCPattern pattern(n, std::move(new_indptr), std::move(new_indices));
  return CSCMatrix(std::move(pattern), std::move(new_data));
}

CSCMatrix& CSCMatrix::operator+=(const CSCMatrix& other)
{
  int n = pattern_.n();
  if (other.n() != n) {
    fatal_error(fmt::format(
      "Cannot add CSC matrices with different dimensions ({} vs {})", n,
      other.n()));
  }

  // Fast path: if this matrix's pattern is a superset of the other's,
  // we can add values in-place without reallocating.
  if (pattern_ == other.pattern()) {
    for (int k = 0; k < nnz(); ++k) {
      data_[k] += other.data_[k];
    }
    return *this;
  }

  // General path: fall back to operator+
  *this = *this + other;
  return *this;
}

CSCMatrix operator*(double scalar, const CSCMatrix& mat)
{
  vector<double> new_data(mat.data_.size());
  for (size_t k = 0; k < mat.data_.size(); ++k) {
    new_data[k] = scalar * mat.data_[k];
  }
  // Copy the pattern (shares indptr/indices structure)
  CSCPattern new_pattern(
    mat.pattern_.n(),
    vector<int>(mat.pattern_.indptr()),
    vector<int>(mat.pattern_.indices()));
  return CSCMatrix(std::move(new_pattern), std::move(new_data));
}

void CSCMatrix::scale(double scalar)
{
  for (size_t k = 0; k < data_.size(); ++k) {
    data_[k] *= scalar;
  }
}

} // namespace openmc
