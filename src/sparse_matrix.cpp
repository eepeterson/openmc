//! \file sparse_matrix.cpp
//! \brief Implementation of CSCPattern, CSCMatrix, and ComplexCSCMatrix

#include "openmc/sparse_matrix.h"

#include <algorithm>  // for sort, fill
#include <complex>    // for complex
#include <functional> // for plus
#include <numeric>    // for iota
#include <utility>    // for pair

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
  return n_ == other.n_ && indptr_ == other.indptr_;
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

//==============================================================================
// ComplexCSCMatrix implementation
//==============================================================================

ComplexCSCMatrix ComplexCSCMatrix::from_real(
  const CSCPattern& target_pattern, const CSCMatrix& A, double scale)
{
  int n = target_pattern.n();
  const auto& t_indptr = target_pattern.indptr();
  const auto& t_indices = target_pattern.indices();
  const auto& a_indptr = A.indptr();
  const auto& a_indices = A.indices();
  const auto& a_data = A.data();

  // Allocate zero-initialized complex data
  vector<std::complex<double>> data(target_pattern.nnz(), {0.0, 0.0});

  // Copy scaled real values into matching positions
  for (int col = 0; col < n; ++col) {
    int t_pos = t_indptr[col];
    int t_end = t_indptr[col + 1];
    int a_pos = a_indptr[col];
    int a_end = a_indptr[col + 1];

    // Merge: target rows are sorted, A rows are sorted
    while (t_pos < t_end && a_pos < a_end) {
      if (t_indices[t_pos] == a_indices[a_pos]) {
        data[t_pos] = scale * a_data[a_pos];
        ++t_pos;
        ++a_pos;
      } else if (t_indices[t_pos] < a_indices[a_pos]) {
        ++t_pos;
      } else {
        ++a_pos;
      }
    }
  }

  // Copy the pattern (target_pattern is a superset of A's pattern)
  CSCPattern pattern_copy(n, vector<int>(t_indptr.begin(), t_indptr.end()),
    vector<int>(t_indices.begin(), t_indices.end()));

  return ComplexCSCMatrix(std::move(pattern_copy), std::move(data));
}

} // namespace openmc
