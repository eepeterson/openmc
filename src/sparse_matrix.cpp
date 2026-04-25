//! \file sparse_matrix.cpp
//! \brief Implementation of CSCPattern and CSCMatrix

#include "openmc/sparse_matrix.h"

#include <algorithm>
#include <iterator>  // for back_inserter
#include <numeric>   // for iota
#include <stdexcept>

#include <fmt/core.h>

namespace openmc {

//==============================================================================
// CSCPattern implementation
//==============================================================================

CSCPattern::CSCPattern(int n, vector<int> indptr, vector<int> indices)
  : n_(n), indptr_(std::move(indptr)), indices_(std::move(indices))
{
  if (static_cast<int>(indptr_.size()) != n_ + 1) {
    throw std::invalid_argument {fmt::format(
      "CSCPattern: indptr size ({}) != n + 1 ({})", indptr_.size(), n_ + 1)};
  }
  if (indptr_[0] != 0) {
    throw std::invalid_argument {
      fmt::format("CSCPattern: indptr[0] ({}) != 0", indptr_[0])};
  }
  if (indptr_[n_] != static_cast<int>(indices_.size())) {
    throw std::invalid_argument {
      fmt::format("CSCPattern: indptr[n] ({}) != indices size ({})",
        indptr_[n_], indices_.size())};
  }

  for (int j = 0; j < n_; ++j) {
    for (int k = indptr_[j]; k < indptr_[j + 1]; ++k) {
      if (indices_[k] < 0 || indices_[k] >= n_) {
        throw std::invalid_argument {fmt::format(
          "CSCPattern: row index {} out of bounds [0, {}) in column {}",
          indices_[k], n_, j)};
      }
      if (k > indptr_[j] && indices_[k - 1] >= indices_[k]) {
        throw std::invalid_argument {
          fmt::format("CSCPattern: row indices not sorted in column {} "
                      "(indices[{}]={} >= indices[{}]={})",
            j, k - 1, indices_[k - 1], k, indices_[k])};
      }
    }
  }
}

//==============================================================================
// CSCMatrix implementation
//==============================================================================

CSCMatrix::CSCMatrix(CSCPattern pattern, vector<double> data)
  : pattern_(std::move(pattern)), data_(std::move(data))
{
  if (static_cast<int>(data_.size()) != pattern_.nnz()) {
    throw std::invalid_argument {
      fmt::format("CSCMatrix: data size ({}) != pattern nnz ({})",
        data_.size(), pattern_.nnz())};
  }
}

CSCMatrix::CSCMatrix(
  int n, vector<int> indptr, vector<int> indices, vector<double> data)
  : pattern_(n, std::move(indptr), std::move(indices)), data_(std::move(data))
{
  if (static_cast<int>(data_.size()) != pattern_.nnz()) {
    throw std::invalid_argument {
      fmt::format("CSCMatrix: data size ({}) != pattern nnz ({})",
        data_.size(), pattern_.nnz())};
  }
}

bool CSCPattern::operator==(const CSCPattern& other) const
{
  return n_ == other.n_ && indptr_ == other.indptr_ &&
         indices_ == other.indices_;
}

CSCPattern CSCPattern::with_diagonal() const
{
  // Single-pass merge: for each column, copy the existing sorted row indices
  // while inserting `col` in its sorted position if absent. The final nnz is
  // at most nnz() + n_, so we reserve that upper bound.
  vector<int> new_indptr(n_ + 1);
  vector<int> new_indices;
  new_indices.reserve(indices_.size() + n_);

  for (int col = 0; col < n_; ++col) {
    new_indptr[col] = static_cast<int>(new_indices.size());
    bool diag_written = false;
    for (int idx = indptr_[col]; idx < indptr_[col + 1]; ++idx) {
      int row = indices_[idx];
      if (!diag_written && row >= col) {
        new_indices.push_back(col);
        diag_written = true;
        if (row == col)
          continue; // don't duplicate an existing diagonal
      }
      new_indices.push_back(row);
    }
    if (!diag_written) {
      new_indices.push_back(col);
    }
  }
  new_indptr[n_] = static_cast<int>(new_indices.size());

  return CSCPattern(n_, std::move(new_indptr), std::move(new_indices));
}

//==============================================================================
// LU factorization helpers
//==============================================================================

SymbolicLUFactorization symbolic_factorize(CSCPattern pattern)
{
  int n = pattern.n();
  const auto& indptr = pattern.indptr();
  const auto& indices = pattern.indices();

  // Build L and U fill patterns column by column.
  vector<vector<int>> l_cols(n);
  vector<bool> marked(n, false);
  vector<int> u_work, l_work;

  // Temporary storage for U column patterns before CSC assembly.
  vector<vector<int>> u_cols(n);

  for (int j = 0; j < n; ++j) {
    u_work.clear();
    l_work.clear();

    // Scatter: mark off-diagonal structural entries in column j.
    for (int p = indptr[j]; p < indptr[j + 1]; ++p) {
      int i = indices[p];
      if (i == j)
        continue;
      if (!marked[i]) {
        marked[i] = true;
        if (i < j) {
          u_work.push_back(i);
        } else {
          l_work.push_back(i);
        }
      }
    }

    // Propagate fill through previously discovered L columns.
    for (size_t idx = 0; idx < u_work.size(); ++idx) {
      int k = u_work[idx];
      for (int row : l_cols[k]) {
        if (row == j)
          continue;
        if (!marked[row]) {
          marked[row] = true;
          if (row < j) {
            u_work.push_back(row);
          } else {
            l_work.push_back(row);
          }
        }
      }
    }

    std::sort(u_work.begin(), u_work.end());
    std::sort(l_work.begin(), l_work.end());
    u_cols[j] = u_work;
    l_cols[j] = l_work;

    for (int k : u_work)
      marked[k] = false;
    for (int i : l_work)
      marked[i] = false;
  }

  vector<int> l_indptr(n + 1);
  vector<int> l_rowidx;
  for (int j = 0; j < n; ++j) {
    l_indptr[j] = static_cast<int>(l_rowidx.size());
    for (int r : l_cols[j])
      l_rowidx.push_back(r);
  }
  l_indptr[n] = static_cast<int>(l_rowidx.size());

  // Store the diagonal as the last U entry in each column.
  vector<int> u_indptr(n + 1);
  vector<int> u_rowidx;
  for (int j = 0; j < n; ++j) {
    u_indptr[j] = static_cast<int>(u_rowidx.size());
    for (int r : u_cols[j])
      u_rowidx.push_back(r);
    u_rowidx.push_back(j);
  }
  u_indptr[n] = static_cast<int>(u_rowidx.size());

  return {std::move(pattern),
    CSCPattern(n, std::move(l_indptr), std::move(l_rowidx)),
    CSCPattern(n, std::move(u_indptr), std::move(u_rowidx))};
}

//==============================================================================
// CSCPattern: from_triplets, topological_sort, reachability
//==============================================================================

CSCPattern CSCPattern::from_triplets(
  int n, const vector<int>& rows, const vector<int>& cols)
{
  if (rows.size() != cols.size()) {
    throw std::invalid_argument {fmt::format(
      "CSCPattern::from_triplets: rows.size ({}) != cols.size ({})",
      rows.size(), cols.size())};
  }
  int nt = static_cast<int>(rows.size());

  // Sort triplets by (col, row).
  vector<int> order(nt);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return cols[a] < cols[b] || (cols[a] == cols[b] && rows[a] < rows[b]);
  });

  // Build CSC arrays, collapsing duplicate (row, col) pairs.
  vector<int> indptr(n + 1, 0);
  vector<int> indices;
  indices.reserve(nt);

  int prev_col = -1;
  int prev_row = -1;
  for (int k = 0; k < nt; ++k) {
    int i = rows[order[k]];
    int j = cols[order[k]];
    if (i < 0 || i >= n || j < 0 || j >= n) {
      throw std::invalid_argument {
        fmt::format("CSCPattern::from_triplets: index ({}, {}) out of "
                    "bounds for n={}", i, j, n)};
    }
    if (j == prev_col && i == prev_row)
      continue;

    indices.push_back(i);
    for (int c = prev_col + 1; c <= j; ++c) {
      indptr[c] = static_cast<int>(indices.size()) - 1;
    }
    prev_col = j;
    prev_row = i;
  }
  for (int c = prev_col + 1; c <= n; ++c) {
    indptr[c] = static_cast<int>(indices.size());
  }

  return CSCPattern(n, std::move(indptr), std::move(indices));
}

vector<int> CSCPattern::topological_sort() const
{
  // Build in-degree count from off-diagonal entries.
  // Graph edge: col -> row for each off-diagonal (row, col) entry.
  vector<int> in_degree(n_, 0);
  for (int col = 0; col < n_; ++col) {
    for (int p = indptr_[col]; p < indptr_[col + 1]; ++p) {
      int row = indices_[p];
      if (row != col) {
        ++in_degree[row];
      }
    }
  }

  // Use a min-heap of zero-in-degree nodes for deterministic ordering.
  vector<int> queue;
  for (int i = 0; i < n_; ++i) {
    if (in_degree[i] == 0) {
      queue.push_back(i);
    }
  }
  std::make_heap(queue.begin(), queue.end(), std::greater<int>());

  vector<int> perm;
  perm.reserve(n_);

  while (!queue.empty()) {
    std::pop_heap(queue.begin(), queue.end(), std::greater<int>());
    int node = queue.back();
    queue.pop_back();
    perm.push_back(node);

    for (int p = indptr_[node]; p < indptr_[node + 1]; ++p) {
      int row = indices_[p];
      if (row != node && --in_degree[row] == 0) {
        queue.push_back(row);
        std::push_heap(queue.begin(), queue.end(), std::greater<int>());
      }
    }
  }

  if (static_cast<int>(perm.size()) != n_) {
    throw std::invalid_argument {fmt::format(
      "CSCPattern::topological_sort: graph contains a cycle "
      "({} of {} nodes processed)", perm.size(), n_)};
  }
  return perm;
}

void CSCPattern::reachability(const vector<int>& perm,
  vector<int>& reach_indptr, vector<int>& reach_indices) const
{
  if (static_cast<int>(perm.size()) != n_) {
    throw std::invalid_argument {fmt::format(
      "CSCPattern::reachability: perm size ({}) != n ({})", perm.size(), n_)};
  }

  // Inverse permutation: inv_perm[old_idx] = new_idx.
  vector<int> inv_perm(n_);
  for (int i = 0; i < n_; ++i) {
    inv_perm[perm[i]] = i;
  }

  // Build off-diagonal lower-triangular column structure in permuted space.
  vector<int> lt_indptr(n_ + 1, 0);
  for (int old_col = 0; old_col < n_; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr_[old_col]; p < indptr_[old_col + 1]; ++p) {
      int new_row = inv_perm[indices_[p]];
      if (new_row != new_col) {
        ++lt_indptr[new_col + 1];
      }
    }
  }
  for (int j = 0; j < n_; ++j) {
    lt_indptr[j + 1] += lt_indptr[j];
  }

  int lt_nnz = lt_indptr[n_];
  vector<int> lt_rowidx(lt_nnz);
  vector<int> col_pos(n_, 0);
  for (int old_col = 0; old_col < n_; ++old_col) {
    int new_col = inv_perm[old_col];
    for (int p = indptr_[old_col]; p < indptr_[old_col + 1]; ++p) {
      int new_row = inv_perm[indices_[p]];
      if (new_row != new_col) {
        lt_rowidx[lt_indptr[new_col] + col_pos[new_col]++] = new_row;
      }
    }
  }
  for (int j = 0; j < n_; ++j) {
    std::sort(lt_rowidx.begin() + lt_indptr[j],
      lt_rowidx.begin() + lt_indptr[j + 1]);
  }

  // Memoized transitive closure walked leaves-first. reach[j] is the sorted
  // list of permuted indices reachable from j (excluding j itself).
  vector<vector<int>> reach(n_);
  vector<int> merge_buf;

  for (int j = n_ - 1; j >= 0; --j) {
    int n_children = lt_indptr[j + 1] - lt_indptr[j];
    if (n_children == 0)
      continue;

    if (n_children == 1) {
      int c = lt_rowidx[lt_indptr[j]];
      auto& rc = reach[c];
      reach[j].resize(1 + rc.size());
      auto it = std::lower_bound(rc.begin(), rc.end(), c);
      auto pos = it - rc.begin();
      std::copy(rc.begin(), it, reach[j].begin());
      reach[j][pos] = c;
      std::copy(it, rc.end(), reach[j].begin() + pos + 1);
    } else {
      int c0 = lt_rowidx[lt_indptr[j]];
      auto& rc0 = reach[c0];
      merge_buf.clear();
      merge_buf.reserve(rc0.size() + 1);
      auto it0 = std::lower_bound(rc0.begin(), rc0.end(), c0);
      merge_buf.insert(merge_buf.end(), rc0.begin(), it0);
      merge_buf.push_back(c0);
      merge_buf.insert(merge_buf.end(), it0, rc0.end());

      for (int lp = lt_indptr[j] + 1; lp < lt_indptr[j + 1]; ++lp) {
        int c = lt_rowidx[lp];
        auto& rc = reach[c];
        vector<int> child_set;
        child_set.reserve(rc.size() + 1);
        auto itc = std::lower_bound(rc.begin(), rc.end(), c);
        child_set.insert(child_set.end(), rc.begin(), itc);
        child_set.push_back(c);
        child_set.insert(child_set.end(), itc, rc.end());

        vector<int> merged;
        merged.reserve(merge_buf.size() + child_set.size());
        std::set_union(merge_buf.begin(), merge_buf.end(), child_set.begin(),
          child_set.end(), std::back_inserter(merged));
        merge_buf = std::move(merged);
      }

      reach[j] = std::move(merge_buf);
    }
  }

  // Flatten reach into CSC-like format.
  reach_indptr.assign(n_ + 1, 0);
  for (int j = 0; j < n_; ++j) {
    reach_indptr[j + 1] =
      reach_indptr[j] + static_cast<int>(reach[j].size());
  }
  reach_indices.resize(reach_indptr[n_]);
  for (int j = 0; j < n_; ++j) {
    std::copy(reach[j].begin(), reach[j].end(),
      reach_indices.begin() + reach_indptr[j]);
  }
}

//==============================================================================
// CSCMatrix::from_triplets
//==============================================================================

CSCMatrix CSCMatrix::from_triplets(int n, const vector<int>& rows,
  const vector<int>& cols, const vector<double>& vals)
{
  if (rows.size() != cols.size() || rows.size() != vals.size()) {
    throw std::invalid_argument {
      fmt::format("CSCMatrix::from_triplets: size mismatch (rows={}, "
                  "cols={}, vals={})",
        rows.size(), cols.size(), vals.size())};
  }
  int nt = static_cast<int>(rows.size());

  // Sort triplets by (col, row).
  vector<int> order(nt);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return cols[a] < cols[b] || (cols[a] == cols[b] && rows[a] < rows[b]);
  });

  // First pass: collect entries with duplicates summed.
  struct Entry {
    int row, col;
    double val;
  };
  vector<Entry> entries;
  entries.reserve(nt);

  int prev_col = -1;
  int prev_row = -1;
  for (int k = 0; k < nt; ++k) {
    int i = rows[order[k]];
    int j = cols[order[k]];
    double v = vals[order[k]];
    if (i < 0 || i >= n || j < 0 || j >= n) {
      throw std::invalid_argument {
        fmt::format("CSCMatrix::from_triplets: index ({}, {}) out of "
                    "bounds for n={}", i, j, n)};
    }
    if (j == prev_col && i == prev_row) {
      entries.back().val += v;
      continue;
    }
    entries.push_back({i, j, v});
    prev_col = j;
    prev_row = i;
  }

  // Second pass: emit only nonzero entries.
  vector<int> indptr(n + 1, 0);
  vector<int> indices;
  vector<double> data;
  indices.reserve(entries.size());
  data.reserve(entries.size());

  prev_col = -1;
  for (const auto& e : entries) {
    if (e.val == 0.0)
      continue;
    indices.push_back(e.row);
    data.push_back(e.val);
    for (int c = prev_col + 1; c <= e.col; ++c) {
      indptr[c] = static_cast<int>(indices.size()) - 1;
    }
    prev_col = e.col;
  }
  for (int c = prev_col + 1; c <= n; ++c) {
    indptr[c] = static_cast<int>(indices.size());
  }

  return CSCMatrix(
    CSCPattern(n, std::move(indptr), std::move(indices)), std::move(data));
}

} // namespace openmc
