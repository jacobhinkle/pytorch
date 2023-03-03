#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! A tree-based union-find (aka disjoint-set) data structure using ! subtree
//! sizes instead of ranks.
//! cf. https://en.wikipedia.org/wiki/Disjoint-set_data_structure
template <typename T>
class UnionFind {
 public:
  UnionFind(size_t size) {
    value_.resize(size);
    parent_.resize(size);
    size_.resize(size);
    // Initialize with all singletoons
    for (size_t i = 0; i < size; ++i) {
      parent_[i] = i;
      size_[i] = 1;
    }
  }

  UnionFind(std::vector<T> vals) : UnionFind(vals.size()) {
    for (size_t i = 0; i < vals.size(); ++i) {
      this->set_value(i, vals[i]);
    }
  }

  UnionFind(std::unordered_set<T> vals) : UnionFind(vals.size()) {
    size_t i = 0;
    for (auto v : vals) {
      this->set_value(i++, v);
    }
  }

  void set_value(int pos, const T& val) {
    value_[pos] = val;
    val_to_pos_[val] = pos;
  }
  T get_value(int pos) {
    TORCH_CHECK(
        pos < value_.size(),
        "Passed invalid position ",
        pos,
        " for UnionFind with ",
        value_.size(),
        " entries");
    return value_[pos];
  }

  //! Insert the given value and return the new number of elements
  size_t insert_value(const T& val) {
    auto pos = parent_.size();
    parent_.push_back(pos);
    size_.push_back(1);
    val_to_pos_[val] = pos;
    value_.push_back(val);
    return pos + 1;
  }

  //! Find the integer position of val
  size_t get_position(const T& val) {
    return val_to_pos_.at(val);
  }

  //! Get the integer index of the set from given position
  size_t find_set(size_t v) {
    if (v == parent_[v])
      return v;
    // Note that this step actually updates the tree to point directly to the
    // root index, meaning subsequent look-ups will not need to recurse.
    return parent_[v] = find_set(parent_[v]);
  }
  //! Get the integer index of the set for a given value
  size_t find_set_from_value(T val) {
    return find_set(get_position(val));
  }

  //! Get all elements in the set with given index (up to O(n^2))
  std::vector<T> get_set(size_t idx) {
    std::vector<T> s;
    for (size_t i = 0; i < parent_.size(); ++i) {
      if (find_set(i) == idx) {
        s.push_back(value_.at(i));
      }
    }
    return s;
  }

  //! Get a vector of all sets of values
  std::vector<std::vector<T>> get_sets() {
    std::vector<std::vector<T>> out;
    for (size_t i = 0; i < parent_.size(); ++i) {
      auto s = get_set(i);
      if (s.size() > 0) {
        out.push_back(s);
      }
    }
    return out;
  }

  //! Merge two sets in the partition
  void merge_sets(size_t a, size_t b) {
    if (a != b) {
      if (size_[a] < size_[b])
        std::swap(a, b);
      parent_[b] = a;
      size_[a] += size_[b];
    }
  }
  //! Merge the sets containing two given values
  void merge_sets_from_values(T val_a, T val_b) {
    auto a = find_set(get_position(val_a));
    auto b = find_set(get_position(val_b));
    merge_sets(a, b);
  }

 private:
  std::vector<T> value_;
  std::unordered_map<T, size_t> val_to_pos_;
  std::vector<size_t> parent_;
  std::vector<int> size_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
