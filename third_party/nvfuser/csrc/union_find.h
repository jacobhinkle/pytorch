#pragma once

#include <memory>
#include <unordered_map>
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
      this->setValue(i, vals[i]);
    }
  }

  UnionFind(std::unordered_set<T> vals) : UnionFind(vals.size()) {
    size_t i = 0;
    for (auto v : vals) {
      this->setValue(i++, v);
    }
  }

  void setValue(int pos, const T& val) {
    value_[pos] = val;
    val_to_pos_[val] = pos;
  }
  T getValue(int pos) {
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
  size_t insertValue(const T& val) {
    auto pos = parent_.size();
    parent_.push_back(pos);
    size_.push_back(1);
    val_to_pos_[val] = pos;
    value_.push_back(val);
    return pos + 1;
  }

  //! Find the integer position of val
  size_t getPosition(const T& val) {
    return val_to_pos_.at(val);
  }

  //! Get the integer index of the set from given position
  size_t findSet(size_t v) {
    if (v == parent_[v])
      return v;
    // Note that this step actually updates the tree to point directly to the
    // root index, meaning subsequent look-ups will not need to recurse.
    return parent_[v] = findSet(parent_[v]);
  }
  //! Get the integer index of the set for a given value
  size_t findSetFromValue(T val) {
    return findSet(getPosition(val));
  }

  //! Get all elements in the set with given index (up to O(n^2))
  std::vector<T> getSet(size_t idx) {
    std::vector<T> s;
    for (size_t i = 0; i < parent_.size(); ++i) {
      if (findSet(i) == idx) {
        s.push_back(value_.at(i));
      }
    }
    return s;
  }

  //! Get a vector of all sets of values
  std::vector<std::vector<T>> getSets() {
    std::vector<std::vector<T>> out;
    for (size_t i = 0; i < parent_.size(); ++i) {
      auto s = getSet(i);
      if (s.size() > 0) {
        out.push_back(s);
      }
    }
    return out;
  }

  //! Get a vector of set indexes
  std::vector<size_t> getSetIndices() {
    std::vector<size_t> ids;
    for (size_t i = 0; i < parent_.size(); ++i) {
      if (parent_[i] == i) {
        ids.push_back(i);
      }
    }
    return ids;
  }

  //! Merge two sets in the partition
  void mergeSets(size_t a, size_t b) {
    if (a != b) {
      if (size_[a] < size_[b])
        std::swap(a, b);
      parent_[b] = a;
      size_[a] += size_[b];
    }
  }
  //! Merge the sets containing two given values
  void mergeSetsFromValues(T val_a, T val_b) {
    auto a = findSet(getPosition(val_a));
    auto b = findSet(getPosition(val_b));
    mergeSets(a, b);
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
