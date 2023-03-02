#include "id_e_graph.h"

#include <deque>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// A fixed-size tree-based union-find (aka disjoint-set) data structure using
// subtree sizes instead of ranks.
// cf. https://en.wikipedia.org/wiki/Disjoint-set_data_structure
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
    for (auto v: vals) {
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
      "Passed invalid position ", pos, " for UnionFind with ", value_.size(), " entries"
    );
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
    std::vector<std::vector<T> > out;
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

IterDomainEGraph::IterDomainEGraph(const Fusion& fusion) {
  // Initialize a partition of all IterDomains in the Fusion, initially
  // containing all singleton classes.
  std::vector<IterDomain*> all_ids;
  std::unordered_set<Val*> all_extents; // Also track extents to find which are equivalent
  for (auto v: fusion.vals()) {
    if (v->getValType() == ValType::IterDomain) {
      auto id = reinterpret_cast<IterDomain*>(v);
      all_ids.push_back(id);
      all_extents.insert(id->extent());
    }
  }
  UnionFind<IterDomain*> id_partition(all_ids);
  UnionFind<Val*> extent_partition(all_extents);

  for (auto expr: fusion.unordered_exprs()) {
    // Expressions fall into one of the following categories:
    //   Broadcast
    //   Reduction
    //   Reshape
    //   Permute
    //   Pointwise
    if (expr->isA<BroadcastOp>()) {
      std::cout << "Broadcast op: " << expr->toString() << std::endl;
      auto bop = reinterpret_cast<BroadcastOp*>(expr);
      auto bcast_flags = bop->getBroadcastDimFlags();
      auto inp = reinterpret_cast<TensorView*>(bop->in());
      auto outp = reinterpret_cast<TensorView*>(bop->out());
      auto indom = inp->domain()->noReductions();
      auto outdom = outp->domain()->noReductions();
      for (size_t i = 0, j = 0; i < bcast_flags.size(); ++i) {
        if (bcast_flags[i]) {
          // This output dim is a new bcast dimension
          continue;
        }
        auto idin = indom[j++];
        auto idout = outdom[i];
        id_partition.merge_sets_from_values(idin, idout);
        extent_partition.merge_sets_from_values(idin->extent(), idout->extent());
      }
    } else if (expr->isA<ReductionOp>()) {
      std::cout << "Reduction op: " << expr->toString() << std::endl;
      auto rop = reinterpret_cast<ReductionOp*>(expr);
      // Instead of flags, we just look at the output type to find reduction
      // IDs, which will tell us which axes are being reduced over
      auto inp = reinterpret_cast<TensorView*>(rop->in());
      auto outp = reinterpret_cast<TensorView*>(rop->out());
      auto indom = inp->domain()->domain();
      auto outdom = outp->domain()->domain();
      for (size_t i = 0; i < indom.size(); ++i) {
        auto idin = indom[i];
        auto idout = outdom[i];
        if (idout->isReduction()) {
          // Don't merge serial input with rdomain output
          // TODO: set REDUCES relation
          continue;
        }
        if (idin->isBroadcast()) {
          // TODO: set RESOLVES relation
          continue;
        }
        id_partition.merge_sets_from_values(idin, idout);
        extent_partition.merge_sets_from_values(idin->extent(), idout->extent());
      }
    } else if (expr->isA<TransposeOp>()) {
      auto top = reinterpret_cast<TransposeOp*>(expr);
      auto inp = reinterpret_cast<TensorView*>(top->in());
      auto outp = reinterpret_cast<TensorView*>(top->out());
      auto indom = inp->domain()->domain();
      auto outdom = outp->domain()->domain();
      auto n2o = top->new2old();
      for (size_t i = 0; i < outdom.size(); ++i) {
        auto oldi = n2o[i];
        auto idin = indom[oldi];
        auto idout = outdom[i];
        id_partition.merge_sets_from_values(idin, idout);
        extent_partition.merge_sets_from_values(idin->extent(), idout->extent());
      }
    } else if (expr->isA<ViewOp>()) {
      std::cout << "Skipping reshape op: " << expr->toString() << std::endl;
    //} else if (expr->isA<ResizeOp>()) { // pending Naoya's recent work
    // TODO: Work through the list of other ops:
      /*
     *FullOp # nothing to do
     *ARangeOp # nothing to do
     *EyeOp # nothing to do
     *UnaryOp # handled by else
     *BinaryOp # handled by else
     *TernaryOp # handled by else
      SelectOp
      IndexSelectOp
      TorchGatherOp
      RNGOp
     *ReductionOp
      GroupedReductionOp
      WelfordOp
      GroupedWelfordOp
      LoadStoreOp
      MmaOp
     *BroadcastOp
      SqueezeOp
     *TransposeOp
      ExpandOp
      ShiftOp
      GatherOp
      ViewAsScalar
      ViewOp
      Split
      Merge
      Swizzle2D
      */
    } else {
      // For pointwise Exprs (or ExpandOp), this is most clear:
      //   - We simply merge matching (same position, ignoring reduction
      //   domains) serial IterDomain classes with one another.
      //   - If the input IterDomain is a bcast and the output Iterdomain is a
      //   bcast, we merge their e-classes.
      //   - If the input is a bcast and the output is serial (resolution of
      //   the broadcast), then we do not merge.
      //   Instead, in this case we add a relation that the e-class of the
      //   output ID _resolves_ the e-class of the input ID.
      for (auto inp_val: expr->inputs()) {
        if (inp_val->getValType() != ValType::TensorView) {
          continue;
        }
        auto inp = (TensorView*)inp_val;
        auto indom = inp->domain()->noReductions();
        for (auto outp_val: expr->outputs()) {
          if (outp_val->getValType() != ValType::TensorView) {
            continue;
          }
          auto outp = (TensorView*)outp_val;
          // Reduction domains aren't preserved through pointwise ops, so ignore them
          // For each non-reduction ID in inp and outp
          auto outdom = outp->domain()->noReductions();
          TORCH_CHECK(
              indom.size() == outdom.size(),
              "Input and output noReductions domains must have equal length in pointwise op"
          );
          for (auto idin=indom.begin(), idout=outdom.begin();
              idin != indom.end() && idout != outdom.end();
              idin++, idout++) {
            TORCH_CHECK(
              !(*idout)->isBroadcast(),
              "Output IterDomains of pointwise ops should not be of broadcast type"
            );
            // TODO: check for broadcast IDs in input
            if ((*idin)->isBroadcast()) {
              continue;
            }
            id_partition.merge_sets_from_values(*idin, *idout);
            extent_partition.merge_sets_from_values((*idin)->extent(), (*idout)->extent());
          }
        }
      }
    }
  }
  std::cout << "Equivalence classes of IterDomains:" << std::endl;
  for (auto s: id_partition.get_sets()) {
    std::cout << "  ";
    for (auto id: s) {
      std::cout << id->toString() << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << "Equivalence classes of extents:" << std::endl;
  for (auto s: extent_partition.get_sets()) {
    std::cout << "  ";
    for (auto e: s) {
      std::cout << e->toString() << ", ";
    }
    std::cout << std::endl;
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
