#pragma once

#include <ir_all_nodes.h>
#include <union_find.h>

#include <memory>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API IterDomainENode {
 public:
  IterDomainENode() = default;

 private:
  std::weak_ptr<IterDomain> iter_domain_;
};

enum RelationType {
  ResolvesBroadcast,
  Reduces,
  InnerToInInput,
  InnerToInOutput
};

//! Abstract class for binary relations
class Relation {
 public:
  Relation(size_t left_id, size_t right_id)
      : left_id_(left_id), right_id_(right_id){};
  virtual ~Relation() {}
  virtual RelationType type() const = 0;

 protected:
  size_t left_id_, right_id_;
};

//! Left domain is inner to right domain in at least one input TensorView
class InnerToInInputRelation : Relation {
  RelationType type() const {
    return RelationType::InnerToInInput;
  }
};

//! Left domain is inner to right domain in at least one output TensorView
class InnerToInOutputRelation : Relation {
  RelationType type() const {
    return RelationType::InnerToInOutput;
  }
};

//! Left domain resolves the right (broadcast) domain
class ResolvesBroadcastRelation : Relation {
  RelationType type() const {
    return RelationType::ResolvesBroadcast;
  }
};

//! Left domain is a reduction domain over right domain
class ReducesRelation : Relation {
  RelationType type() const {
    return RelationType::Reduces;
  }
};

//! Implements an E-graph whose "terms" are IterDomains.
class TORCH_CUDA_CU_API IterDomainEGraph {
 public:
  IterDomainEGraph(const Fusion& fusion);

  //! Merge the sets representing two values, preserving relations from each
  void merge_sets_from_values(const Val* a, const Val* b);

 private:
  Fusion* fusion;
  std::vector<int> e_class_ids_;
  std::vector<Relation*> relations_;
  std::vector<IterDomain*> all_ids_;
  std::unordered_set<Val*>
      all_extents_; // Also track extents to find which are equivalent
  std::unique_ptr<UnionFind<IterDomain*>> id_partition_;
  std::unique_ptr<UnionFind<Val*>> extent_partition_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
