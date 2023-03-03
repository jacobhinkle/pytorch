#pragma once

#include <ir_all_nodes.h>
#include <union_find.h>

#include <iostream>
#include <memory>
#include <unordered_set>
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
template <typename T>
class Relation {
 public:
  Relation(T left_id, T right_id) : left_id_(left_id), right_id_(right_id){};
  virtual ~Relation() {}
  virtual RelationType type() const = 0;

 protected:
  T left_id_, right_id_;
};

//! Left domain is inner to right domain in at least one input TensorView
template <typename T>
class InnerToInInputRelation : Relation<T> {
  RelationType type() const {
    return RelationType::InnerToInInput;
  }
};

//! Left domain is inner to right domain in at least one output TensorView
template <typename T>
class InnerToInOutputRelation : Relation<T> {
  RelationType type() const {
    return RelationType::InnerToInOutput;
  }
};

//! Left domain resolves the right (broadcast) domain
template <typename T>
class ResolvesBroadcastRelation : Relation<T> {
  RelationType type() const {
    return RelationType::ResolvesBroadcast;
  }
};

//! Left domain is a reduction domain over right domain
template <typename T>
class ReducesRelation : Relation<T> {
  RelationType type() const {
    return RelationType::Reduces;
  }
};

//! Implements an E-graph whose "terms" are IterDomains.
class TORCH_CUDA_CU_API IterDomainEGraph {
 public:
  IterDomainEGraph(const Fusion& fusion);

  //! Get all class IDs, which are not guaranteed to be a contiguous list of
  //! size_t's
  std::vector<size_t> getClassIds();

  //! Convert internal representation to relations over _classes_
  std::vector<Relation<size_t>*> getClassRelations();

  //! Print out a diagram in mermaid format
  //! https://mermaid.js.org
  //! https://mermaid.live
  void printMermaid(std::ostream& stream) {
    stream << "graph TD" << std::endl;
  }

 private:
  Fusion* fusion;
  std::vector<int> e_class_ids_;
  std::vector<Relation<IterDomain*>*> id_relations_;
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
