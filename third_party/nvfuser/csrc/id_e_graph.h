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

enum RelationType { ResolvesBroadcast, Reduces };

std::string printRelationType(RelationType r) {
  switch (r) {
    case RelationType::ResolvesBroadcast:
      return "resolvesBcast";
    case RelationType::Reduces:
      return "reduces";
  }
  return "";
}

class Relation {
 public:
  Relation(RelationType type, IterDomain* left, IterDomain* right)
      : type_(type), left_(left), right_(right){};

  IterDomain* getLeft() const {
    return left_;
  }
  IterDomain* getRight() const {
    return right_;
  }

  RelationType type() const {
    return type_;
  }

  std::string typeString() const {
    return printRelationType(type());
  }

 protected:
  RelationType type_;
  IterDomain *left_, *right_;
};

//! Implements an E-graph whose "terms" are IterDomains.
class TORCH_CUDA_CU_API IterDomainEGraph {
 public:
  IterDomainEGraph(Fusion& fusion) : fusion_(fusion) {
    initGraph();
  };

  void initGraph();

  //! Print out a diagram as a hierarchical graph in GraphViz's .dot format
  void printDot(std::ostream& stream = std::cout);

 private:
  Fusion& fusion_;
  std::vector<int> e_class_ids_;
  std::vector<Relation> id_relations_;
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
