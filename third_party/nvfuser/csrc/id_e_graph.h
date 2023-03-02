#pragma once

#include <ir_all_nodes.h>

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

//! Implements an equivalence class of IterDomains.
class TORCH_CUDA_CU_API IterDomainEClass {
 private:
  std::vector<IterDomainENode> e_nodes_;
};

//! Implements an E-graph whose "terms" are IterDomains.
class TORCH_CUDA_CU_API IterDomainEGraph {
 public:
   IterDomainEGraph(const Fusion& fusion);


 private:
   Fusion* fusion;
   std::vector<IterDomainEClass> e_classes_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
