#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <arith.h>
#include <compute_at_map.h>
#include <executor.h>
#include <inlining.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <scheduler/all_schedulers.h>
#include <id_e_graph.h>

#include <test/cpp/jit/test_utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

#include <torch/torch.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;


// Play with forming equivalence classes of IterDomains
TEST_F(NVFuserTest, FusionIDEGraph) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // [w]
  //auto tv0 = makeSymbolicTensor(1);
  auto tv0 = makeConcreteTensor({5});
  fusion.addInput(tv0);

  // [w, x]
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  // [w, y]
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = set(tv0);
  // [w]
  auto tv4 = broadcast(tv3, {false, true});
  // [w, 1]
  auto tv5 = add(tv4, tv2);
  // [w, x]
  fusion.addOutput(tv5);

  // [w]
  auto tv6 = broadcast(tv3, {false, true});
  // [w, 1]
  auto tv7 = add(tv6, tv2);
  // [y]
  fusion.addOutput(tv7);

  auto tv8 = sum(tv1, {0});
  auto tv9 = add(tv8, tv0);
  fusion.addOutput(tv9);

  fusion.printMath();
  //fusion.print();

  IterDomainEGraph eg(fusion);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
