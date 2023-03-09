#if defined(USE_CUDA)
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <arith.h>
#include <compute_at_map.h>
#include <executor.h>
#include <id_e_graph.h>
#include <inlining.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ops/alias.h>
#include <scheduler/all_schedulers.h>

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
  // auto tv0 = makeSymbolicTensor(1);
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
  // fusion.print();

  IterDomainEGraph eg(fusion);

  eg.printDot();
}

// Very simple graph with a broadcast and no reductions
TEST_F(NVFuserTest, FusionSimpleMulIDGraph) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({2});
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = mul(tv0, tv2);

  fusion.addOutput(tv3);

  fusion.printMath();
  // fusion.print();

  IterDomainEGraph eg(fusion);

  eg.printDot();
}

// Simple reshape example
TEST_F(NVFuserTest, FusionReshapeIDGraph) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 5, 12});
  fusion.addInput(tv0);

  auto tv1 = view(tv0, {2, 3, 5, 12}, {6, 5, 4, 3});

  fusion.addOutput(tv1);

  fusion.printMath();
  // fusion.print();

  IterDomainEGraph eg(fusion);

  eg.printDot();
}

// Gram matrix (inner product matrix) example
TEST_F(NVFuserTest, FusionGramMatrixIdGraph) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [n, d]
  auto tv0 = makeConcreteTensor({5, 7});
  fusion.addInput(tv0);

  // [1, n, d]
  auto tv1 = broadcast(tv0, {true, false, false});
  // [n, 1, d]
  auto tv2 = broadcast(tv0, {false, true, false});

  // [n, n, d]
  auto tv3 = mul(tv1, tv2);

  // [n, n]
  auto tv4 = sum(tv3, {2});

  fusion.addOutput(tv4);

  fusion.printMath();
  // fusion.print();

  IterDomainEGraph eg(fusion);

  eg.printDot();

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({5, 7}, options);

  auto reduction_params = getReductionHeuristics(&fusion, {aten_input});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  auto lparams = reduction_params->lparams;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input}, lparams);
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = fe.runFusion({aten_input}, lparams);
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
