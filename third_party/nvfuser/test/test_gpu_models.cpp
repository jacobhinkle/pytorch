#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <executor_params.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

// CrossEntropyLoss+backward from issue #2556
TEST_F(NVFuserTest, FusionCrossEntropyTurnaround_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t N=512, num_classes=32768;

  auto tv0 = makeContigConcreteTensor({N, num_classes}, DataType::Float);  // logits
  auto tv1 = makeContigConcreteTensor({N}, DataType::Int);  // true labels
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Forward
  auto tv2 = log_softmax(tv0, 1);
  auto tv1b = broadcast(tv1, {false, true});
  auto tv1e = expand(tv1b, {tv0->axis(0)->extent(), fusion->oneVal()});
  auto tv3 = torch_gather(tv2, 1, tv1e);
  auto tv4 = mean(tv3, {0, 1}, false);
  auto tv5 = neg(tv4);
  fusion->addOutput(tv5);  // the loss
  
  // Backward
  auto tv6 = exp(tv2);  // softmax
  // subtract a one-hot at Y[i]
  auto tv7 = iota(tv0->axis(1)->extent());
  auto tv8 = broadcast(tv7, {true, false});
  auto tv9 = broadcast(tv1, {false, true});
  auto tv10 = eq(tv8, tv9);
  auto tv11 = where(tv10, castOp(DataType::Float, fusion->oneVal()), fusion->zeroVal()); // one-hot
  auto tv12 = sub(tv6, tv11);
  auto tv13 = mul(tv12, reciprocal(tv0->axis(0)->extent()));  // account for averaging over N in forward
  fusion->addOutput(tv13);  // gradient wrt tv0

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({N, num_classes}, options);
  at::Tensor t1 = at::randint(num_classes, {N}, options.dtype(at::kLong));

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  t0.detach_();
  t0.set_requires_grad(true);
  auto l = torch::cross_entropy_loss(t0, t1);
  l.backward();

  testValidate(executor_cache.fusion(), cg_outputs, {t0, t1}, {l, t0.grad()}, __LINE__, __FILE__);

}

// Test file size should be up to 10K LoC. Create a new file for more tests.

} // namespace nvfuser

