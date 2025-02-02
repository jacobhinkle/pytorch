#pragma once
#include <executor_params.h>
#include <executor_utils.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_cloner.h>
#include <ir_printer.h>
#include <lower2device.h>
#include <utils.h>

#include <c10/core/DeviceType.h>

namespace nvfuser {

TORCH_CUDA_CU_API bool shouldFillAllocationWithNan();
TORCH_CUDA_CU_API void setFillAllocationWithNan(bool value);

// Note [Limitation of boundary assert]:
// When set to true we will add boundary check to the generated kernel's
// Tensor::operator[]. However, this does not always work and can have false
// positives and false negatives.
//
// False positive:
// For some cases, such as reduction, we generate code like
//  int index = 1025;
//  blockReduce(/*reference*/T0[index], ..., index < 1024);
// In the above example, we do not really read from T0[index], thanks to the
// predicate, however, the boundary check in operator[] is still executed.
// As a result, this would causes false alarm.
//
// False negative:
// Not all global memory accesses use operator[], for example, vectorized access
// uses loadGeneric on pointers. And this might miss cases like
//   int index = 1024;
//   loadGeneric<dtype, 4>(dest, &T0[index]); // T0.size[0] == 1026
TORCH_CUDA_CU_API bool shouldAssertOutOfBound();
TORCH_CUDA_CU_API void setAssertOutOfBound(bool value);

// TODO: Should this actually be in launch params?
struct TORCH_CUDA_CU_API CompileOptions {
  c10::Device device = c10::Device(c10::DeviceType::CUDA, 0);
  KernelIndexMode index_mode = KernelIndexMode::INT64;
};

class TORCH_CUDA_CU_API FusionExecutor : public NonCopyable {
 public:
  // Unsafe compilation that's useful for debugging kernels, iterating over
  // slight modifications of a generated kernel
  void debugCompileFusionFromStr(
      Fusion* fusion,
      const std::string& code,
      const std::string& name,
      int id,
      CompileOptions options = CompileOptions());

  //! infers output sizes via returning non-allocated KernelArgumentHolder.
  //! this function is useful for async compilation for segmented fusion
  KernelArgumentHolder inferOutputSizes(
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints);

  void compileFusion(
      Fusion* fusion,
      const KernelArgumentHolder& args,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams());

  // TODO: merge it with the overload above.
  //! This API is merely here so we don't have to go back and update all cpp
  //! tests.
  void compileFusion(
      Fusion* fusion,
      const at::ArrayRef<c10::IValue>& inputs = {},
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams()) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    compileFusion(fusion, args, launch_constraints, compile_params);
  }

  std::vector<at::Tensor> runFusion(
      KernelArgumentHolder& args,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const std::vector<at::Tensor>& outputs = {});

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<c10::IValue>& inputs,
      const std::vector<at::Tensor>& outputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const c10::optional<size_t>& opt_code = c10::nullopt) {
    KernelArgumentHolder args =
        KernelArgumentHolder::createKernelArgumentHolder(inputs);
    if (opt_code.has_value()) {
      args.setCacheId(*opt_code);
    }
    return runFusion(args, launch_constraints, compile_params, outputs);
  }

  std::vector<at::Tensor> runFusion(
      const at::ArrayRef<c10::IValue>& inputs,
      const LaunchParams& launch_constraints = LaunchParams(),
      CompileParams compile_params = CompileParams(),
      const c10::optional<size_t>& opt_code = c10::nullopt) {
    return runFusion(inputs, {}, launch_constraints, compile_params, opt_code);
  }

  // function to query whether a `FusionExecutor` has a compiled kernel to
  // execute
  bool compiled() const {
    return fusion_id_ != -1 && lowered_;
  };

  void evictCache(size_t cache_id) {
    executor_entry_lookup_.erase(cache_id);
  }

  // struct used to hold necessary information to launch compiled kernel on a
  // given input set.
  //
  // TODO: strides would also be important when we handle permutations in
  //       codegen.
  //
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct ExecutorEntry {
    bool init = false;
    LaunchParams launch_params;
    std::vector<std::pair<int, int>> io_alias_indices;
    std::vector<std::vector<int64_t>> output_sizes;
    std::vector<std::vector<int64_t>> output_strides;
    std::vector<at::ScalarType> output_types;
    std::vector<std::vector<int64_t>> buffer_sizes;
    std::vector<at::ScalarType> buffer_types;
    std::vector<bool> buffer_zero_init;
    uint64_t rand_offset;
  };

  using ExecutorCompileTimeInfoCache =
      executor_utils::caching::ExecutorCompileTimeInfoCache;

  kir::Kernel* kernel() const {
    TORCH_INTERNAL_ASSERT(lowered_);
    return lowered_->kernel();
  }

  //! Internal knob used for debugging/profiling only
  void setExecuteKernelFlag(bool execute_kernel) {
    execute_kernel_ = execute_kernel;
  }

  //! Internal knob used for debugging/profiling only
  void setMeasureKernelTimeFlag(bool measure_kernel_time) {
    measure_kernel_time_ = measure_kernel_time;
  }

  //! Internal knob used for debugging/profiling only
  void setSaveCompiledBinaryFlag(bool save_compiled_binary) {
    save_compiled_binary_ = save_compiled_binary;
  }

  //! Returns the last kernel execution time, in milliseconds
  //!
  //! \note The kernel time is only tracked if enabled by calling
  //!    setMeasureKernelTimeFlag(true)
  //!
  float kernelTimeMs() const {
    return measure_kernel_time_ ? kernel_time_ms_ : 0;
  }

  //! Returns the number of bytes processed last kernel execution
  int64_t bytesProcessed() const {
    return bytes_processed_;
  }

  //! Returns the launch parameters from the last kernel execution
  LaunchParams lastLaunchParams() const {
    return launch_params_;
  }

  //! Returns the string of the compiled kernel
  std::string kernelString() const {
    return kernel_code_;
  }

  //! Returns the latest compile log
  std::string compilerLog() const {
    return last_compiler_log_;
  }

  //! Returns the latest compiled binary
  std::vector<char> compiledBinary() const {
    return last_compiled_binary_;
  }

  //! Returns the disassembled latest compiled binary
  std::string disassembledBinary(const std::string& nvdisasm_args = "") const {
    return executor_utils::disassembleBinary(
        last_compiled_binary_, nvdisasm_args);
  }

  //! Returns the disassembled latest compiled binary
  std::string disassembledKernelSASS() const {
    return executor_utils::disassembleBinary(
        last_compiled_binary_, "-fun 1 -c");
  }

  std::string kernelName() const {
    std::stringstream ss;
    ss << "kernel" << fusion_id_;
    return ss.str();
  }

  //! Internal tests only. Compiles CUDA code with NVRTC directly from
  //! string. This util provides a path to test runtime code, i.e. the resource
  //! strings.
  void compileRtc(
      const std::string& code,
      const std::string& name,
      bool structured = false,
      CompileOptions options = CompileOptions());

  //! Internal tests only. Runs the compiled CUDA kernel from
  //! compileRtc. Return the elapsed milliseconds.
  float runRtc(
      const LaunchParams& launch_params,
      const std::vector<at::Tensor>& args,
      KernelIndexMode index_mode = KernelIndexMode::INT64);

  //! Internal knob used for debugging/profiling only
  void disableLaunchParamCache() {
    disable_parameter_cache_ = true;
  }

  //! Used in distributed setting where we only want to
  //!  allocate output space and receive output data from
  //!  a different rank instead of computing them.
  std::vector<at::Tensor> allocOutputSpace(
      const at::ArrayRef<c10::IValue>& inputs);

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  struct GlobalBuffers {
    std::vector<at::Tensor> buffers;
    std::vector<bool> zero_init;
    at::Tensor profile_buffer;
  };

  static std::string kernelNamespace() {
    return "CudaCodeGen";
  }

  // Add preamble and wrap in namespace
  std::string getStructuredCode(const std::string& kernel);

  LaunchParams computeLaunchParams(
      const LaunchParams& launch_constraints,
      ExpressionEvaluator& expr_eval,
      const int warp_size);

  uint64_t computeSharedMemory(
      ExpressionEvaluator& expr_eval,
      const std::vector<const kir::Allocate*>& buffers,
      bool align_padding = false,
      uint64_t total = 0);

  // return a pair of vector of tensors, where tensors in the first vector are
  // not initialized, while the second vector contains zero-initiliazed tensors
  GlobalBuffers allocGlobalVals(ExpressionEvaluator& expr_eval);

  // alias_index: index of outputs that are aliases to inputs, hence we should
  // skip allocating real storage for those, but still maintain its spot to
  // maintain the indexing from output aliases to inputs
  std::vector<at::Tensor> allocOutputs(
      const KernelArgumentHolder& args,
      ExpressionEvaluator& expr_eval,
      const std::unordered_set<int>& alias_indices = {});

  void setUsedTVs();

  const std::vector<TensorView*>& getUsedTVs() const {
    return used_tvs_;
  };

  ExecutorCompileTimeInfoCache* compileTimeDataCache() {
    return &compile_time_info_cache_;
  }

  //! returns KernelArgumentHolder representing the output sizes from kernel
  //! execution. Note: 1. this API would ignoring aliased outputs and instead
  //! pushing scalar int 0 as a place holder; 2. this API doesn't actually
  //! allocate output in memory, but rather is used just to infer output sizes.
  KernelArgumentHolder evaluateOutputSizes(
      const KernelArgumentHolder& args,
      ExpressionEvaluator& expr_eval,
      const std::unordered_set<int>& alias_indices = {});

 private:
  CompileOptions options_;

  //! Current configured total shared mem size from cudaDeviceProp
  size_t configured_device_smem_ = std::numeric_limits<size_t>().max();

  //! Available shared memory space for dynamic allocation for the current
  //!  compiled kernel at the current shared memory/L1 configuration
  c10::optional<size_t> maybe_available_dynamic_smem_ = c10::nullopt;

  //! Absolute limit of all available shared mem space from cudaDeviceProp
  size_t device_smem_limit_ = std::numeric_limits<size_t>().max();

  // Assuming sm70 or above:
  //  limit of statically allocated smem is 48 KB:
  // See:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
  const uint64_t max_static_smem_ = 48 << 10;
  int warp_size_ = 0;
  executor_utils::NvrtcFunction compiled_kernel_;

  // TensorViews actually used in the kernel.
  std::vector<TensorView*> used_tvs_;

  // Counter to be used for kernel name.
  int fusion_id_ = -1;
  static int fusion_id_counter_;

  std::unique_ptr<GpuLower> lowered_;
  // Copy of lowered_->kernel()
  Fusion* fusion_ = nullptr;

  // Track the block size this kernel was compiled with. If the block size
  // increases, recompile to adjust maxregister count.
  int64_t block_size_high_water_mark = 1;
  int maxrregcount_high_water_mark = 255;

  // lookup table to take short cut to retrieve recorded information in order to
  // launch kernels without re-inference parameters.
  std::unordered_map<size_t, ExecutorEntry> executor_entry_lookup_;

  // Compile time information caching. This is used for shape inference
  //  support. The cache stores graph information that are available
  //  without shape information so that each shape inference call will
  //  not need to re-compute them.
  ExecutorCompileTimeInfoCache compile_time_info_cache_;

  // Cached expr eval
  std::unique_ptr<PrecomputedValues> evaluator_precomputed_values_ = nullptr;

  // Profiling support: knob to control wheter we actually execute the
  // kernel on the GPU or not
  bool execute_kernel_ = true;

  // Profiling support: knob to enable measuring kernel execution time
  bool measure_kernel_time_ = false;

  // Profiling support: the last kernel execution time, if measure_kernel_time_
  // is true
  float kernel_time_ms_ = 0;

  // Profiling support: the last kernel Bytes processed
  int64_t bytes_processed_ = 0;

  // Profiling support: the last launch param used
  LaunchParams launch_params_;

  // Profiling support: disable caching of launch params and output allocation
  // output allocation is also disable when output sizes are dependent on
  // runtime scalar inputs, such as for the case of tensor factory. see
  // https://github.com/csarofeen/pytorch/issues/2002
  bool disable_parameter_cache_ = false;

  // Profiling support: kept copy of the cuda kernel
  std::string kernel_code_;

  // Profiling support: nvrtc log for debugging
  std::string last_compiler_log_;

  // save compiled binary
  bool save_compiled_binary_ = false;

  // nvrtc compiled binary
  std::vector<char> last_compiled_binary_;
};

} // namespace nvfuser
