// ============================ CODE GEN ============================

#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Logging.h>
#include <functional>
#include <string>
// #include <aotriton/dtypes.h>
// #include <aotriton/flash.h>
// #include <aotriton/pointwise.h>
#include "../include/_internal/triton_kernel.h"
#include "../include/runtime.h"
#include "../utils/device_utils.h"

namespace flaggems::shim {

struct AddKernelParams {
  // Function related arguments
  const at::Tensor* x_ptr;
  const at::Tensor* y_ptr;
  const at::Tensor* out_ptr;
  int32_t n_elements;
  // Performance related arguments for current selection
  int32_t BLOCK_SIZE;

  // TritonKernel* selected_kernel = nullptr;
  TritonKernel selected_kernel_ = TritonKernel("", "");
  TritonKernel* selected_kernel = &selected_kernel_;
  const char* _debug_kernel_name = nullptr;
  int64_t godel_number() const;
};

class AddKernelContext {
 public:
  std::function<dim3(const AddKernelParams&)> grid_calculator;

  cudaError_t launch(const AddKernelParams& params, cudaStream_t stream);
  static int64_t get_arch_number(GpuArch arch);

 private:
  GpuArch kernel_arch = GPU_ARCH_UNKNOWN;
};

}  // namespace flaggems::shim
