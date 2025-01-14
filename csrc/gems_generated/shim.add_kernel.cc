#include "shim.add_kernel.h"
#include <iostream>

namespace flaggems::shim {

int64_t AddKernelParams::godel_number() const {
  int64_t sum = 0;
  {
    int64_t number = 0;
    // if (x_ptr->dtype() == DType::kFloat16) number = 0;
    // if (x_ptr->dtype() == DType::kBFloat16) number = 1;
    // if (x_ptr->dtype() == DType::kFloat32) number = 2;
    sum += number * 1;
  }

  return sum;
}

cudaError_t AddKernelContext::launch(const AddKernelParams& params, cudaStream_t stream) {
  CUdeviceptr global_scratch = 0;
  const void* x_ptr_ptr = params.x_ptr->data_ptr();
  const void* y_ptr_ptr = params.y_ptr->data_ptr();
  const void* output_ptr_ptr = params.out_ptr->data_ptr();
  std::vector<void*> args = {
      const_cast<void*>(static_cast<const void*>(&x_ptr_ptr)),
      const_cast<void*>(static_cast<const void*>(&y_ptr_ptr)),
      const_cast<void*>(static_cast<const void*>(&output_ptr_ptr)),
      const_cast<void*>(static_cast<const void*>(&params.n_elements)),
      const_cast<void*>(static_cast<const void*>(&global_scratch)),
  };
  dim3 grid = grid_calculator(params);
  std::cout << "Before invoke a kernel" << std::endl;
  return params.selected_kernel->invoke("add_kernel", grid, args, stream);
}

int64_t AddKernelContext::get_arch_number(GpuArch arch) {
  if (arch == GPU_ARCH_NVIDIA_SM_80) return 0;
  return -1;
}

}  // namespace flaggems::shim
