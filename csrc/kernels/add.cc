#include "add.h"
#include <iostream>

namespace flaggems {
namespace impl {
  cudaError_t add_out_impl(at::Tensor& out,
                           const at::Tensor& self,
                           const at::Tensor& other,
                           const at::Scalar& alpha) {
    std::cout << "add_out_impl" << std::endl;
    return kernels::add_kernel(self, other, out, self.numel(), Stream());
  }
}  // namespace impl
namespace kernels {

  cudaError_t add_kernel(const at::Tensor& x,
                         const at::Tensor& y,
                         at::Tensor& out,
                         int32_t n_elements,
                         flaggems::Stream stream_wrap) {
    shim::AddKernelContext context;
    auto stream = stream_wrap.native();
    std::cout << "add_kernel" << std::endl;

    context.grid_calculator = [](const shim::AddKernelParams& params) -> dim3 {
      std::cout << "grid_calculator" << std::endl;
      dim3 grid {
          // flaggems::cdiv<uint32_t>(params.x_ptr->size(0), params.BLOCK_SIZE),
          flaggems::cdiv<uint32_t>(params.x_ptr->size(0), 128),
          uint32_t(1),
          uint32_t(128),
          // uint32_t(params.BLOCK_SIZE),
      };
      std::cout << "Grid config : " << grid.x << " " << grid.y << " " << grid.z << std::endl;
      // VLOG(3) << "Grid config : " << grid.x << " " << grid.y << " " << grid.z;
      return grid;
    };

    shim::AddKernelParams params = {.x_ptr = &x, .y_ptr = &y, .out_ptr = &out, .n_elements = n_elements};
    cudaError_t err = context.launch(params, stream);
    return err;
  }

}  // namespace kernels
}  // namespace flaggems
