#include "../gems_generated/shim.add_kernel.h"

namespace flaggems::impl {
cudaError_t add_out_impl(at::Tensor& out,
                         const at::Tensor& self,
                         const at::Tensor& other,
                         const at::Scalar& alpha);
}  // namespace flaggems::impl
namespace flaggems::kernels {

cudaError_t add_kernel(const at::Tensor& x,
                       const at::Tensor& y,
                       at::Tensor& out,
                       int32_t n_elements,
                       flaggems::Stream stream_wrap = Stream());

}  // namespace flaggems::kernels
