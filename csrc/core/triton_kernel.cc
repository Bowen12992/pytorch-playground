#include "../include/_internal/triton_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/LazyNVRTC.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include "../include/runtime.h"

namespace flaggems {

TritonKernel::TritonKernel(const char* package_path, const char* stem_name)
    : package_path_(package_path), stem_name_(stem_name) {
}

cudaError_t TritonKernel::invoke(const char* kernel_name,
                                 dim3 grid,
                                 std::vector<void*>& args,
                                 cudaStream_t stream) {
  int device_id;
  GEMS_CUDA_CHECK_RETURN(cudaGetDevice(&device_id));
  // We need a function cache here.
  cudaFunction_t func = nullptr;
  cudaError_t err;
  std::cout << "Invoking TritonKernel " << this << " with kernel_name = \"" << kernel_name << '"'
            << std::endl;

  std::tie(func, err) = load_for_device(device_id, kernel_name);
  auto r = cuLaunchKernel(func,
                          grid.x,
                          grid.y,
                          grid.z,
                          block_.x,
                          block_.y,
                          block_.z,
                          shared_memory_size_,
                          stream,
                          args.data(),
                          nullptr);
  std::cout << "TritonKernel Res:" << static_cast<cudaError_t>((r)) << std::endl;
  return static_cast<cudaError_t>((r));
}

char* load_cubin(const char* file_path, size_t& size) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);  // 打开文件并定位到末尾
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return nullptr;
  }
  size = file.tellg();
  file.seekg(0, std::ios::beg);

  char* buffer = new char[size];
  file.read(buffer, size);
  if (!file) {
    std::cerr << "Failed to read file: " << file_path << std::endl;
    delete[] buffer;
    return nullptr;
  }
  return buffer;
}

std::tuple<cudaFunction_t, cudaError_t> TritonKernel::load_for_device(int device_id,
                                                                      const char* kernel_name) {
  if (!kernel_image_) {
    // return std::make_tuple(nullptr, cudaErrorInvalidValue);
  }
  CUmodule mod;
  cudaFunction_t func;

  const char* cubin_path =
      "/work/aotriton/build/src/pointwise/gpu_kernel_image.add_kernel/"
      "add_kernel-Sig-F__^fp32@16__P__32__CO__-Gpu-RTX3090.cubin";
  size_t cubin_size;
  char* kernel_image_ = load_cubin(cubin_path, cubin_size);
  std::cout << "kernel_image" << kernel_image_ << std::endl;

  // Can we use  `cuModuleLoadDataEx(&mod, kernel_image_, 5, opt, optval));` here?
  GEMS_CUDA_CHECK_RETURN(cuModuleLoadData(&mod, kernel_image_));
  GEMS_CUDA_CHECK_RETURN(cuModuleGetFunction(&func, mod, kernel_name));
  // if we have a function cache, add `(device_id, mod, func)` into it.
  return std::make_tuple(func, cudaSuccess);
}

}  // namespace flaggems
