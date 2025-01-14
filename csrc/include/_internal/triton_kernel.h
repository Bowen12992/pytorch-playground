#pragma once

#include <memory>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../runtime.h"

#define TO_STRING(s) TO_STRING_I(s)
#define TO_STRING_I(s) #s

#define GEMS_CUDA_CHECK_RETURN(expr)                                    \
  do {                                                                  \
    auto r = static_cast<cudaError_t>((expr));                          \
    if (r != static_cast<cudaError_t>(cudaSuccess))                     \
      throw std::runtime_error("FAILURE at Line " TO_STRING(__LINE__)); \
  } while (0)

namespace flaggems {

class TritonKernel {
 public:
  TritonKernel(const char* package_path, const char* stem_name);

  cudaError_t invoke(const char* kernel_name, dim3 grid, std::vector<void*>& args, cudaStream_t stream);

 private:
  std::tuple<cudaFunction_t, cudaError_t> load_for_device(int device_id, const char* kernel_name);

  const char* package_path_ = nullptr;
  const char* stem_name_ = nullptr;

  int shared_memory_size_ = 0;
  dim3 block_ {256, 1, 1};
  const void* kernel_image_ = nullptr;
  bool kernel_loaded_ = false;
};

}  // namespace flaggems
