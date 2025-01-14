#include <stdint.h>
#include <functional>
#include <string_view>

// #include "dtypes.h"
// #include "runtime.h"

namespace flaggems {

constexpr uint64_t CAT(uint32_t high, uint32_t low) {
  uint64_t high64 = high;
  uint64_t low64 = low;
  return (high64 << 32) | low64;
}

template <typename T>
T cdiv(T numerator, T denominator) {
  return (numerator + (denominator - 1)) / denominator;
}

// Use PCI IDs to avoid allocating numbers by ourselves
enum GpuVendor : uint32_t {
  kAMD = 0x1002,
  kNVIDIA = 0x10de,
  kINTEL = 0x8086,
};

// More bits for potential non-PCI architectures
enum GpuArch : uint64_t {
  GPU_ARCH_UNKNOWN = 0,
  GPU_ARCH_NVIDIA_SM_80 = CAT(GpuVendor::kNVIDIA, 0x90a),
};

GpuArch getArchFromStream(cudaStream_t);

}  // namespace flaggems
