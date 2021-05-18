#ifndef TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_

// #if GOOGLE_CUDA
// #include "third_party/gpus/cuda/include/cuda.h"
// #include "third_party/gpus/cuda/include/cuda_runtime_api.h"
// #endif
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

namespace xla{
// TODO: use transfer manager
void GPUSwapOut(
  cudaStream_t stream, 
  void **buffers, 
  const char *opaque, size_t opaque_len);
void GPUSwapIn(
  cudaStream_t stream, 
  void **buffers, 
  const char *opaque, size_t opaque_len);

XLA_REGISTER_CUSTOM_CALL_TARGET(GPUSwapOut, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET(GPUSwapIn, "CUDA");
};  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_