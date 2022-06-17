/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file contains nccl api for alpa to use. 

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_

#include <memory>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif

#include "pybind11/pybind11.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"

#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_client.h"

namespace xla {
namespace gpu {

Status ncclResultToStatus(ncclResult_t s, const char* file, int64_t line,
                          const char* expr);

#define XLA_CUDA_STATUS(expr) \
  xla::gpu::ncclResultToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_CUDA_RETURN_IF_ERROR(expr) \
  do {                                 \
    Status s = XLA_CUDA_STATUS(expr);  \
    if (!s.ok()) {                     \
      return s;                        \
    }                                  \
  } while (0)

#define XLA_CUDA_WARN_IF_ERROR(expr)  \
  do {                                \
    Status s = XLA_CUDA_STATUS(expr); \
    if (!s.ok()) {                    \
      LOG(ERROR) << s.ToString();     \
    }                                 \
  } while (0)

ncclDataType_t ToNcclDataType(PrimitiveType element_type);

int SizeOfType(ncclDataType_t element_type);

tensorflow::StatusOr< std::shared_ptr< std::vector<ncclComm_t> > > NcclInitCommunicator(std::vector<int> devices_vec);

tensorflow::Status NcclLocalAllGather(std::vector<ncclComm_t> comms, 
                                      std::vector<PyBuffer::object> buffers, 
                                      std::vector<uint> local_start_positions, 
                                      uint global_start, 
                                      uint n_elements);

tensorflow::Status NcclDestroyComms(std::vector<ncclComm_t> comms);

tensorflow::Status NcclBroadcastPartialGPUs(std::vector<ncclComm_t> comms, 
                                            std::vector<PyBuffer::object> buffers, 
                                            std::vector<uint> local_start_positions, 
                                            uint n_elements, 
                                            int root_rank);

tensorflow::Status NcclSend(std::vector<ncclComm_t> comms, 
                            PyBuffer::object buffer, 
                            uint start, 
                            uint n_elements, 
                            int peer_p2p_rank);

tensorflow::Status NcclRecv(std::vector<ncclComm_t> comms,
                            PyBuffer::object buffer,
                            uint start,
                            uint n_elements,
                            int peer_p2p_rank);

std::vector<char> NcclUidSerialize(ncclUniqueId nccl_uid);

ncclUniqueId NcclUidDeserialize(std::vector<char> nccl_uid_chars);

tensorflow::StatusOr< std::vector<char> > NcclGetUniqueId();

tensorflow::StatusOr<int> NcclGetVersion();

tensorflow::StatusOr< std::shared_ptr< std::vector<ncclComm_t> > > NcclCreateCommunicators(int world_size,
                                                                                           std::vector<int> devices_global_rank,
                                                                                           std::vector<int> devices_ids,
                                                                                           std::vector<char> nccl_uid);

tensorflow::StatusOr<int> GetBufferDeviceId(PyBuffer::object buffer);


}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_
