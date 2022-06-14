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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_

#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif

// #include "pybind11/pybind11.h"
// #include "pybind11/attr.h"
// #include "pybind11/cast.h"
// #include "pybind11/numpy.h"
// #include "pybind11/pytypes.h"
// #include "pybind11/stl_bind.h"

// #include "tensorflow/compiler/xla/python/py_buffer.h"
// #include "tensorflow/compiler/xla/python/py_client.h"

namespace xla {
namespace gpu {

// ncclDataType_t MyToNcclDataType(PrimitiveType element_type);

// int SizeOfType(ncclDataType_t element_type);

// std::shared_ptr< std::vector<ncclComm_t> > InitCommunicator(int n_devices, std::vector<int> devices_vec);

// void LocalAllGather(int n_devices, 
//                     std::vector<ncclComm_t> comms, 
//                     std::vector<PyBuffer::object> buffers, 
//                     std::vector<int> devices_ids, 
//                     uint n_elements);

// std::shared_ptr< std::vector<ncclComm_t> > nccl_InitCommunicator(int n_devices, std::vector<int> devices_vec);

// void nccl_LocalAllGather(int n_devices, 
//                          std::vector<ncclComm_t> comms, 
//                          std::vector<PyBuffer::object> buffers, 
//                          std::vector<int> devices_ids, 
//                          std::vector<uint> local_starts, 
//                          uint global_start, 
//                          uint n_elements);

// void nccl_DestroyComms(std::vector<ncclComm_t> comms);

// void nccl_BroadcastPartialGPUs(int n_devices, 
//                                std::vector<ncclComm_t> comms, 
//                                std::vector<PyBuffer::object> buffers, 
//                                std::vector<uint> local_start_positions, 
//                                uint n_elements, 
//                                int root_rank);

// void nccl_Send(std::vector<ncclComm_t> comms, 
//                PyBuffer::object buffer, 
//                uint start, 
//                uint n_elements, 
//                int peer_p2p_rank);

// void nccl_Recv(std::vector<ncclComm_t> comms, 
//                PyBuffer::object buffer, 
//                uint start, 
//                uint n_elements, 
//                int peer_p2p_rank);

// ncclUniqueId nccl_GetUniqueId();

// int nccl_GetVersion();

// std::shared_ptr< std::vector<ncclComm_t> > nccl_CreateCommunicators(int n_devices, 
//                                                                     int world_size, 
//                                                                     std::vector<int> devices_global_rank, 
//                                                                     std::vector<int> devices_ids, 
//                                                                     ncclUniqueId nccl_uid);

// int get_buffer_device_id(PyBuffer::object buffer);

ncclRedOp_t ToNcclReduction(ReductionKind kind);
StatusOr<std::pair<ncclDataType_t, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type);

bool IsGlobalNcclConfig();
bool IsNcclLaunchModeParallel();

Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                const char* expr);

// Macros to return or warn on CUDA/NCCL errors.  (The same macro works for both
// NCCL and CUDA errors.)
//
// It's tempting to say these macros belong in an XLA header somewhere, but in
// practice we don't do much direct-to-CUDA-API stuff outside of this file.
#define XLA_CUDA_STATUS(expr) \
  xla::gpu::ToStatus(expr, __FILE__, __LINE__, #expr)

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

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices);  // may be null

StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback,  // may be null
    bool is_local);

// Represents a type that requires mutually exclusive access.
template <typename T>
class Lockable {
 public:
  // RAII type that will release the exclusive lock when it is destroyed.
  using Lock = std::unique_ptr<T, std::function<void(T*)>>;

  explicit Lockable(T value = T()) : value_(std::move(value)) {}

  Lock Acquire() {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(&is_unlocked_));
    is_unlocked_ = false;

    return {&value_, [this](T*) {
              absl::MutexLock lock(&mutex_);
              CHECK(!is_unlocked_);
              is_unlocked_ = true;
            }};
  }

 private:
  T value_;
  absl::Mutex mutex_;
  bool is_unlocked_ ABSL_GUARDED_BY(mutex_) = true;
};

TF_LIB_GTL_DEFINE_INT_TYPE(OpId, int64_t);

struct NcclComm : public Lockable<ncclComm_t> {
  NcclComm() : Lockable(nullptr) {}
};

StatusOr<NcclComm::Lock> AcquireNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_
