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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_NCCL_WRAPPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_NCCL_WRAPPER_H_
// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif

#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/service/rendezvous.h"

namespace xla {
namespace gpu {
namespace alpa {
using AlpaNcclUid = std::vector<int8_t>;
using AlpaUuids = std::vector<int>;

class CommGroup {
 public:
  CommGroup(std::shared_ptr<PyClient> backend);
  // Communicator related functions:
  Status NcclCreateCommunicators(int world_size,
                                 const std::vector<int> &device_global_ranks,
                                 const std::vector<int> &device_ids,
                                 const AlpaNcclUid &nccl_uid_vec);

  Status NcclDestroyComms(const AlpaNcclUid &storage);

  // Communication operations:
  Status NcclLocalAllGather(const AlpaNcclUid &key,
                            std::vector<PyBuffer::object> buffers,
                            std::vector<uint> local_start_positions,
                            uint global_start, uint n_elements,
                            bool use_default_stream);

  Status NcclBroadcastPartialGPUs(const AlpaNcclUid &key,
                                  std::vector<PyBuffer::object> buffers,
                                  std::vector<uint> local_start_positions,
                                  uint n_elements, int root_rank,
                                  bool use_recv_stream,
                                  bool use_default_stream);

  Status NcclSend(const AlpaNcclUid &key, PyBuffer::object buffer, uint start,
                  uint n_elements, int peer_p2p_rank, bool use_default_stream);

  Status NcclRecv(const AlpaNcclUid &key, PyBuffer::object buffer, uint start,
                  uint n_elements, int peer_p2p_rank, bool use_default_stream);

  // Sync functions:
  Status CommunicatorRecordEvents(const AlpaUuids &uuids, int num_devices,
                                  bool is_send);

  Status CommunicatorWaitEvents(const AlpaUuids &uuids, int num_devices,
                                bool is_send);

  void CommWaitCompute(bool is_send, bool is_compute, int device_id);

  void ComputeWaitComm(bool is_send, bool is_compute, int device_id);

 private:
  std::vector<std::unique_ptr<se::Stream>> send_streams, recv_streams;
  ThreadSafeMap<std::pair<AlpaNcclUid, int>, NcclComm> comm_map;
  absl::flat_hash_map<AlpaNcclUid, std::vector<int>> local_ids;

  std::vector<se::StreamExecutor *> executors;
  PjRtStreamExecutorClient *client;
};

// We add them here rather than alpa_events to avoid circular deps in Bazel:
// alpa_events > done_event_thunk > executable > client >
// ComputationWaitEvents
Status ComputationWaitEvents(const AlpaUuids &uuids,
                             std::shared_ptr<PyClient> client);

// Event context management
void ResetEventContext(std::shared_ptr<PyClient> client);
// Other functions:
ncclUniqueId NcclUidDeserialize(const AlpaNcclUid& nccl_uid_chars);

StatusOr<AlpaNcclUid> NcclGetUniqueId();

StatusOr<int> NcclGetVersion();

StatusOr<int> GetBufferDeviceId(PyBuffer::object buffer);
}  // namespace alpa
}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_NCCL_WRAPPER_H_
