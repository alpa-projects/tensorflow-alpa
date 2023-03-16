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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_NCCL_GROUP_BASE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_NCCL_GROUP_BASE_H_
// Common place for all collective thunks to include nccl/rccl headers.
#if TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#else
#include "third_party/nccl/nccl.h"
#endif

#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/service/rendezvous.h"

namespace xla {
namespace gpu {
namespace alpa {
using AlpaNcclUid = std::vector<int8_t>;
using AlpaUuids = std::vector<int>;

class CommGroup {
 public:
  CommGroup(PjRtStreamExecutorClient *client);
  // Communicator related functions:
  Status NcclCreateCommunicators(int world_size,
                                 const std::vector<int> &device_global_ranks,
                                 const std::vector<int> &device_ids,
                                 const AlpaNcclUid &nccl_uid_vec);

  Status NcclDestroyComms(const AlpaNcclUid &storage);

  // Communication operations:
  Status NcclLocalAllGatherImpl(const AlpaNcclUid &key,
                                std::vector<PjRtBuffer *> buffers,
                                std::vector<uint> local_start_positions,
                                uint global_start, uint n_elements,
                                bool use_default_stream);

  Status NcclBroadcastPartialGPUsImpl(const AlpaNcclUid &key,
                                      std::vector<PjRtBuffer *> buffers,
                                      std::vector<uint> local_start_positions,
                                      uint n_elements, int root_rank,
                                      bool use_recv_stream,
                                      bool use_default_stream);

  Status NcclSendImpl(const AlpaNcclUid &key, PjRtBuffer *buffer, uint start,
                      uint n_elements, int peer_p2p_rank,
                      bool use_default_stream);

  Status NcclRecvImpl(const AlpaNcclUid &key, PjRtBuffer *buffer, uint start,
                      uint n_elements, int peer_p2p_rank,
                      bool use_default_stream);

  // Other functions
  NcclComm::Lock AcquireComm(const AlpaNcclUid &uuids, int device_id);

 protected:
  std::vector<std::unique_ptr<se::Stream>> send_streams, recv_streams;
  absl::flat_hash_map<AlpaNcclUid, std::vector<int>> local_ids;
  std::vector<se::StreamExecutor *> executors;
  PjRtStreamExecutorClient *client_;

 private:
  ThreadSafeMap<std::pair<AlpaNcclUid, int>, NcclComm> comm_map;
};

// Cross-mesh allreduce thunk related
void SetCommGroup(std::string key, std::shared_ptr<CommGroup> g,
                  const AlpaNcclUid &uid);

NcclComm::Lock GetCommunicator(std::string key, size_t device_id);

// Other functions
ncclUniqueId NcclUidDeserialize(const AlpaNcclUid &nccl_uid_chars);

StatusOr<AlpaNcclUid> NcclGetUniqueId();

StatusOr<int> NcclGetVersion();
}  // namespace alpa
}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_NCCL_GROUP_BASE_H_
