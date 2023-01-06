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

// This file implements nccl apis for alpa to use.
#include "tensorflow/compiler/xla/service/gpu/alpa_nccl_group_base.h"

#ifdef XLA_ENABLE_XCCL
#include "tensorflow/compiler/xla/service/gpu/alpa_events.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace stream_executor {};
namespace se = ::stream_executor;

namespace xla {
namespace gpu {
namespace alpa {
namespace {
StatusOr<std::uintptr_t> ToUnsafePointer(PjRtBuffer *buf) {
  return buf->client()->UnsafeBufferPointer(buf);
}

StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type) {
  // FIXME(yonghao): throw an error for other cases
  switch (element_type) {
    case S8:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return Unimplemented("Nccl does not support the type.");
  }
}

int SizeOfType(ncclDataType_t element_type) {
  switch (element_type) {
    case ncclInt8:
      return 1;
    case ncclUint8:
      return 1;
    case ncclInt32:
      return 4;
    case ncclUint32:
      return 4;
    case ncclInt64:
      return 8;
    case ncclUint64:
      return 8;
    case ncclFloat16:
      return 2;
    case ncclFloat32:
      return 4;
    case ncclFloat64:
      return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      return 2;
#endif
    default:
      return 4;
  }
}

CUstream GetCudaStream(se::Stream *stream) {
  return reinterpret_cast<CUstream>(se::gpu::AsGpuStreamValue(stream));
}

using CrossMeshCommInfo = std::pair<std::shared_ptr<CommGroup>, AlpaNcclUid>;
absl::flat_hash_map<std::string, CrossMeshCommInfo> cross_mesh_comms;
CUstream default_stream = NULL;
};  // namespace

CommGroup::CommGroup(PjRtStreamExecutorClient *client) : client_(client) {
  if (client != nullptr) {
    for (int device_id = 0; device_id < client->device_count(); ++device_id) {
      auto executor = client->device_state(device_id).executor();
      auto i_stream = std::make_unique<se::Stream>(executor);
      auto o_stream = std::make_unique<se::Stream>(executor);
      i_stream->Init();
      o_stream->Init();
      recv_streams.emplace_back(std::move(i_stream));
      send_streams.emplace_back(std::move(o_stream));
      executors.push_back(executor);
    }
  }
}

// Communicator related functions:
Status CommGroup::NcclCreateCommunicators(
    int world_size, const std::vector<int> &device_global_ranks,
    const std::vector<int> &device_ids, const AlpaNcclUid &nccl_uid_vec) {
#if XLA_ENABLE_XCCL
  int n_devices = device_global_ranks.size();
  CHECK_EQ(n_devices, device_ids.size());
  ncclUniqueId nccl_uid = NcclUidDeserialize(nccl_uid_vec);
  // Create Communicators
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < n_devices; i++) {
    cudaSetDevice(device_ids[i]);
    int rank = device_global_ranks[i];
    auto comm_key = std::make_pair(nccl_uid_vec, device_ids[i]);
    NcclComm::Lock comm = comm_map[comm_key].Acquire();
    XLA_CUDA_RETURN_IF_ERROR(
        ncclCommInitRank(comm.get(), world_size, nccl_uid, rank));
  }
  local_ids[nccl_uid_vec] = device_ids;
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status CommGroup::NcclDestroyComms(const AlpaNcclUid &nccl_uid_vec) {
#if XLA_ENABLE_XCCL
  for (int device_id : local_ids[nccl_uid_vec]) {
    auto key = std::make_pair(nccl_uid_vec, device_id);
    XLA_CUDA_RETURN_IF_ERROR(ncclCommDestroy(*comm_map[key].Acquire()));
  }
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

// Communication operation related functions:
// FIXME: local allgather is deprecated
Status CommGroup::NcclLocalAllGatherImpl(
    const AlpaNcclUid &key, std::vector<PjRtBuffer *> buffers,
    std::vector<uint> local_start_positions, uint global_start, uint n_elements,
    bool use_default_stream) {
#if XLA_ENABLE_XCCL
  const auto &device_ids = local_ids[key];
  int n_devices = device_ids.size();
  CHECK_EQ(n_devices, buffers.size());
  CHECK_EQ(n_devices, local_start_positions.size());

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t dtype,
      ToNcclDataType(buffers[0]->on_device_shape().element_type()));
  int dtype_size = SizeOfType(dtype);
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < n_devices; ++i) {
    // FIXME(yonghao): use assign or return
    TF_ASSIGN_OR_RETURN(std::uintptr_t sendbuff, ToUnsafePointer(buffers[i]));
    sendbuff = sendbuff + local_start_positions[i] * dtype_size;
    TF_ASSIGN_OR_RETURN(std::uintptr_t recvbuff, ToUnsafePointer(buffers[i]));
    recvbuff = recvbuff + global_start * dtype_size;
    auto comm = *comm_map[std::make_pair(key, device_ids[i])].Acquire();
    auto stream = (use_default_stream ? default_stream
                                      : GetCudaStream(recv_streams[i].get()));
    XLA_CUDA_RETURN_IF_ERROR(ncclAllGather((void *)sendbuff, (void *)recvbuff,
                                           n_elements, dtype, comm, stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status CommGroup::NcclBroadcastPartialGPUsImpl(
    const AlpaNcclUid &key, std::vector<PjRtBuffer *> buffers,
    std::vector<uint> local_start_positions, uint n_elements, int root_rank,
    bool use_recv_stream, bool use_default_stream) {
#if XLA_ENABLE_XCCL
  const auto &device_ids = local_ids[key];
  int n_devices = device_ids.size();
  CHECK_EQ(n_devices, buffers.size());
  CHECK_EQ(n_devices, local_start_positions.size());

  TF_ASSIGN_OR_RETURN(
      ncclDataType_t dtype,
      ToNcclDataType(buffers[0]->on_device_shape().element_type()));
  int dtype_size = SizeOfType(dtype);

  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < n_devices; ++i) {
    int device_id = device_ids[i];
    TF_ASSIGN_OR_RETURN(std::uintptr_t sendbuff, ToUnsafePointer(buffers[i]));
    sendbuff = sendbuff + local_start_positions[i] * dtype_size;
    TF_ASSIGN_OR_RETURN(std::uintptr_t recvbuff, ToUnsafePointer(buffers[i]));
    recvbuff = recvbuff + local_start_positions[i] * dtype_size;

    auto comm = *comm_map[std::make_pair(key, device_id)].Acquire();
    auto se_stream = use_default_stream
                         ? nullptr
                         : use_recv_stream ? recv_streams[device_id].get()
                                           : send_streams[device_id].get();
    auto stream =
        use_default_stream ? default_stream : GetCudaStream(se_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclBroadcast((void *)sendbuff, (void *)recvbuff,
                                           n_elements, dtype, root_rank, comm,
                                           stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status CommGroup::NcclSendImpl(const AlpaNcclUid &key, PjRtBuffer *buffer,
                               uint start, uint n_elements, int peer_p2p_rank,
                               bool use_default_stream) {
#if XLA_ENABLE_XCCL
  const int device_id = local_ids[key][0];
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype,
                      ToNcclDataType(buffer->on_device_shape().element_type()));
  int dtype_size = SizeOfType(dtype);
  TF_ASSIGN_OR_RETURN(std::uintptr_t sendbuff, ToUnsafePointer(buffer));
  sendbuff = sendbuff + start * dtype_size;
  auto comm = *comm_map[std::make_pair(key, device_id)].Acquire();
  auto stream = use_default_stream
                    ? default_stream
                    : GetCudaStream(send_streams[device_id].get());
  XLA_CUDA_RETURN_IF_ERROR(ncclSend((void *)sendbuff, n_elements, dtype,
                                    peer_p2p_rank, comm, stream));
  // cudaDeviceSynchronize();
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status CommGroup::NcclRecvImpl(const AlpaNcclUid &key, PjRtBuffer *buffer,
                               uint start, uint n_elements, int peer_p2p_rank,
                               bool use_default_stream) {
#if XLA_ENABLE_XCCL
  const int device_id = local_ids[key][0];
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype,
                      ToNcclDataType(buffer->on_device_shape().element_type()));
  int dtype_size = SizeOfType(dtype);
  TF_ASSIGN_OR_RETURN(std::uintptr_t recvbuff, ToUnsafePointer(buffer));
  recvbuff = recvbuff + start * dtype_size;
  auto comm = *comm_map[std::make_pair(key, device_id)].Acquire();
  auto stream = use_default_stream
                    ? default_stream
                    : GetCudaStream(recv_streams[device_id].get());
  XLA_CUDA_RETURN_IF_ERROR(ncclRecv((void *)recvbuff, n_elements, dtype,
                                    peer_p2p_rank, comm, stream));
  // cudaDeviceSynchronize();
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

// Other function
NcclComm::Lock CommGroup::AcquireComm(const AlpaNcclUid &key, int device_id) {
  return comm_map[std::make_pair(key, device_id)].Acquire();
}

// Cross-mesh allreduce thunk related
void SetCommGroup(std::string key, std::shared_ptr<CommGroup> g,
                  const AlpaNcclUid &uid) {
  cross_mesh_comms.emplace(key, std::make_pair(g, uid));
}

NcclComm::Lock GetCommunicator(std::string key, size_t device_id) {
  CrossMeshCommInfo &info = cross_mesh_comms.at(key);
  return info.first->AcquireComm(info.second, device_id);
}

// Other functions
AlpaNcclUid NcclUidSerialize(ncclUniqueId nccl_uid) {
  AlpaNcclUid nccl_uid_vec(sizeof(nccl_uid.internal), 0);
  memcpy(nccl_uid_vec.data(), &nccl_uid.internal, sizeof(nccl_uid.internal));
  return nccl_uid_vec;
}

ncclUniqueId NcclUidDeserialize(const AlpaNcclUid &nccl_uid_vec) {
  ncclUniqueId nccl_uid;
  CHECK_EQ(sizeof(nccl_uid.internal), nccl_uid_vec.size());
  memcpy(&nccl_uid.internal, nccl_uid_vec.data(), sizeof(nccl_uid.internal));
  return nccl_uid;
}

StatusOr<AlpaNcclUid> NcclGetUniqueId() {
#if XLA_ENABLE_XCCL
  ncclUniqueId id;
  XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  AlpaNcclUid nccl_uid_vec = NcclUidSerialize(id);
  return nccl_uid_vec;
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

StatusOr<int> NcclGetVersion() {
#if XLA_ENABLE_XCCL
  int version;
  XLA_CUDA_RETURN_IF_ERROR(ncclGetVersion(&version));
  return version;
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}
}  // namespace alpa
}  // namespace gpu
}  // namespace xla
