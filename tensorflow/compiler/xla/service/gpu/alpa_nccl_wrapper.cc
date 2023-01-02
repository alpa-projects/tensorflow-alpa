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
#include "tensorflow/compiler/xla/service/gpu/alpa_nccl_wrapper.h"

#include "tensorflow/compiler/xla/python/py_executable.h"

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
void AddCallBackReleasingBuffer(se::Stream *stream, PyBuffer::object &buf_obj) {
  // Holding the shared ptr of the buffer because we use it in an unsafe way.
  // This prevents XLA from freeing the buffer.
  std::shared_ptr<PjRtBuffer> pjrt_ref = buf_obj.buf()->shared_ptr_buffer();
  // TODO(yonghao): shall we add a usage hold for the buffer here?
  // Follows the HostCallback in cuda_gpu_executor.cc
  stream->ThenDoHostCallback([pjrt_ref]() {});
}

se::Stream *GetXlaStream(PjRtStreamExecutorClient *client, bool is_compute,
                         int device_id) {
  return is_compute ? client->device_state(device_id).compute_stream()
                    : client->device_state(0).GetLastDeviceToDeviceStream();
}
};  // namespace

PyCommGroup::PyCommGroup(std::shared_ptr<PyClient> backend)
    : CommGroup(backend == nullptr
                    ? nullptr
                    : tensorflow::down_cast<PjRtStreamExecutorClient *>(
                          backend->pjrt_client())) {}

// Communication operation related functions:
// FIXME: local allgather is deprecated
Status PyCommGroup::NcclLocalAllGather(const AlpaNcclUid &key,
                                       std::vector<PyBuffer::object> buffers,
                                       std::vector<uint> local_start_positions,
                                       uint global_start, uint n_elements,
                                       bool use_default_stream) {
#if XLA_ENABLE_XCCL
  std::vector<PjRtBuffer *> pjrt_buffers;
  for (PyBuffer::object &buf : buffers) {
    pjrt_buffers.push_back(buf.buf()->buffer());
  }
  TF_RETURN_IF_ERROR(NcclLocalAllGatherImpl(key, pjrt_buffers,
                                            local_start_positions, global_start,
                                            n_elements, use_default_stream));
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status PyCommGroup::NcclBroadcastPartialGPUs(
    const AlpaNcclUid &key, std::vector<PyBuffer::object> buffers,
    std::vector<uint> local_start_positions, uint n_elements, int root_rank,
    bool use_recv_stream, bool use_default_stream) {
#if XLA_ENABLE_XCCL
  std::vector<PjRtBuffer *> pjrt_buffers;
  for (PyBuffer::object &buf : buffers) {
    pjrt_buffers.push_back(buf.buf()->buffer());
  }
  TF_RETURN_IF_ERROR(NcclBroadcastPartialGPUsImpl(
      key, pjrt_buffers, local_start_positions, n_elements, root_rank,
      use_recv_stream, use_default_stream));

  const auto &device_ids = local_ids[key];
  int n_devices = device_ids.size();
  for (int i = 0; i < n_devices; ++i) {
    int device_id = device_ids[i];
    auto se_stream = use_default_stream
                         ? nullptr
                         : use_recv_stream ? recv_streams[device_id].get()
                                           : send_streams[device_id].get();
    if (!use_recv_stream && !use_default_stream) {
      AddCallBackReleasingBuffer(se_stream, buffers[i]);
    }
  }
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status PyCommGroup::NcclSend(const AlpaNcclUid &key, PyBuffer::object buffer,
                             uint start, uint n_elements, int peer_p2p_rank,
                             bool use_default_stream) {
#if XLA_ENABLE_XCCL
  const int device_id = local_ids[key][0];
  TF_RETURN_IF_ERROR(NcclSendImpl(key, buffer.buf()->buffer(), start,
                                  n_elements, peer_p2p_rank,
                                  use_default_stream));
  if (!use_default_stream) {
    AddCallBackReleasingBuffer(send_streams[device_id].get(), buffer);
  }
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

Status PyCommGroup::NcclRecv(const AlpaNcclUid &key, PyBuffer::object buffer,
                             uint start, uint n_elements, int peer_p2p_rank,
                             bool use_default_stream) {
#if XLA_ENABLE_XCCL
  TF_RETURN_IF_ERROR(NcclRecvImpl(key, buffer.buf()->buffer(), start,
                                  n_elements, peer_p2p_rank,
                                  use_default_stream));
  return OkStatus();
#else   // XLA_ENABLE_XCCL
  return Unimplemented("NCCL support is not available.");
#endif  // XLA_ENABLE_XCCL
}

// Sync functions:
Status PyCommGroup::CommunicatorRecordEvents(const AlpaUuids &uuids,
                                             int num_devices, bool is_send) {
  for (int uuid : uuids) {
    TF_RETURN_IF_ERROR(ResetEvents(uuid));
  }
  for (int device_id = 0; device_id < num_devices; ++device_id) {
    se::Stream *stream =
        is_send ? send_streams[device_id].get() : recv_streams[device_id].get();
    for (int uuid : uuids) {
      se::Event *event = SetEvent(uuid, device_id, executors[device_id]);
      TF_RET_CHECK(event->Init());
      stream->ThenRecordEvent(event);
    }
  }
  return OkStatus();
}

Status PyCommGroup::CommunicatorWaitEvents(const AlpaUuids &uuids,
                                           int num_devices, bool is_send) {
  auto &streams = is_send ? send_streams : recv_streams;
  for (int uuid : uuids) {
    TF_RETURN_IF_ERROR(WaitEventOnStreams(uuid, streams));
  }
  return OkStatus();
}

void PyCommGroup::CommWaitCompute(bool is_send, bool is_compute,
                                  int device_id) {
  se::Stream *waited = GetXlaStream(client_, is_compute, device_id);
  se::Stream *waiting =
      is_send ? send_streams[device_id].get() : recv_streams[device_id].get();
  waiting->ThenWaitFor(waited);
}

void PyCommGroup::ComputeWaitComm(bool is_send, bool is_compute,
                                  int device_id) {
  se::Stream *waiting = GetXlaStream(client_, is_compute, device_id);
  se::Stream *waited =
      is_send ? send_streams[device_id].get() : recv_streams[device_id].get();
  waiting->ThenWaitFor(waited);
}

// Cross Mesh Communication
void SetPyCommGroup(std::string key, std::shared_ptr<PyCommGroup> g,
                    const AlpaNcclUid &uid) {
  SetCommGroup(key, g, uid);
}

// Sync function
Status ComputationWaitEvents(const AlpaUuids &uuids,
                             std::shared_ptr<PyClient> client) {
  std::vector<se::Stream *> streams;
  PjRtStreamExecutorClient *pjrt_client =
      tensorflow::down_cast<PjRtStreamExecutorClient *>(client->pjrt_client());
  int num_devices = pjrt_client->device_count();
  for (int device_ordinal = 0; device_ordinal < num_devices; ++device_ordinal) {
    streams.push_back(
        pjrt_client->device_state(device_ordinal).compute_stream());
  }
  for (int uuid : uuids) {
    TF_RETURN_IF_ERROR(WaitEventOnStreams(uuid, streams));
  }
  return OkStatus();
}

// Event context management
void ResetEventContext(std::shared_ptr<PyClient> client) { ResetAlpaEvents(); }
// Other functions
StatusOr<int> GetBufferDeviceId(PyBuffer::object buffer) {
  return buffer.buf()->device()->local_hardware_id();
}
}  // namespace alpa
}  // namespace gpu
}  // namespace xla
