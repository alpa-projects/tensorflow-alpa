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

#include "tensorflow/compiler/xla/service/gpu/alpa_nccl_wrapper.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/env.h"

#include "pybind11/pybind11.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"

#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_client.h"

PYBIND11_MAKE_OPAQUE(std::vector<ncclComm_t>);

namespace xla {
namespace alpa_nccl {

ncclDataType_t MyToNcclDataType(PrimitiveType element_type) {
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
    default:
      return 4;
  }
}

std::shared_ptr< std::vector<ncclComm_t> > InitCommunicator(int n_devices, std::vector<int> devices_vec) {
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*n_devices);

    int* devices_ids = devices_vec.data();

    ncclCommInitAll(comms, n_devices, devices_ids);

    std::vector<ncclComm_t> comm_vec(comms, comms+n_devices);
    return std::make_shared< std::vector<ncclComm_t> >(std::move(comm_vec));
}

void LocalAllGather(int n_devices, 
                    std::vector<ncclComm_t> comms, 
                    std::vector<PyBuffer::object> buffers, 
                    std::vector<int> devices_ids, 
                    uint n_elements) {
  ncclDataType_t dtype = MyToNcclDataType(buffers[0].buf()->buffer()->on_device_shape().element_type());
  int dtype_size = SizeOfType(dtype);
  ncclGroupStart();
  for (int i = 0; i < n_devices; ++i) {
    std::uintptr_t sendbuff = buffers[i].buf()->UnsafeBufferPointer().ValueOrDie();
    sendbuff = sendbuff + n_elements*i*dtype_size;
    std::uintptr_t recvbuff = buffers[i].buf()->UnsafeBufferPointer().ValueOrDie();
    ncclAllGather((void*)sendbuff, (void*)recvbuff, n_elements, dtype, comms[i], cudaStreamLegacy);
  }
  ncclGroupEnd();
}

std::shared_ptr< std::vector<ncclComm_t> > nccl_InitCommunicator(int n_devices, std::vector<int> devices_vec) {
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*n_devices);
    int* devices_ids = devices_vec.data();
    ncclCommInitAll(comms, n_devices, devices_ids);
    std::vector<ncclComm_t> comm_vec(comms, comms+n_devices);
    return std::make_shared< std::vector<ncclComm_t> >(std::move(comm_vec));
}

void nccl_LocalAllGather(int n_devices, 
                         std::vector<ncclComm_t> comms, 
                         std::vector<PyBuffer::object> buffers, 
                         std::vector<int> devices_ids, 
                         std::vector<uint> local_starts, // TODO(hexu): is the range of uint too small?
                         uint global_start, 
                         uint n_elements) {
  ncclDataType_t dtype = MyToNcclDataType(buffers[0].buf()->buffer()->on_device_shape().element_type());
  int dtype_size = SizeOfType(dtype);
  ncclGroupStart();
  for (int i = 0; i < n_devices; ++i) {
    std::uintptr_t sendbuff = buffers[i].buf()->UnsafeBufferPointer().ValueOrDie();
    sendbuff = sendbuff + local_starts[i]*dtype_size;
    std::uintptr_t recvbuff = buffers[i].buf()->UnsafeBufferPointer().ValueOrDie();
    recvbuff = recvbuff + global_start*dtype_size;
    ncclAllGather((void*)sendbuff, (void*)recvbuff, n_elements, dtype, comms[i], cudaStreamLegacy);
  }
  ncclGroupEnd();
}


void nccl_DestroyComms(std::vector<ncclComm_t> comms) {
  for (auto comm : comms) 
    ncclCommDestroy(comm);
}

void nccl_BroadcastPartialGPUs(int n_devices, 
                               std::vector<ncclComm_t> comms, 
                               std::vector<PyBuffer::object> buffers, 
                               std::vector<uint> local_start_positions, 
                               uint n_elements, 
                               int root_rank) {
  ncclDataType_t dtype = MyToNcclDataType(buffers[0].buf()->buffer()->on_device_shape().element_type());
  int dtype_size = SizeOfType(dtype);
  ncclGroupStart();
  for (int i = 0; i < n_devices; ++i) {
    std::uintptr_t sendbuff = buffers[i].buf()->UnsafeBufferPointer().ValueOrDie();
    sendbuff = sendbuff + local_start_positions[i]*dtype_size;
    std::uintptr_t recvbuff = buffers[i].buf()->UnsafeBufferPointer().ValueOrDie();
    recvbuff = recvbuff + local_start_positions[i]*dtype_size;
    ncclBroadcast((void*)sendbuff, (void*)recvbuff, n_elements, dtype, root_rank, comms[i], cudaStreamLegacy);
  }
  ncclGroupEnd();
}

void nccl_Send(std::vector<ncclComm_t> comms, 
               PyBuffer::object buffer, 
               uint start, 
               uint n_elements, 
               int peer_p2p_rank) {
  ncclDataType_t dtype = MyToNcclDataType(buffer.buf()->buffer()->on_device_shape().element_type());
  int dtype_size = SizeOfType(dtype);
  std::uintptr_t sendbuff = buffer.buf()->UnsafeBufferPointer().ValueOrDie();
  sendbuff = sendbuff + start*dtype_size;
  ncclSend((void*)sendbuff, n_elements, dtype, peer_p2p_rank, comms[0], cudaStreamLegacy);
}

void nccl_Recv(std::vector<ncclComm_t> comms, 
               PyBuffer::object buffer, 
               uint start, 
               uint n_elements, 
               int peer_p2p_rank) {
  ncclDataType_t dtype = MyToNcclDataType(buffer.buf()->buffer()->on_device_shape().element_type());
  int dtype_size = SizeOfType(dtype);

  std::uintptr_t recvbuff = buffer.buf()->UnsafeBufferPointer().ValueOrDie();
  recvbuff = recvbuff + start*dtype_size;
  ncclRecv((void*)recvbuff, n_elements, dtype, peer_p2p_rank, comms[0], cudaStreamLegacy);
}

ncclUniqueId nccl_GetUniqueId() {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  return id;
}

int nccl_GetVersion() {
  int *version;
  ncclGetVersion(version);
  return *version;
}

std::shared_ptr< std::vector<ncclComm_t> > nccl_CreateCommunicators(int n_devices, 
                                                                    int world_size, 
                                                                    std::vector<int> devices_global_rank, 
                                                                    std::vector<int> devices_ids, 
                                                                    ncclUniqueId nccl_uid) {
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*n_devices);

  ncclGroupStart();
  for (int i=0; i<n_devices; i++) {
    cudaSetDevice(devices_ids[i]);
    ncclCommInitRank(comms+i, world_size, nccl_uid, devices_global_rank[i]);
  }
  ncclGroupEnd();
  std::vector<ncclComm_t> comm_vec(comms, comms+n_devices);
  return std::make_shared< std::vector<ncclComm_t> >(std::move(comm_vec));
}

int get_buffer_device_id(PyBuffer::object buffer){
  printf("local hardware id = %d\n", buffer.buf()->device()->local_hardware_id());
  return buffer.buf()->device()->local_hardware_id();
}

}  // namespace alpa_nccl

}  // namespace xla
