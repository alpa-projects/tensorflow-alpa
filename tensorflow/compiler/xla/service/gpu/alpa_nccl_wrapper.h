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
namespace alpa_nccl {

ncclDataType_t MyToNcclDataType(PrimitiveType element_type);

int SizeOfType(ncclDataType_t element_type);

std::shared_ptr< std::vector<ncclComm_t> > nccl_InitCommunicator(int n_devices, std::vector<int> devices_vec);

void nccl_LocalAllGather(int n_devices, 
                         std::vector<ncclComm_t> comms, 
                         std::vector<PyBuffer::object> buffers, 
                         std::vector<int> devices_ids, 
                         std::vector<uint> local_starts, 
                         uint global_start, 
                         uint n_elements);

void nccl_DestroyComms(std::vector<ncclComm_t> comms);

void nccl_BroadcastPartialGPUs(int n_devices, 
                               std::vector<ncclComm_t> comms, 
                               std::vector<PyBuffer::object> buffers, 
                               std::vector<uint> local_start_positions, 
                               uint n_elements, 
                               int root_rank);

void nccl_Send(std::vector<ncclComm_t> comms, 
               PyBuffer::object buffer, 
               uint start, 
               uint n_elements, 
               int peer_p2p_rank);

void nccl_Recv(std::vector<ncclComm_t> comms, 
               PyBuffer::object buffer, 
               uint start, 
               uint n_elements, 
               int peer_p2p_rank);

std::vector<char> nccl_uid_serialize(ncclUniqueId nccl_uid);

ncclUniqueId nccl_uid_deserialize(std::vector<char> nccl_uid_chars);

std::vector<char> nccl_GetUniqueId();

int nccl_GetVersion();

std::shared_ptr< std::vector<ncclComm_t> > nccl_CreateCommunicators(int n_devices, 
                                                                    int world_size, 
                                                                    std::vector<int> devices_global_rank, 
                                                                    std::vector<int> devices_ids, 
                                                                    std::vector<char> nccl_uid);

int get_buffer_device_id(PyBuffer::object buffer);


}  // namespace alpa_nccl
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_UTILS_H_
