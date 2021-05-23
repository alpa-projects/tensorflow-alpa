/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/attr.h"
#include "pybind11/cast.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl_bind.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/distributed/client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/pjrt/distributed/service.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/interpreter_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/compiler/xla/python/dlpack.h"
#include "tensorflow/compiler/xla/python/jax_jit.h"
#include "tensorflow/compiler/xla/python/ops.h"
#include "tensorflow/compiler/xla/python/outfeed_receiver_py.h"
#include "tensorflow/compiler/xla/python/pmap_lib.h"
#include "tensorflow/compiler/xla/python/profiler.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_executable.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/pytree.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/xla_compiler.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/python/lib/core/bfloat16.h"

// TODO(phawkins): remove host_id properties after JAX is update to avoid them.

namespace xla {
namespace {

namespace py = pybind11;

bool IsOptimizedBuild() {
#if NDEBUG
  return true;
#else
  return false;
#endif  // NDEBUG
}

}  // namespace

PYBIND11_MODULE(xla_extension, m) {
  CHECK(tensorflow::RegisterNumpyBfloat16());

  // Types
  py::enum_<PrimitiveType>(m, "PrimitiveType")
      .value("PRIMITIVE_TYPE_INVALID", PRIMITIVE_TYPE_INVALID)
      .value("PRED", PRED)
      .value("S8", S8)
      .value("S16", S16)
      .value("S32", S32)
      .value("S64", S64)
      .value("U8", U8)
      .value("U16", U16)
      .value("U32", U32)
      .value("U64", U64)
      .value("F16", F16)
      .value("BF16", BF16)
      .value("F32", F32)
      .value("F64", F64)
      .value("C64", C64)
      .value("C128", C128)
      .value("TUPLE", TUPLE)
      .value("OPAQUE_TYPE", OPAQUE_TYPE)
      .value("TOKEN", TOKEN);

  m.def("bfloat16_dtype",
        []() { return py::handle(tensorflow::Bfloat16Dtype()); });

  // Must be before PyClient.compile.
  BuildXlaCompilerSubmodule(m);

  py::class_<PjRtDevice, ClientAndPtr<PjRtDevice>>(
      m, "Device",
      "A descriptor of an available device.\n\nSubclasses are used to "
      "represent specific types of devices, e.g. CPUs, GPUs. Subclasses may "
      "have additional properties specific to that device type.")
      .def_property_readonly(
          "id", &PjRtDevice::id,
          "Integer ID of this device.\n\nUnique across all available devices "
          "of this type, including remote devices on multi-host platforms.")
      .def_property_readonly(
          "process_index", &PjRtDevice::process_index,
          "Integer index of this device's process.\n\n"
          "This is always 0 except on multi-process platforms.")
      .def_property_readonly("host_id", &PjRtDevice::process_index,
                             "Deprecated; please use process_index")
      .def_property_readonly("task_id", &PjRtDevice::process_index,
                             "Deprecated; please use process_index")
      .def_property_readonly("platform",
                             [](const PjRtDevice& device) {
                               return device.client()->platform_name();
                             })
      .def_property_readonly("device_kind", &PjRtDevice::device_kind)
      .def_property_readonly(
          "client",
          [](const ClientAndPtr<PjRtDevice>& device) { return device.client; })
      .def("__str__", &PjRtDevice::DebugString)
      .def("synchronize_all_activity", [](PjRtDevice& device) {
             PjRtStreamExecutorDevice* stream_device =
               dynamic_cast<PjRtStreamExecutorDevice*>(&device);
             CHECK_NE(stream_device, nullptr);
             TF_ASSIGN_OR_RETURN(LocalDeviceState* local_device,
                                 stream_device->GetLocalDeviceState());
             local_device->SynchronizeAllActivity();
             return Status::OK();
           })
      .def("transfer_to_infeed",
           [](PjRtDevice& device, const LiteralSlice& literal) {
             GlobalPyRefManager()->CollectGarbage();
             py::gil_scoped_release gil_release;
             return device.TransferToInfeed(literal);
           })
      .def("transfer_from_outfeed",
           [](PjRtDevice& device, const Shape& shape) -> StatusOr<py::object> {
             GlobalPyRefManager()->CollectGarbage();
             std::shared_ptr<Literal> literal;
             {
               py::gil_scoped_release gil_release;
               Shape shape_with_layout = shape;
               ShapeUtil::ForEachMutableSubshape(
                   &shape_with_layout, [](Shape* subshape, const ShapeIndex&) {
                     if (!subshape->has_layout()) {
                       LayoutUtil::SetToDefaultLayout(subshape);
                     }
                   });
               literal = std::make_shared<Literal>(shape_with_layout);
               TF_RETURN_IF_ERROR(device.TransferFromOutfeed(literal.get()));
             }
             return LiteralToPython(std::move(literal));
           });

  py::class_<CpuDevice, PjRtDevice, ClientAndPtr<CpuDevice>>(m, "CpuDevice")
      .def("__repr__", [](const CpuDevice& device) {
        return absl::StrFormat("CpuDevice(id=%i)", device.id());
      });

  py::class_<GpuDevice, PjRtDevice, ClientAndPtr<GpuDevice>>(m, "GpuDevice")
      .def("__repr__", [](const GpuDevice& device) {
        return absl::StrFormat("GpuDevice(id=%i, process_index=%i)",
                               device.id(), device.process_index());
      });

  py::class_<PjRtTpuDevice, PjRtDevice, ClientAndPtr<PjRtTpuDevice>>(
      m, "TpuDevice")
      .def_property_readonly(
          "coords",
          [](const PjRtTpuDevice& device) -> pybind11::tuple {
            return IntSpanToTuple(device.coords());
          },
          "The coordinates of this TpuDevice's chip in the TPU mesh network.")
      .def_property_readonly(
          "core_on_chip", &PjRtTpuDevice::core_on_chip,
          "The index of this TpuDevice's core on the TPU chip.")
      .def("__repr__", [](const PjRtTpuDevice& device) {
        return absl::StrFormat(
            "TpuDevice(id=%i, process_index=%i, coords=(%s), core_on_chip=%i)",
            device.id(), device.process_index(),
            absl::StrJoin(device.coords(), ","), device.core_on_chip());
      });

  // Local XLA client methods.

  py::class_<GpuAllocatorConfig> alloc_config(m, "GpuAllocatorConfig");
  alloc_config.def(py::init<>())
      .def_readwrite("kind", &GpuAllocatorConfig::kind)
      .def_readwrite("memory_fraction", &GpuAllocatorConfig::memory_fraction)
      .def_readwrite("preallocate", &GpuAllocatorConfig::preallocate);
  py::enum_<GpuAllocatorConfig::Kind>(alloc_config, "Kind")
      .value("DEFAULT", GpuAllocatorConfig::Kind::kDefault)
      .value("PLATFORM", GpuAllocatorConfig::Kind::kPlatform)
      .value("BFC", GpuAllocatorConfig::Kind::kBFC)
      .value("CUDA_ASYNC", GpuAllocatorConfig::Kind::kCudaAsync);

  py::enum_<PjRtClient::HostBufferSemantics>(m, "HostBufferSemantics")
      .value("IMMUTABLE_ONLY_DURING_CALL",
             PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall)
      .value("IMMUTABLE_UNTIL_TRANSFER_COMPLETES",
             PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes)
      .value("ZERO_COPY", PjRtClient::HostBufferSemantics::kZeroCopy);

  py::class_<PyClient, std::shared_ptr<PyClient>> py_local_client(m, "Client");
  py_local_client.def_property_readonly("platform", &PyClient::platform_name)
      .def_property_readonly("platform_version", &PyClient::platform_version)
      .def_property_readonly("runtime_type", &PyClient::runtime_type)
      .def("device_count", &PyClient::device_count)
      .def("local_device_count", &PyClient::addressable_device_count)
      .def("devices", &PyClient::Devices)
      .def("local_devices", &PyClient::LocalDevices)
      .def("live_buffers", &PyClient::LiveBuffers)
      .def("process_index", &PyClient::process_index)
      .def("host_id", &PyClient::process_index)
      .def("task_id", &PyClient::process_index)
      .def("get_default_device_assignment",
           &PyClient::GetDefaultDeviceAssignment)
      // TODO(skye): delete after all callers can handle 2D output
      .def("get_default_device_assignment",
           &PyClient::GetDefaultDeviceAssignment1D)
      .def("create_channel_handle", &PyClient::CreateChannelHandle)
      .def("create_device_to_host_channel_handle",
           &PyClient::CreateDeviceToHostChannelHandle)
      .def("create_host_to_device_channel_handle",
           &PyClient::CreateHostToDeviceChannelHandle)
      .def("buffer_from_pyval", &PyClient::BufferFromPyval, py::arg("argument"),
           py::arg("device") = nullptr, py::arg("force_copy") = false,
           py::arg("host_buffer_semantics") =
               PjRtClient::HostBufferSemantics::kZeroCopy)
      .def("compile", &PyClient::Compile, py::arg("computation"),
           py::arg("compile_options") = CompileOptions())
      .def("heap_profile", &PyClient::HeapProfile)
      // TODO(zhangqiaorjc): Experimental.
      .def("defragment", &PyClient::Defragment);

  m.def(
      "get_cpu_client",
      [](bool asynchronous) -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetCpuClient(asynchronous));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true);
  m.def(
      "get_tfrt_cpu_client",
      [](bool asynchronous) -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                            GetTfrtCpuClient(asynchronous));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true);
  m.def("get_interpreter_client", []() -> StatusOr<std::shared_ptr<PyClient>> {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                        GetInterpreterClient());
    return std::make_shared<PyClient>(std::move(client));
  });
  m.def(
      "get_gpu_client",
      [](bool asynchronous, const GpuAllocatorConfig& allocator_config,
         std::shared_ptr<DistributedRuntimeClient> distributed_client,
         int node_id) -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<PjRtClient> client,
            GetGpuClient(asynchronous, allocator_config,
                         std::move(distributed_client), node_id));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("asynchronous") = true,
      py::arg("allocator_config") = GpuAllocatorConfig(),
      py::arg("distributed_client") = nullptr, py::arg("node_id") = 0);
  m.def(
      "get_tpu_client",
      [](int max_inflight_computations) -> StatusOr<std::shared_ptr<PyClient>> {
        TF_ASSIGN_OR_RETURN(std::shared_ptr<PjRtClient> client,
                            GetTpuClient(max_inflight_computations));
        return std::make_shared<PyClient>(std::move(client));
      },
      py::arg("max_inflight_computations") = 32);

  TF_CHECK_OK(PyBuffer::RegisterTypes(m));

  py::class_<PyExecutable, std::shared_ptr<PyExecutable>> executable(
      m, "Executable");
  executable.def_property_readonly("client", &PyExecutable::client)
      .def("local_logical_device_ids",
           [](PyExecutable* exec) {
             auto span = exec->addressable_device_logical_ids();
             // Not on dispatch critical path, so ok to have heap allocation.
             std::vector<std::pair<int, int>> addressable_device_logic_ids;
             addressable_device_logic_ids.reserve(span.size());
             for (const auto& logical_device_id : span) {
               addressable_device_logic_ids.push_back(std::make_pair(
                   logical_device_id.replica, logical_device_id.partition));
             }
           })
      .def("local_devices", &PyExecutable::AddressableDevices)
      .def("size_of_generated_code_in_bytes",
           &PyExecutable::SizeOfGeneratedCodeInBytes)
      .def("delete", &PyExecutable::Delete)
      .def("execute", &PyExecutable::Execute, py::arg("arguments"))
      .def("execute_sharded_on_local_devices",
           &PyExecutable::ExecuteShardedOnLocalDevices, py::arg("arguments"))
      .def("hlo_modules", &PyExecutable::HloModules)
      .def_property_readonly("traceback", &PyExecutable::traceback)
      .def_property_readonly("fingerprint",
                             [](PyExecutable* exec) -> py::object {
                               if (exec->fingerprint().has_value()) {
                                 return py::bytes(*exec->fingerprint());
                               } else {
                                 return py::none();
                               }
                             })
      .def("total_allocation_size", [](PyExecutable* exec){
             const PjRtExecutable* pjrt_executable = &exec->pjrt_executable();
             const PjRtStreamExecutorExecutable* stream_executable = dynamic_cast<const PjRtStreamExecutorExecutable*>(pjrt_executable);
             absl::Span<const std::shared_ptr<LocalExecutable>> local_executables =\
                 stream_executable->executables();
             Executable* executable = local_executables[0]->executable();
             return executable->TotalAllocationSize();
           })
      .def_property_readonly("traceback", &PyExecutable::traceback);

  m.def("buffer_to_dlpack_managed_tensor", BufferToDLPackManagedTensor,
        py::arg("buffer"), py::arg("take_ownership") = true);
  m.def("dlpack_managed_tensor_to_buffer", DLPackManagedTensorToBuffer);

  m.def("init_nccl_communicators", [](
      PyClient* py_client,
      std::shared_ptr<DistributedRuntimeClient> distributed_client,
      int node_id,
      PyExecutable* py_executable) {
    absl::Span<PjRtDevice* const> devices = py_client->pjrt_client()->devices();
    std::vector<int> device_to_node;
    device_to_node.reserve(devices.size());
    for (auto device : devices) {
      device_to_node.push_back(device->process_index());
    }

    TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                        py_executable->pjrt_executable().GetHloModules());

    for (const auto& hlo_module : hlo_modules) {
      std::vector<std::vector<GlobalDeviceId>> communication_groups =
        GetCommunicationGroups(hlo_module.get());

      TF_RETURN_IF_ERROR(InitNcclCommunicators(
        std::move(distributed_client), node_id, device_to_node, communication_groups));
    }

    return Status::OK();
  });

  BuildProfilerSubmodule(&m);
  BuildOpsSubmodule(&m);
  BuildOutfeedReceiverSubmodule(&m);
  BuildPytreeSubmodule(m);
  jax::BuildJaxjitSubmodule(m);
  jax::BuildPmapSubmodule(m);
  BuildTracebackSubmodule(m);

  py::class_<DistributedRuntimeService,
             std::unique_ptr<DistributedRuntimeService>>
      distributed_runtime_service(m, "DistributedRuntimeService");
  py::class_<DistributedRuntimeClient,
             std::shared_ptr<DistributedRuntimeClient>>
      distributed_runtime_client(m, "DistributedRuntimeClient");
  distributed_runtime_client.def("connect", &DistributedRuntimeClient::Connect)
      .def("shutdown", &DistributedRuntimeClient::Shutdown);

  //m.def("get_distributed_runtime_service", &GetDistributedRuntimeService);
  //m.def("get_distributed_runtime_client", &GetDistributedRuntimeClient);

  m.def("get_distributed_runtime_service", [](std::string address, int num_nodes) {
    DistributedRuntimeServiceImpl::Options service_options;
    service_options.num_nodes = num_nodes;
    return GetDistributedRuntimeService(address, service_options);
  });
  m.def("get_distributed_runtime_client", [](std::string address, int node_id) {
    DistributedRuntimeClient::Options client_options;
    client_options.node_id = node_id;
    return GetDistributedRuntimeClient(address, client_options);
  });

  m.def("collect_garbage", []() { GlobalPyRefManager()->CollectGarbage(); });

  m.def("is_optimized_build", &IsOptimizedBuild);
}  // NOLINT(readability/fn_size)

}  // namespace xla
