#include "tensorflow/compiler/xla/service/gpu/swap_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace gpu {

SwapOutThunk::SwapOutThunk(ThunkInfo thunk_info, 
                          std::vector<BufferAllocation::Slice> operands,
                          BufferAllocation::Slice result,
                          std::vector<int64> byte_sizes, 
                          int64 key
                          )
    : Thunk(Thunk::kSwapOut, thunk_info),  // todo: kCopy? kSwap? 
      operands_(std::move(operands)),
      result_(std::move(result)),
      byte_sizes_(std::move(byte_sizes)), 
      key_(key),
      executable_key_(-1) {}

Status SwapOutThunk::Initialize(const GpuExecutable& executable,
                                se::StreamExecutor* executor) {
  // register the key of the executable
  executable_key_ = 0;  // TODO
  return Status::OK();
}

SwapOutThunk::~SwapOutThunk() {
  if (executable_key_ != -1) {
    // deallocate memory for this thunk
    auto list_ptr = local_host_memory_table().GetOrNull(executable_key_, key_);
    if (list_ptr != nullptr) {
      for (auto iter = list_ptr->begin(); iter != list_ptr->end();++iter) {
        executor_->HostMemoryDeallocate(*iter);
      }
    }
    local_host_memory_table().remove(executable_key_, key_);
  }
}

Status SwapOutThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  static int64 formal_out = 0;
  
  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  int partitionId = logical_id.computation_id;

  auto host_memory_ref = local_host_memory_table().GetOrCreate(executable_key_, key_);
  if (host_memory_ref->empty()) {
    // alloc memory for the first time. todo: will this influence profile? 
    executor_ = params.stream->parent();
    for (int64 byte_size : byte_sizes_) {
      host_memory_ref->push_back(executor_->HostMemoryAllocate(byte_size));
    }
      // todo: GpuExecutor's HostMemoryAllocate is simply a new char[]. It does not consider NUMA. Allocate it manually and then uses a HostMemoryRegister instead. 
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  for (int32 i = 0;i < operands_.size();++i) {
    const BufferAllocation::Slice& slice = operands_.at(i);
    if (!slice.allocation())
      return InternalError("custom call input missing buffer allocation");
    se::DeviceMemoryBase destination_data = 
      params.buffer_allocations->GetDeviceAddress(slice);

    void *source_address_ = host_memory_ref->at(i);
    params.stream->ThenMemcpy(source_address_, destination_data, byte_sizes_.at(i));
  }
  se::DeviceMemoryBase destination_data = 
    params.buffer_allocations->GetDeviceAddress(result_);
  params.stream->ThenMemcpy(&destination_data, &formal_out, 8);
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Swap on GPU are not supported in this configuration. Please "
      "build with --config=cuda");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Status::OK();
}

SwapInThunk::SwapInThunk(ThunkInfo thunk_info, 
                        BufferAllocation::Slice operand,
                        std::vector<BufferAllocation::Slice> results,
                        std::vector<int64> byte_sizes, 
                        int64 key)
    : Thunk(Thunk::kSwapIn, thunk_info),  // todo: kCopy? kSwap?
      operand_(std::move(operand)),
      results_(std::move(results)),
      byte_sizes_(std::move(byte_sizes)), 
      key_(key),
      executable_key_(-1) {}

Status SwapInThunk::Initialize(const GpuExecutable& executable,
                                se::StreamExecutor* executor) {
  executable_key_ = 0;  // TODO
  return Status::OK();
}

Status SwapInThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  int partitionId = logical_id.computation_id;

  auto host_memory_ref = local_host_memory_table().Get(executable_key_, key_);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  for (int32 i = 0; i < results_.size();++i) {
    const BufferAllocation::Slice& slice = results_.at(i);
    if (!slice.allocation())
      return InternalError("custom call input missing buffer allocation");
    se::DeviceMemoryBase destination_data = 
      params.buffer_allocations->GetDeviceAddress(slice);

    void *source_address_ = host_memory_ref->at(i);
    params.stream->ThenMemcpy(&destination_data, source_address_, byte_sizes_.at(i));
  }
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Swap on GPU are not supported in this configuration. Please "
      "build with --config=cuda");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Status::OK();
}

} // namespace gpu
} // namespace xla