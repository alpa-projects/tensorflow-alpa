#include "tensorflow/compiler/xla/service/gpu/swap_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace gpu {

int64 SwapThunk::GetExecutableKey(const GpuExecutable* executable) {
  static absl::flat_hash_map<const GpuExecutable*, int64> map;
  static int64 counter = 0;
  auto iter = map.find(executable);
  if (iter != map.end()) {
    return iter->second;
  } else {
    map.emplace(executable, counter);
    return counter++;
  }
}

absl::flat_hash_map<int64, se::Event*> SwapThunk::swap_finish_events_;

se::Event* SwapThunk::GetEvent(int64 key) {
  auto iter = swap_finish_events_.find(key);
  CHECK(iter != swap_finish_events_.end()) << "no such key: " << key;
  return iter->second;
}

SwapThunk::SwapThunk(Kind kind, ThunkInfo thunk_info, int64 event_key)
    : Thunk(kind, thunk_info), event_key_(event_key) {}

void SwapThunk::SetEvent(se::StreamExecutor* executor) {
  swap_finish_event_ = absl::make_unique<se::Event>(executor);
  swap_finish_event_->Init();
  swap_finish_events_.insert({event_key_, swap_finish_event_.get()});
}

SwapOutThunk::SwapOutThunk(ThunkInfo thunk_info,
                           std::vector<BufferAllocation::Slice> operands,
                           std::vector<int64> byte_sizes, int64 key,
                           int64 event_key)
    : SwapThunk(Thunk::kSwapOut, thunk_info, event_key),
      operands_(std::move(operands)),
      byte_sizes_(std::move(byte_sizes)),
      key_(key) {}

Status SwapOutThunk::Initialize(const GpuExecutable& executable,
                                se::StreamExecutor* executor) {
  // register the key of the executable
  executable_key_ = GetExecutableKey(&executable);
  SetEvent(executor);
  return Status::OK();
}

SwapOutThunk::~SwapOutThunk() {
  if (executable_key_ != -1) {
    // deallocate memory for this thunk
    auto host_memory_ref =
        local_host_memory_table().GetOrNull(executable_key_, key_);
    const std::vector<void*>& address_list = host_memory_ref->address_list_;
    if (host_memory_ref != nullptr) {
      for (auto iter = address_list.begin(); iter != address_list.end();
           ++iter) {
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

  auto host_memory_ref =
      local_host_memory_table().GetOrCreate(executable_key_, key_);
  std::vector<void*>& address_list = host_memory_ref->address_list_;
  if (address_list.empty()) {
    // alloc memory for the first time. todo: will this influence profile?
    executor_ = params.stream->parent();
    for (int64 byte_size : byte_sizes_) {
      address_list.push_back(executor_->HostMemoryAllocate(byte_size));
    }
    // todo: GpuExecutor's HostMemoryAllocate is simply a new char[]. It does
    // not consider NUMA. Allocate it manually and then uses a
    // HostMemoryRegister instead.
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  for (int32 i = 0; i < operands_.size(); ++i) {
    const BufferAllocation::Slice& slice = operands_.at(i);
    if (!slice.allocation()) {
      return InternalError("custom call input missing buffer allocation");
    }
    se::DeviceMemoryBase destination_data =
        params.buffer_allocations->GetDeviceAddress(slice);

    void* source_address_ = address_list.at(i);
    params.stream->ThenMemcpy(source_address_, destination_data,
                              byte_sizes_.at(i));
  }
  params.stream->ThenRecordEvent(swap_finish_event_.get());
  host_memory_ref->swap_out_event_ = swap_finish_event_.get();
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Swap on GPU are not supported in this configuration. Please "
      "build with --config=cuda");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Status::OK();
}

SwapInThunk::SwapInThunk(ThunkInfo thunk_info,
                         std::vector<BufferAllocation::Slice> results,
                         std::vector<int64> byte_sizes, int64 key,
                         int64 event_key)
    : SwapThunk(Thunk::kSwapIn, thunk_info, event_key),
      results_(std::move(results)),
      byte_sizes_(std::move(byte_sizes)),
      key_(key) {}

Status SwapInThunk::Initialize(const GpuExecutable& executable,
                               se::StreamExecutor* executor) {
  executable_key_ = GetExecutableKey(&executable);
  SetEvent(executor);
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

  params.stream->ThenWaitFor(host_memory_ref->swap_out_event_);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  for (int32 i = 0; i < results_.size(); ++i) {
    const BufferAllocation::Slice& slice = results_.at(i);
    if (!slice.allocation()) {
      return InternalError("custom call output missing buffer allocation");
    }
    se::DeviceMemoryBase destination_data =
        params.buffer_allocations->GetDeviceAddress(slice);

    void* source_address_ = host_memory_ref->address_list_.at(i);
    params.stream->ThenMemcpy(&destination_data, source_address_,
                              byte_sizes_.at(i));
  }
  params.stream->ThenRecordEvent(swap_finish_event_.get());
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Swap on GPU are not supported in this configuration. Please "
      "build with --config=cuda");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Status::OK();
}

SwapDoneThunk::SwapDoneThunk(ThunkInfo thunk_info, int64 event_key)
    : Thunk(Thunk::kSwapDone, thunk_info), event_key_(event_key) {}

Status SwapDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  params.stream->ThenWaitFor(SwapThunk::GetEvent(event_key_));
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla