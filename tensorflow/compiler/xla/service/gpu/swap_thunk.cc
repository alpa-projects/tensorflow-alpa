#include "tensorflow/compiler/xla/service/gpu/swap_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace gpu {

SwapThunk::SwapThunk(Kind kind, ThunkInfo thunk_info)
    : Thunk(kind, thunk_info) {}

void SwapThunk::SetEvent(se::StreamExecutor* executor) {
  swap_finish_event_ = absl::make_unique<se::Event>(executor);
  swap_finish_event_->Init();
}

SwapOutThunk::SwapOutThunk(ThunkInfo thunk_info,
                           std::vector<BufferAllocation::Slice> operands,
                           std::vector<int64> byte_sizes)
    : SwapThunk(Thunk::kSwapOut, thunk_info),
      operands_(std::move(operands)),
      byte_sizes_(std::move(byte_sizes)) {}

Status SwapOutThunk::Initialize(const GpuExecutable& executable,
                                se::StreamExecutor* executor) {
  // register the key of the executable
  SetEvent(executor);
  return Status::OK();
}

SwapOutThunk::~SwapOutThunk() {
  // deallocate memory for this thunk
  if (!address_list_.empty()) {
    for (auto iter : address_list_) {
      executor_->HostMemoryDeallocate(iter);
    }
  }
}

Status SwapOutThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  int PartitionId = logical_id.computation_id;

  if (address_list_.empty()) {
    // alloc memory for the first time. todo: will this influence profile?
    executor_ = params.stream->parent();
    for (int64 byte_size : byte_sizes_) {
      address_list_.push_back(executor_->HostMemoryAllocate(byte_size));
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

    void* source_address_ = address_list_.at(i);
    params.stream->ThenMemcpy(source_address_, destination_data,
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

SwapInThunk::SwapInThunk(ThunkInfo thunk_info,
                         std::vector<BufferAllocation::Slice> results,
                         std::vector<int64> byte_sizes,
                         SwapOutThunk* memory_ref,
                         absl::InlinedVector<const SwapThunk*, 3> waits_for)
    : SwapThunk(Thunk::kSwapIn, thunk_info),
      results_(std::move(results)),
      byte_sizes_(std::move(byte_sizes)),
      memory_ref_(memory_ref),
      waits_for_(waits_for) {}

Status SwapInThunk::Initialize(const GpuExecutable& executable,
                               se::StreamExecutor* executor) {
  SetEvent(executor);
  return Status::OK();
}

Status SwapInThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.

  TF_ASSIGN_OR_RETURN(const GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());
  TF_ASSIGN_OR_RETURN(const DeviceAssignment::LogicalID logical_id,
                      params.device_assn->LogicalIdForDevice(global_device_id));
  int PartitionId = logical_id.computation_id;  // TODO(yonghao)

  params.stream->ThenWaitFor(memory_ref_->DoneEvent());
  for (const SwapThunk* thunk : waits_for_) {
    params.stream->ThenWaitFor(thunk->DoneEvent());
  }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  for (int32 i = 0; i < results_.size(); ++i) {
    const BufferAllocation::Slice& slice = results_.at(i);
    if (!slice.allocation()) {
      return InternalError("custom call output missing buffer allocation");
    }
    se::DeviceMemoryBase destination_data =
        params.buffer_allocations->GetDeviceAddress(slice);

    void* source_address_ = memory_ref_->AddressList().at(i);
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

SwapDoneThunk::SwapDoneThunk(ThunkInfo thunk_info, const SwapThunk* start)
    : Thunk(Thunk::kSwapDone, thunk_info), start_(start) {}

Status SwapDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  params.stream->ThenWaitFor(start_->DoneEvent());
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla