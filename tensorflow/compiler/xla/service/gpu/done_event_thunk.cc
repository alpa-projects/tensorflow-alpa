#include "tensorflow/compiler/xla/service/gpu/done_event_thunk.h"

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

namespace xla {
namespace gpu {

DoneEventThunk::DoneEventThunk(ThunkInfo thunk_info, size_t index)
    : Thunk(Kind::kDoneEvent, thunk_info), output_index(index) {}

Status DoneEventThunk::Initialize(const GpuExecutable& executable,
                                  se::StreamExecutor* executor) {
  if (executor->device_ordinal() == 0) {
    event_stats = GetEventStats(output_index);
  }
  return OkStatus();
}

Status DoneEventThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream& stream = *params.stream;
  int device_ordinal = stream.parent()->device_ordinal();
  if (device_ordinal == 0) {
    absl::MutexLock lock(&mu_);
    cur_run_id = params.nccl_params.run_id;
  } else {
    absl::MutexLock lock(&mu_);
    auto run_id_match = [&]() {
      return params.nccl_params.run_id == cur_run_id;
    };
    mu_.Await(absl::Condition(&run_id_match));
  }
  se::Event* done_event =
      event_stats->RecordEvent(device_ordinal, stream.parent());
  TF_RET_CHECK(done_event->Init());
  stream.ThenRecordEvent(done_event);
  return OkStatus();
}
};  // namespace gpu
};  // namespace xla
