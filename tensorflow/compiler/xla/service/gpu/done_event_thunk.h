#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DONE_EVENT_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DONE_EVENT_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/alpa_events.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {
// Thunk to record an early done of a buffer
class DoneEventThunk : public Thunk {
 public:
  DoneEventThunk(ThunkInfo thunk_info, size_t index);

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const size_t output_index;
  std::shared_ptr<DoneEventStats> event_stats;
  absl::Mutex mu_;
  RunId cur_run_id ABSL_GUARDED_BY(mu_);
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_DONE_EVENT_THUNK_H_
