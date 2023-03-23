#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_EVENTS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_EVENTS_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
// This file provides APIs to sync Alpa's multi-stream behavior with
// XLA's own streams
namespace xla {
namespace gpu {
namespace se = ::stream_executor;

class DoneEventStats {
 public:
  void WaitForAllEventRecorded();
  se::Event* RecordEvent(int device_ordinal, se::StreamExecutor* executor);
  Status WaitOnStreams(std::vector<se::Stream*>& streams);
  Status WaitOnStreams(std::vector<std::unique_ptr<se::Stream>>& streams);

  bool empty() const { return value_.empty(); }

 private:
  absl::Mutex stat_mutex;
  int64_t record_counter ABSL_GUARDED_BY(stat_mutex) = 0;
  absl::flat_hash_map<int, std::unique_ptr<se::Event>> value_;
};

// Init function
void SetNumDeviceOnHost(int nd);

// Set idx to uuid for computations to record events
Status XlaSetIdxToUuid(const std::vector<int>& mapping);

// Reset event ids
Status ResetEvents(int uuid);

// Set events for communications to record events
se::Event* SetEvent(int uuid, int device_ordinal, se::StreamExecutor* executor);

// Get event stats for computation to record it
std::shared_ptr<DoneEventStats> GetEventStats(int index);

// Sync events
Status WaitEventOnStreams(int uuid, std::vector<se::Stream*>& streams);

Status WaitEventOnStreams(int uuid,
                          std::vector<std::unique_ptr<se::Stream>>& streams);

// Manage event stacks
// In case we only use events in cross-mesh communication and all of them are
// inside a PipeShardExecutable, we can reset it at the beginning of a
// PipeShardExecutable.
void ResetAlpaEvents();
};      // namespace gpu
};      // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_ALPA_EVENTS_H_
