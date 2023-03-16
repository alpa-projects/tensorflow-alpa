#include <stack>

#include "tensorflow/compiler/xla/service/gpu/alpa_events.h"

// FIXME(yonghao): only record events used for cross-mesh resharding
namespace xla {
namespace gpu {
absl::Mutex events_mu_;
using UuidToEvent_t = absl::flat_hash_map<int, std::shared_ptr<DoneEventStats>>;
std::shared_ptr<UuidToEvent_t> uuid_to_events =
    std::make_shared<UuidToEvent_t>();
std::vector<int> index_to_uuid;
int num_devices = -1;

void DoneEventStats::WaitForAllEventRecorded() {
  absl::MutexLock lock(&stat_mutex);
  auto all_event_recorded = [&]() { return record_counter == num_devices; };
  stat_mutex.Await(absl::Condition(&all_event_recorded));
}

se::Event* DoneEventStats::RecordEvent(int device_ordinal,
                                       se::StreamExecutor* executor) {
  absl::MutexLock lock(&stat_mutex);
  CHECK(record_counter < num_devices);
  value_[device_ordinal] = std::make_unique<se::Event>(executor);
  record_counter += 1;
  return value_[device_ordinal].get();
}

Status DoneEventStats::WaitOnStreams(std::vector<se::Stream*>& streams) {
  WaitForAllEventRecorded();
  CHECK_EQ(streams.size(), num_devices);
  for (int device_ordinal = 0; device_ordinal < num_devices; ++device_ordinal) {
    streams[device_ordinal]->ThenWaitFor(value_[device_ordinal].get());
  }
  return OkStatus();
}

Status DoneEventStats::WaitOnStreams(
    std::vector<std::unique_ptr<se::Stream>>& streams) {
  WaitForAllEventRecorded();
  CHECK_EQ(streams.size(), num_devices);
  for (int device_ordinal = 0; device_ordinal < num_devices; ++device_ordinal) {
    streams[device_ordinal]->ThenWaitFor(value_[device_ordinal].get());
  }
  return OkStatus();
}

void SetNumDeviceOnHost(int nd) { num_devices = nd; }

Status XlaSetIdxToUuid(const std::vector<int>& mapping) {
  index_to_uuid = mapping;
  for (int uuid : mapping) {
    ResetEvents(uuid);
  }
  return OkStatus();
}

Status ResetEvents(int uuid) {
  absl::MutexLock lock(&events_mu_);
  uuid_to_events->insert_or_assign(uuid, std::make_shared<DoneEventStats>());
  return OkStatus();
}

se::Event* SetEvent(int uuid, int device_ordinal,
                    se::StreamExecutor* executor) {
  absl::MutexLock lock(&events_mu_);
  return uuid_to_events->at(uuid)->RecordEvent(device_ordinal, executor);
}

std::shared_ptr<DoneEventStats> GetEventStats(int index) {
  int uuid = index_to_uuid[index];
  absl::MutexLock lock(&events_mu_);
  return uuid_to_events->at(uuid);
}

Status WaitEventOnStreams(int uuid, std::vector<se::Stream*>& streams) {
  absl::MutexLock lock(&events_mu_);
  if (uuid_to_events->count(uuid)) {
    TF_RETURN_IF_ERROR(uuid_to_events->at(uuid)->WaitOnStreams(streams));
  }
  return OkStatus();
}

Status WaitEventOnStreams(int uuid,
                          std::vector<std::unique_ptr<se::Stream>>& streams) {
  absl::MutexLock lock(&events_mu_);
  if (uuid_to_events->count(uuid)) {
    TF_RETURN_IF_ERROR(uuid_to_events->at(uuid)->WaitOnStreams(streams));
  }
  return OkStatus();
}

void ResetAlpaEvents() { uuid_to_events = std::make_shared<UuidToEvent_t>(); }
};  // namespace gpu
};  // namespace xla
