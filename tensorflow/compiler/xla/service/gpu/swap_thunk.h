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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SWAP_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SWAP_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

class SwapThunk : public Thunk {
 public:
  SwapThunk(Kind kind, ThunkInfo thunk_info);

  se::Event* SwapFinishEvent() { return swap_finish_event_.get(); }

  se::Event* DoneEvent() const { return swap_finish_event_.get(); }

 protected:
  void SetEvent(se::StreamExecutor* executor);
  std::unique_ptr<se::Event> swap_finish_event_ = nullptr;
};

// Thunk to run a GPU swap out
class SwapOutThunk : public SwapThunk {
 public:
  SwapOutThunk(ThunkInfo thunk_info,
               std::vector<BufferAllocation::Slice> operands,
               std::vector<int64> byte_sizes);

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

  const std::vector<void*>& AddressList() const { return address_list_; }

  ~SwapOutThunk() override;

 private:
  const std::vector<BufferAllocation::Slice> operands_;
  const std::vector<int64> byte_sizes_;
  se::StreamExecutor* executor_ = nullptr;
  std::vector<void*> address_list_;
};

// Thunk to run a GPU swap in
class SwapInThunk : public SwapThunk {
 public:
  SwapInThunk(ThunkInfo thunk_info,
              std::vector<BufferAllocation::Slice> results,
              std::vector<int64> byte_sizes, SwapOutThunk* memory_ref,
              absl::InlinedVector<const SwapThunk*, 3> waits_for);

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const std::vector<BufferAllocation::Slice> results_;
  const std::vector<int64> byte_sizes_;
  const absl::InlinedVector<const SwapThunk*, 3> waits_for_;
  const SwapOutThunk* memory_ref_;
};

// Thunk to sync a swap(in)
class SwapDoneThunk : public Thunk {
 public:
  SwapDoneThunk(ThunkInfo thunk_info, const SwapThunk* start);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const SwapThunk* start_;
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SWAP_THUNK_H_
