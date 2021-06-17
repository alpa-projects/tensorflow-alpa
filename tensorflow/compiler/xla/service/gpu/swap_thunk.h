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

#include "tensorflow/compiler/xla/pjrt/swap.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"

namespace xla {
namespace gpu {

// Thunk to run a GPU swap out
class SwapOutThunk : public Thunk {
 public:
  SwapOutThunk(ThunkInfo thunk_info,
               std::vector<BufferAllocation::Slice> operands,
               BufferAllocation::Slice result, std::vector<int64> byte_sizes,
               int64 key);

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

  ~SwapOutThunk() override;

  se::Event* SwapFinishEvent() { return swap_finish_event_.get(); }

 private:
  const std::vector<BufferAllocation::Slice> operands_;
  const BufferAllocation::Slice result_;
  const std::vector<int64> byte_sizes_;
  const int64 key_;
  int64 executable_key_;
  se::StreamExecutor* executor_ = nullptr;
  std::unique_ptr<se::Event> swap_finish_event_;

  friend class SwapDoneThunk;
};

// Thunk to run a GPU swap in
class SwapInThunk : public Thunk {
 public:
  SwapInThunk(ThunkInfo thunk_info, BufferAllocation::Slice operands,
              std::vector<BufferAllocation::Slice> results,
              std::vector<int64> byte_sizes, int64 key);

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

  se::Event* SwapFinishEvent() { return swap_finish_event_.get(); }

 private:
  const BufferAllocation::Slice operand_;
  const std::vector<BufferAllocation::Slice> results_;
  const std::vector<int64> byte_sizes_;
  const int64 key_;
  int64 executable_key_;
  std::unique_ptr<se::Event> swap_finish_event_;
};

class SwapDoneThunk : public Thunk {
 public:
  SwapDoneThunk(ThunkInfo thunk_info);

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  se::Event* swap_finish_event_;
};
}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SWAP_THUNK_H_
