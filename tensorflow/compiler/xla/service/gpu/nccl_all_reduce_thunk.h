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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_

#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct NcclAllReduceConfig {
  NcclCollectiveConfig config;
  ReductionKind reduction_kind;
};

// Thunk that performs a NCCL-based All-Reduce or Reduce-Scatter among CUDA
// GPU-based replicas.
class NcclAllReduceReduceScatterThunkBase : public NcclCollectiveThunk {
 public:
  static std::optional<ReductionKind> MatchAllReduceComputation(
      mlir::Region& computation);

  NcclAllReduceReduceScatterThunkBase(Kind kind, ThunkInfo thunk_info,
                                      NcclAllReduceConfig config,
                                      std::vector<Buffer> buffers);

  // Added by Alpa
  void set_module_name(const std::string& module_name) {
    skip_env_name_ = module_name + "XLA_SKIP_NCCL_COLLECTIVE_IDS";
  }

 protected:
  const NcclCollectiveConfig& config() const override { return config_.config; }

  const NcclAllReduceConfig config_;
  const std::vector<Buffer> buffers_;
  std::string skip_env_name_ = "";  // Added by Alpa
};

// -----------------------------------------------------------------------------
// AllReduce thunks
// -----------------------------------------------------------------------------

class NcclAllReduceThunkBase : public NcclAllReduceReduceScatterThunkBase {
 public:
  using NcclAllReduceReduceScatterThunkBase::
      NcclAllReduceReduceScatterThunkBase;

 protected:
  Status RunAllReduce(const ExecuteParams& params, se::Stream& stream,
                      ncclComm_t comm);
};

class NcclAllReduceThunk : public NcclAllReduceThunkBase {
 public:
  NcclAllReduceThunk(ThunkInfo thunk_info, mlir::lmhlo::AllReduceOp op,
                     std::vector<Buffer> buffers);

  static const char* GetName() { return "AllReduce"; }

  static bool CanImplement(mlir::lmhlo::AllReduceOp op);
  static bool IsDegenerate(mlir::lmhlo::AllReduceOp op, int64_t replica_count,
                           int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::AllReduceOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;
};

class NcclAllReduceStartThunk : public NcclAllReduceThunkBase {
 public:
  NcclAllReduceStartThunk(ThunkInfo thunk_info,
                          mlir::lmhlo_gpu::AllReduceStartOp op,
                          std::vector<Buffer> buffers);

  static const char* GetName() { return "AllReduceStart"; }

  static bool CanImplement(mlir::lmhlo_gpu::AllReduceStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::AllReduceStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::AllReduceStartOp op);
  static constexpr bool IsAsync() { return true; }

  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class NcclAllReduceDoneThunk : public NcclCollectiveDoneThunk {
 public:
  NcclAllReduceDoneThunk(ThunkInfo thunk_info,
                         NcclCollectiveThunk::AsyncExecutor& async)
      : NcclCollectiveDoneThunk(Thunk::kNcclAllReduceDone, thunk_info, async) {}
};

// -----------------------------------------------------------------------------
// ReduceScatter thunks
// -----------------------------------------------------------------------------

class NcclReduceScatterThunkBase : public NcclAllReduceReduceScatterThunkBase {
 public:
  using NcclAllReduceReduceScatterThunkBase::
      NcclAllReduceReduceScatterThunkBase;

 protected:
  Status RunReduceScatter(const ExecuteParams& params, se::Stream& stream,
                          ncclComm_t comm);
};

class NcclReduceScatterThunk : public NcclReduceScatterThunkBase {
 public:
  NcclReduceScatterThunk(ThunkInfo thunk_info, mlir::lmhlo::ReduceScatterOp op,
                         std::vector<Buffer> buffers);

  static const char* GetName() { return "ReduceScatter"; }

  // Returns whether the given instruction can be lowered to a nccl
  // reduce-scatter call.
  static bool CanImplement(mlir::lmhlo::ReduceScatterOp op);
  static bool IsDegenerate(mlir::lmhlo::ReduceScatterOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(mlir::lmhlo::ReduceScatterOp op);
  static constexpr bool IsAsync() { return false; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;
};

class NcclReduceScatterStartThunk : public NcclReduceScatterThunkBase {
 public:
  NcclReduceScatterStartThunk(ThunkInfo thunk_info,
                              mlir::lmhlo_gpu::ReduceScatterStartOp op,
                              std::vector<Buffer> buffers);

  static const char* GetName() { return "ReduceScatterStart"; }

  // Returns whether the given instruction can be lowered to a nccl
  // reduce-scatter call.
  static bool CanImplement(mlir::lmhlo_gpu::ReduceScatterStartOp op);
  static bool IsDegenerate(mlir::lmhlo_gpu::ReduceScatterStartOp op,
                           int64_t replica_count, int64_t partition_count);
  static CollectiveOpGroupMode GetGroupMode(
      mlir::lmhlo_gpu::ReduceScatterStartOp op);
  static constexpr bool IsAsync() { return true; }
  AsyncExecutor& async_executor() { return async_; }

 protected:
  Status RunNcclCollective(const ExecuteParams& params,
                           ncclComm_t comm) override;

 private:
  AsyncExecutor async_;
};

class NcclReduceScatterDoneThunk : public NcclCollectiveDoneThunk {
 public:
  NcclReduceScatterDoneThunk(ThunkInfo thunk_info,
                             NcclCollectiveThunk::AsyncExecutor& async)
      : NcclCollectiveDoneThunk(Thunk::kNcclReduceScatterDone, thunk_info,
                                async) {}
};

// -----------------------------------------------------------------------------

Status RunAllReduce(const NcclAllReduceConfig& config,
                    std::vector<DeviceBufferPair>& buffers, se::Stream& stream,
                    ncclComm_t comm, const std::string& env_name);

Status RunReduceScatter(ReductionKind reduction_kind,
                        std::vector<DeviceBufferPair>& buffers,
                        se::Stream& stream, ncclComm_t comm);

// Added by Alpa
class CrossMeshNcclAllReduceThunk : public Thunk {
 public:
  using Buffer = NcclCollectiveThunk::Buffer;

  explicit CrossMeshNcclAllReduceThunk(ThunkInfo thunk_info,
                                       std::vector<Buffer> buffers,
                                       ReductionKind reduction_kind,
                                       xla::PrimitiveType op_type,
                                       std::string key);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const NcclAllReduceConfig config_;
  const std::vector<Buffer> buffers_;
  bool first_call_to_execute_ = true;
  std::string key_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_ALL_REDUCE_THUNK_H_
