#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTO_SHARDING
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTO_SHARDING

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

struct AutoShardingOption {
  bool force_data_parallel;
  int64 forward_backward_sep_id;
  int64 force_all_reduce_cost{-1};
  int64 force_reduce_scatter_cost{-1};
};

class AutoSharding : public HloModulePass {
 public:
  AutoSharding() = default;
  ~AutoSharding() override = default;
  absl::string_view name() const override { return "auto_sharding"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_AUTO_SHARDING

