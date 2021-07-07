#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SLICE_AUTO_SHARDED_STAGES
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SLICE_AUTO_SHARDED_STAGES

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

class SliceAutoShardedStages : public HloModulePass {
 public:
  SliceAutoShardedStages() = default;
  ~SliceAutoShardedStages() override = default;
  absl::string_view name() const override {
    return "slice_auto_sharded_stages";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_SLICE_AUTO_SHARDED_STAGES

