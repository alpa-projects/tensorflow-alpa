#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SLICE_AUTO_SHARDED_STAGES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SLICE_AUTO_SHARDED_STAGES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class SliceAutoShardedStages : public HloModulePass {
 public:
  SliceAutoShardedStages() = default;
  ~SliceAutoShardedStages() override = default;
  absl::string_view name() const override {
    return "slice_auto_sharded_stages";
  }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SLICE_AUTO_SHARDED_STAGES_H_
