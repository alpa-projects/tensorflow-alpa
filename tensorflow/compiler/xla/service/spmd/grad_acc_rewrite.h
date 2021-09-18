#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_GRAD_ACC_REWRITE
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_GRAD_ACC_REWRITE

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

class GradAccRewrite : public HloModulePass {
 public:
  GradAccRewrite() = default;
  ~GradAccRewrite() override = default;
  absl::string_view name() const override { return "grad_acc_rewrite"; }

  StatusOr<bool> Run(HloModule* module) override;
};

std::string GetGradSyncChannelIds(const HloModule* module);

std::string GetGradSyncChannelIdsWithHint(const HloModule* module,
                                          const std::vector<int> grad_idx);

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_GRAD_ACC_REWRITE
