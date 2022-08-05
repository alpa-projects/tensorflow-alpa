#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_GRAD_ACC_REWRITE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_GRAD_ACC_REWRITE_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace spmd {

// Rewrite for gradient accumulation. Note that this pass changes
// the semantics of the original HLO. To get correct results, this pass
// should be used together with XLA_SKIP_NCCL_COLLECTIVE_IDS.
//
// Before:
// d = dot(...)
// a = allreduce(d)
// new_grad = add(old_grad, a)
// return new_grad
//
// After:
// d = dot(...)
// new_grad = add(old_grad, d)
// a = allreduce(new_grad)
// return a

class GradAccRewrite : public HloModulePass {
 public:
  GradAccRewrite() = default;
  ~GradAccRewrite() override = default;
  absl::string_view name() const override { return "grad_acc_rewrite"; }

  StatusOr<bool> Run(HloModule* module) override;
};

std::string GetGradSyncChannelIds(const HloModule* module);

const char* const kSkippableAllReduce = "grad_acc_skippable_all_reduce";

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_GRAD_ACC_REWRITE_H_
