#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COMMON_COMPUTATION_ELIMINATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COMMON_COMPUTATION_ELIMINATION_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Eliminate common subcomputation for fused computations.
// This can accelerate compilation speed and reduce the number of kernels.
class CommonComputationElimination : public HloModulePass {
 public:
  CommonComputationElimination() = default;
  ~CommonComputationElimination() override = default;

  absl::string_view name() const override {
    return "common-computation-elimination";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COMMON_COMPUTATION_ELIMINATION_H_
