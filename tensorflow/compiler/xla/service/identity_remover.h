#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_IDENTITY_REMOVER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_IDENTITY_REMOVER_H_

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Remove the identity custom call.
class IdentityRemover : public HloModulePass {
 public:
  IdentityRemover() = default;
  ~IdentityRemover() override = default;

  absl::string_view name() const override { return "identity-remover"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_IDENTITY_REMOVER_H_
