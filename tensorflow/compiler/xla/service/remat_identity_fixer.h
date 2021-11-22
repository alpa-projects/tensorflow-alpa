#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_REMAT_IDENTITY_FIXER_H_H
#define TENSORFLOW_COMPILER_XLA_SERVICE_REMAT_IDENTITY_FIXER_H_H

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Fix the identity custom call marker for rematerialization.
// We move all-reduce out of the region wrapped by identity call,
// so we these all-reduce can be combined by AllReduceCombiner.
//
// Before:
// identity (...)
// b = ...
// a = allreduce(b)
// tuple = identity (a, ...)
// 
// After:
// identity (...)
// b = ..
// tuple = identity (b, ...)
// a = allreduce(tuple.0)

class RematIdentityFixer : public HloModulePass {
 public:
  RematIdentityFixer() = default;
  ~RematIdentityFixer() override = default;

  absl::string_view name() const override { return "remat-identity-fixer"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_REMAT_IDENTITY_FIXER_H_H
