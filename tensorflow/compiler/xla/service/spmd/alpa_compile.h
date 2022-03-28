#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILE
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILE

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {

namespace spmd {

// Compile an Xla Computation into HloModule, then apply Alpa's passes.
// The result hlo is later compiled again to apply spmd and other optimizations.
StatusOr<std::unique_ptr<xla::HloModule>> RunAutoShardingPass(
    const XlaComputation& computation, CompileOptions options);

};  // namespace spmd

};  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILE