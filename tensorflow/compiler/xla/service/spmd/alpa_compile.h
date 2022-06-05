#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILE_H_

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {
namespace spmd {

// Run the auto sharding pass to add sharding anotations
// for each HLO instruction.
Status RunAutoShardingPass(HloModule* hlo_module, const CompileOptions& options);

// Run the SPMD partitioner pass.
Status RunSpmdPartitionerPass(HloModule* hlo_module, const CompileOptions& options);

};  // namespace spmd
};  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_ALPA_COMPILE_H_
