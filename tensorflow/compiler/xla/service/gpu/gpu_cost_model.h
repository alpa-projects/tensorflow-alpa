#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COST_MODEL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COST_MODEL_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace gpu {

double EstimateHloModuleCost(const HloModule* hlo_module);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COST_MODEL_H_
