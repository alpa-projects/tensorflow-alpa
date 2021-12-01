#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_MODLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_MODEL_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

double EstimateHloModuleCost(const HloModule* hlo_module);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
