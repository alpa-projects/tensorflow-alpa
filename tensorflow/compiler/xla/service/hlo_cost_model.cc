#include "tensorflow/compiler/xla/service/hlo_cost_model.h"

#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {

double EstimateHloModuleCost(const HloModule* hlo_module) {
  return 0.0;
}

}  // namespace xla
