/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/optimization_barrier_expander.h"

namespace xla {

StatusOr<bool> OptimizationBarrierExpander::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> barriers;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    bool modified = false;
    for (HloInstruction* inst : computation->instructions()) {
      // Modified by Alpa: add pipeline marker option
      if (inst->IsCustomCall("pipeline_marker") || inst->opcode() == HloOpcode::kOptimizationBarrier) {
        barriers.push_back(inst);
        modified = true;
      }
    }
  }

  // Modified by Alpa: remove the module->has_schedule() branch;

  for (HloInstruction* inst : barriers) {
    // Modified by Alpa: use another way to expand. TODO: maybe no need to change?
    HloInstruction* identity = inst->parent()->AddInstruction(
        HloInstruction::CreateUnary(inst->shape(), HloOpcode::kBitcast,
                                    inst->mutable_operand(0)));
    inst->ReplaceAllUsesWith(identity);
    inst->parent()->RemoveInstruction(inst);
  }

  return !barriers.empty();
}

}  // namespace xla
