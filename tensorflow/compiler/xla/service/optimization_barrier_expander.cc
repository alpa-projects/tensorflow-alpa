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

// Added by alpa
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {

StatusOr<bool> OptimizationBarrierExpander::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> barriers;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    bool modified = false;
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == HloOpcode::kOptimizationBarrier) {
        barriers.push_back(inst);
        modified = true;
      }
    }

    if (modified && module->has_schedule()) {
      const auto& sequences = module->schedule().sequences();
      auto it = sequences.find(computation->unique_id());
      if (it != sequences.end()) {
        std::vector<HloInstruction*> sequence;
        sequence.reserve(it->second.instructions().size());
        absl::c_copy_if(it->second.instructions(), std::back_inserter(sequence),
                        [](HloInstruction* inst) {
                          return inst->opcode() !=
                                 HloOpcode::kOptimizationBarrier;
                        });
        module->schedule().set_sequence(computation, sequence);
      }
    }
  }

  //for (HloInstruction* inst : barriers) {
  //  HloInstruction* arg = inst->mutable_operand(0);
  //  TF_RETURN_IF_ERROR(arg->CopyAllControlDepsFrom(inst));

  //  TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(arg));
  //  TF_RETURN_IF_ERROR(inst->DropAllControlDeps());

  //  TF_RETURN_IF_ERROR(inst->parent()->RemoveInstruction(inst));
  //}

  // Added by Alpa.
  // We use identity markers to group these variables in a tuple.
  // This greatly reduces the memory usage compared to the above
  // default behavior.
  for (HloInstruction* inst : barriers) {
    HloInstruction* identity = inst->parent()->AddInstruction(
	    HloInstruction::CreateCustomCall(inst->shape(), {inst->mutable_operand(0)},
									     "identity"));
    inst->ReplaceAllUsesWith(identity);
    inst->parent()->RemoveInstruction(inst);

     // Set the input and output as alias
     std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
         aliasing;

     ShapeUtil::VisitorFunction visitor = [&](const Shape& shape,
                                              const ShapeIndex& idx) {
       aliasing.push_back(std::make_pair(idx, std::make_pair(0, idx)));
     };
     ShapeUtil::ForEachSubshape(identity->shape(), visitor);
     HloCustomCallInstruction* call = Cast<HloCustomCallInstruction>(identity);
     call->set_output_to_operand_aliasing(aliasing);
  }

  return !barriers.empty();
}

}  // namespace xla
