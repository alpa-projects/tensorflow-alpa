#include "tensorflow/compiler/xla/service/remat_identity_fixer.h"

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

StatusOr<bool> RematIdentityFixer::Run(HloModule* module) {
  bool changed = false;
  return changed;

  for (HloComputation* computation : module->computations()) {

    for (HloInstruction* ins : computation->instructions()) {
      // Only modify the end region makrer
      if (ins->IsCustomCall("identity") && ins->metadata().op_type() == "remat_end") {
        HloInstruction* tuple = ins->mutable_operand(0);
        CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);

        absl::flat_hash_map<int, HloInstruction*> index_to_allreduce;

        // Move all-reduce out of identity marker
        for (int64_t i = 0; i < tuple->operand_count(); i++) {
          HloInstruction* operand = tuple->mutable_operand(i);

          if (operand->opcode() == HloOpcode::kAllReduce) {
            changed = true;

            operand->ReplaceAllUsesWith(operand->mutable_operand(0));
            index_to_allreduce[i] = operand;
          }
        }

        for (auto user : ins->users()) {
          CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
          int index = user->tuple_index();
          auto iter = index_to_allreduce.find(index);
          if (iter != index_to_allreduce.end()) {
            HloInstruction* allreduce = iter->second;
            allreduce->ReplaceOperandWith(0, user);
            user->ReplaceAllUsesWith(allreduce);
          }
        }
      }
    }
  }

  return changed;
};

}  // namespace xla
