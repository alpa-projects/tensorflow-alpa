#include "tensorflow/compiler/xla/service/remat_identity_fixer.h"

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

StatusOr<bool> RematIdentityFixer::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    std::vector<HloInstruction*> to_remove;

    for (HloInstruction* ins : computation->instructions()) {
      // Only modify the end region makrer
      if (ins->IsCustomCall("identity") &&
          ins->metadata().op_type() == "remat_end") {
        HloInstruction* tuple = ins->mutable_operand(0);
        CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);

        absl::flat_hash_map<int, HloInstruction*> index_to_allreduce;

        // Before:
        // r = reduce-scatter(a)
        // t = tuple(r)
        // c = custom-call(tuple)
        // y = get-tuple-element(c, index=0)
        //
        // After:
        // t = tuple(a)
        // c = custom-call(tuple)
        // x = get-tuple-element(c, index=0)
        // y = reduce-scatter(x)

        // Move all-reduce out of identity marker
        for (int64_t i = 0; i < tuple->operand_count(); i++) {
          HloInstruction* operand = tuple->mutable_operand(i);

          if (operand->opcode() == HloOpcode::kAllReduce ||
              operand->opcode() == HloOpcode::kReduceScatter) {
            changed = true;

            HloInstruction* to_allreduce = operand->mutable_operand(0);
            operand->ReplaceAllUsesWithDifferentShape(to_allreduce);
            *(tuple->mutable_shape()->mutable_tuple_shapes(i)) =
                to_allreduce->shape();
            *(ins->mutable_shape()->mutable_tuple_shapes(i)) =
                to_allreduce->shape();
            index_to_allreduce[i] = operand;
          }
        }

        for (auto user : ins->users()) {
          CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
          int index = user->tuple_index();
          auto iter = index_to_allreduce.find(index);
          if (iter != index_to_allreduce.end()) {
            HloInstruction* allreduce = iter->second;
            allreduce->ReplaceOperandWith(
                0, ins->parent()->AddInstruction(
                       HloInstruction::CreateGetTupleElement(
                           allreduce->operand(0)->shape(), ins, index)));
            user->ReplaceAllUsesWith(allreduce);
            to_remove.push_back(user);
          }
        }
      }
    }

    for (auto x : to_remove) {
      computation->RemoveInstruction(x);
    }
  }

  return changed;
}

}  // namespace xla
