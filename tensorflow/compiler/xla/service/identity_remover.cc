#include "tensorflow/compiler/xla/service/identity_remover.h"

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

StatusOr<bool> IdentityRemover::Run(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    std::vector<HloInstruction*> to_remove;

    for (HloInstruction* ins : computation->instructions()) {
      if (ins->IsCustomCall("identity")) {
        static_cast<HloCustomCallInstruction*>(ins)->set_custom_call_schedule(
            CustomCallSchedule::SCHEDULE_LATEST);

        HloInstruction* tuple = ins->mutable_operand(0);
        if (tuple->opcode() == HloOpcode::kTuple) {
          for (HloInstruction* user : ins->users()) {
            to_remove.push_back(user);
            user->ReplaceAllUsesWith(
                tuple->mutable_operand(user->tuple_index()));
          }
          to_remove.push_back(ins);
          to_remove.push_back(tuple);
        }
      }
    }

    for (HloInstruction* ins : to_remove) {
      changed = true;
      computation->RemoveInstruction(ins);
    }
  }

  return changed;
};

}  // namespace xla
