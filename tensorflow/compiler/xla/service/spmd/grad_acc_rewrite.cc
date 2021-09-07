#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace spmd {

StatusOr<bool> GradAccRewrite::Run(HloModule* module) {
  if (!pass_context::GetBool("auto_sharding::rewrite_for_grad_acc", false)) {
    return false;
  }

  //std::cerr << "===== Enter GradAccRewrite =====" << std::endl;
  //std::cerr << module->ToString();
  //std::cerr << "=====================================" << std::endl;

  HloComputation* entry = module->entry_computation();
  HloInstruction* output_tuple = entry->root_instruction();

  for (size_t i = 0; i < output_tuple->operand_count(); ++i) {
    HloInstruction* add_ins = output_tuple->mutable_operand(i);
    if (add_ins->opcode() != HloOpcode::kAdd) { continue; }

    HloInstruction* allreduce_ins = add_ins->mutable_operand(1);

    if (allreduce_ins->opcode() != HloOpcode::kAllReduce) { continue; }

    CHECK(allreduce_ins->operand_count() == 1);

    add_ins->ReplaceOperandWith(1, allreduce_ins->mutable_operand(0));
    allreduce_ins->ReplaceOperandWith(0, add_ins);
    output_tuple->ReplaceOperandWith(i, allreduce_ins);
  }

  //std::cerr << "===== Exit GradAccRewrite =====" << std::endl;
  //std::cerr << module->ToString();
  //std::cerr << "=====================================" << std::endl;

  return true;
}

}  // namespace spmd
}  // namespace xla
