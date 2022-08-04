#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace spmd {

HloInstruction* GetAllReduce(HloInstruction* src) {
  auto opcode = src->opcode();
  if (opcode == HloOpcode::kAllReduce) {
    return src;
  } else if (opcode == HloOpcode::kConvert ||
             opcode == HloOpcode::kReshape ||
             opcode == HloOpcode::kCopy ||
             opcode == HloOpcode::kBitcast ||
             opcode == HloOpcode::kTranspose) {
    return GetAllReduce(src->mutable_operand(0));
  } else if (opcode == HloOpcode::kMultiply) {
    HloInstruction* lhs = GetAllReduce(src->mutable_operand(0));
    HloInstruction* rhs = GetAllReduce(src->mutable_operand(1));

    if (lhs != nullptr && rhs == nullptr) {
      return lhs;
    } else if (lhs == nullptr && rhs != nullptr) {
      return rhs;
    }
  }
  return nullptr;
}

StatusOr<bool> GradAccRewrite::Run(HloModule* module) {
  if (!pass_context::GetBool("auto_sharding::rewrite_for_grad_acc", false)) {
    return false;
  }

  // std::cerr << "===== Enter GradAccRewrite =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  auto indices = pass_context::GetIntVector("auto_sharding::rewrite_indices");

  HloComputation* entry = module->entry_computation();
  HloInstruction* output_tuple = entry->root_instruction();

  for (size_t i : indices) {
    HloInstruction* add_ins = output_tuple->mutable_operand(i);
    if (add_ins->opcode() != HloOpcode::kAdd) {
      continue;
    }

    HloInstruction* allreduce_ins = GetAllReduce(add_ins->mutable_operand(1));

    if (allreduce_ins == nullptr || allreduce_ins->users().size() != 1) {
      continue;
    }

    CHECK_EQ(allreduce_ins->operand_count(), 1);

    HloInstruction* allreduce_user = allreduce_ins->users().front();

    for (size_t i = 0; i < allreduce_user->operand_count(); ++i) {
      if (allreduce_user->operand(i) == allreduce_ins) {
        allreduce_user->ReplaceOperandWith(i,
                                           allreduce_ins->mutable_operand(0));
      }
    }

    allreduce_ins->ReplaceOperandWith(0, add_ins);
    output_tuple->ReplaceOperandWith(i, allreduce_ins);
    allreduce_ins->set_metadata_op_name(kSkippableAllReduce);
    allreduce_ins->mutable_shape()->set_element_type(add_ins->shape().element_type());
  }

  // std::cerr << "===== Exit GradAccRewrite =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

std::string GetGradSyncChannelIds(const HloModule* module) {
  std::string ret = ".";
  for (auto inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kAllReduce &&
        inst->metadata().op_name() == kSkippableAllReduce) {
      ret += std::to_string(inst->channel_id().value()) + ".";
    }
  }

  return ret;
}

}  // namespace spmd
}  // namespace xla
