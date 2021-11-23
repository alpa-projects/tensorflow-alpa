#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace spmd {

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

    HloInstruction* allreduce_ins = add_ins->mutable_operand(1);

    if (allreduce_ins->opcode() != HloOpcode::kAllReduce) {
      continue;
    }

    CHECK(allreduce_ins->operand_count() == 1);

    add_ins->ReplaceOperandWith(1, allreduce_ins->mutable_operand(0));
    allreduce_ins->ReplaceOperandWith(0, add_ins);
    output_tuple->ReplaceOperandWith(i, allreduce_ins);
  }

  // std::cerr << "===== Exit GradAccRewrite =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

void DfsSearch(const HloInstruction* cur, absl::flat_hash_set<int>& ret) {
  switch (cur->opcode()) {
    case HloOpcode::kTuple:
    case HloOpcode::kSlice:
    case HloOpcode::kBitcast: {
      for (size_t i = 0; i < cur->operand_count(); ++i) {
        DfsSearch(cur->operand(i), ret);
      }
      break;
    }
    case HloOpcode::kAllReduce: {
      ret.insert(cur->channel_id().value());
      break;
    }
    default:
      break;
  }
}

std::string GetGradSyncChannelIds(const HloModule* module,
                                  absl::optional<std::vector<int>> grad_idx) {
  absl::flat_hash_set<int> channel_ids;

  HloInstruction* root = module->entry_computation()->root_instruction();
  if (grad_idx) {
    CHECK(root->opcode() == HloOpcode::kTuple) << "The root inst is not tuple";
    for (int idx : grad_idx.value()) {
      DfsSearch(root->operand(idx), channel_ids);
    }
  } else {
    DfsSearch(root, channel_ids);
  }

  std::string ret = ".";
  for (auto x : channel_ids) {
    ret += std::to_string(x) + ".";
  }

  return ret;
}

}  // namespace spmd
}  // namespace xla
