#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace spmd {

HloInstruction* GetAllReduce(HloInstruction* src) {
  if (src->opcode() == HloOpcode::kAllReduce) {
    return src;
  } else if (src->opcode() == HloOpcode::kMultiply) {
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
  }

  // std::cerr << "===== Exit GradAccRewrite =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

void DfsSearch(const HloInstruction* cur,
               absl::flat_hash_set<const HloInstruction*>& touch_set,
               absl::flat_hash_set<const HloInstruction*>& allreduce_set) {
  switch (cur->opcode()) {
    case HloOpcode::kTuple:
    case HloOpcode::kSlice:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kConvert:
    case HloOpcode::kReshape:
    case HloOpcode::kBitcast: {
      touch_set.insert(cur);
      for (size_t i = 0; i < cur->operand_count(); ++i) {
        DfsSearch(cur->operand(i), touch_set, allreduce_set);
      }
      break;
    }
    case HloOpcode::kFusion: {
      // FIXME(lmzheng): we should check the instructions in the body
      // to make sure they are compatible with gradient accumulation.
      touch_set.insert(cur);
      for (size_t i = 0; i < cur->operand_count(); ++i) {
        DfsSearch(cur->operand(i), touch_set, allreduce_set);
      }
      break;
    }
    case HloOpcode::kAllReduce: {
      allreduce_set.insert(cur);
      break;
    }
    default:
      break;
  }
}

std::string GetGradSyncChannelIds(const HloModule* module,
                                  absl::optional<std::vector<int>> grad_idx) {
  absl::flat_hash_set<const HloInstruction*> touch_set;
  absl::flat_hash_set<const HloInstruction*> allreduce_set;

  // std::cerr << "===== Enter GetGradSync =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=============================" << std::endl;

  HloInstruction* root = module->entry_computation()->root_instruction();
  touch_set.insert(root);
  if (grad_idx) {
    CHECK(root->opcode() == HloOpcode::kTuple ||
          root->opcode() == HloOpcode::kAllReduce)
        << "The root inst is not tuple";
    for (int idx : grad_idx.value()) {
      DfsSearch(root->operand(idx), touch_set, allreduce_set);
    }
  } else {
    DfsSearch(root, touch_set, allreduce_set);
  }

  std::string ret = ".";
  for (auto inst : allreduce_set) {
    for (auto user : inst->users()) {
      //CHECK(touch_set.count(user))
      //    << "Invalid users of all-reduce in gradient accumulation. "
      //    << user->ToString();
    }
    ret += std::to_string(inst->channel_id().value()) + ".";
  }

  return ret;
}

}  // namespace spmd
}  // namespace xla
