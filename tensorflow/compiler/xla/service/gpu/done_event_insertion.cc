#include "tensorflow/compiler/xla/service/gpu/done_event_insertion.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace {
const char* const kDoneEvent = "__builtin$DoneEvent";
// An element makes the liveness analysis lift the kernel as early as possible.
const Shape DummyShape = ShapeUtil::MakeScalarShape(S32);

Status AddDoneEvent(HloInstruction* inst, size_t index, HloInstruction* root) {
  if (inst->opcode() == HloOpcode::kBitcast) {
    for (HloInstruction* op : inst->mutable_operands()) {
      TF_RETURN_IF_ERROR(AddDoneEvent(op, index, root));
    }
  } else if (inst->opcode() == HloOpcode::kGetTupleElement) {
    HloInstruction* tuple = inst->mutable_operand(0);
    if (tuple->opcode() == HloOpcode::kTuple) {
      TF_RETURN_IF_ERROR(AddDoneEvent(
          tuple->mutable_operand(inst->tuple_index()), index, root));
    } else {
      // Instructions like fusion generates a tuple
      TF_RETURN_IF_ERROR(AddDoneEvent(inst->mutable_operand(0), index, root));
    }
  } else {
    auto done_event = Cast<HloCustomCallInstruction>(
        inst->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            DummyShape, {inst}, kDoneEvent, std::to_string(index))));
    done_event->set_custom_call_has_side_effect(true);
    if (root != inst) {
      TF_RETURN_IF_ERROR(done_event->AddControlDependencyTo(root));
    }
  }
  return Status::OK();
}
};  // namespace

StatusOr<bool> HloDoneInsertion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();

  if (root->opcode() == HloOpcode::kTuple) {
    for (size_t i = 0; i < root->operand_count(); ++i) {
      TF_RETURN_IF_ERROR(AddDoneEvent(root->mutable_operand(i), i, root));
    }
  } else {
    TF_RETURN_IF_ERROR(AddDoneEvent(root, 0, root));
  }
  return changed;
}
};  // namespace xla
