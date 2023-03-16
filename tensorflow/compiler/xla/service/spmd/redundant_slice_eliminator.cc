#include "tensorflow/compiler/xla/service/spmd/redundant_slice_eliminator.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace spmd {

StatusOr<bool> RedundantSliceEliminator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  const int64_t num_devices = module->config().num_partitions();

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* ins : computation->instructions()) {
      if (ins->opcode() == HloOpcode::kDynamicSlice) {
        HloInstruction* operand = ins->mutable_operand(0);
        if (operand->opcode() == HloOpcode::kBroadcast) {
          // Check whether all sliced dims are broadcasted.
          // If so, this slice is redundant.
          bool is_redundant = true;
          for (size_t i = 0; i < ins->shape().rank(); ++i) {
            if (ins->shape().dimensions(i) != operand->shape().dimensions(i)) {
              if (absl::c_linear_search(operand->dimensions(), i)) {
                is_redundant = false;
              }
            }
          }

          if (is_redundant) {
            changed = true;
            HloInstruction* new_ins =
                ins->parent()->AddInstruction(HloInstruction::CreateBroadcast(
                    ins->shape(), operand->mutable_operand(0),
                    operand->dimensions()));
            ins->ReplaceAllUsesWith(new_ins);
          }
        } else if (false && // Temporarily disable this because it is incompatible with
                            // ReduceScatterCreator. We need to adjust the order of passes.
                   operand->opcode() == HloOpcode::kConstant &&
                   operand->shape().rank() == 1 &&
                   operand->shape().dimensions(0) > 1 &&
                   operand->shape().IsInteger() &&
                   ins->dynamic_slice_sizes().size() == 1 &&
                   ins->dynamic_slice_sizes().front() == 1) {
          const Literal& literal = operand->literal();
          int64_t size = operand->shape().dimensions(0);

          // Try pattern 1:  a * (partition-id % k) + b
          bool match = false;
          int64_t k, a, b;
          for (k = 1; k < size; k++) {
            if (num_devices % k == 0) {
              match = true;
              a = literal.Get<int32_t>({1}) - literal.Get<int32_t>({0});
              b = literal.Get<int32_t>({0});
              for (int64_t i = 0; i < size; ++i) {
                if (literal.Get<int32_t>({i}) != a * (i % k) + b) {
                  match = false;
                  break;
                }
              }

              if (match) {
                break;
              }
            }
          }

          if (match) {
            changed = true;

            HloInstruction* k_ins =
                ins->parent()->AddInstruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<int32_t>(k)));
            HloInstruction* convert_ins =
                ins->parent()->AddInstruction(HloInstruction::CreateConvert(
                    k_ins->shape(), ins->mutable_operand(1)));
            HloInstruction* mod_ins =
                ins->parent()->AddInstruction(HloInstruction::CreateBinary(
                    k_ins->shape(), HloOpcode::kRemainder, convert_ins, k_ins));

            HloInstruction* mul_ins;
            if (a != 1) {
              HloInstruction* a_ins =
                  ins->parent()->AddInstruction(HloInstruction::CreateConstant(
                      LiteralUtil::CreateR0<int32_t>(a)));
              mul_ins =
                  ins->parent()->AddInstruction(HloInstruction::CreateBinary(
                      k_ins->shape(), HloOpcode::kMultiply, a_ins, mod_ins));
            } else {
              mul_ins = mod_ins;
            }

            HloInstruction* add_ins;
            if (b != 0) {
              HloInstruction* b_ins =
                  ins->parent()->AddInstruction(HloInstruction::CreateConstant(
                      LiteralUtil::CreateR0<int32_t>(b)));
              add_ins =
                  ins->parent()->AddInstruction(HloInstruction::CreateBinary(
                      k_ins->shape(), HloOpcode::kAdd, mul_ins, b_ins));
            } else {
              add_ins = mul_ins;
            }

            if (ins->shape().element_type() != add_ins->shape().element_type()) {
              Shape copy_shape = add_ins->shape();
              copy_shape.set_element_type(ins->shape().element_type());
              add_ins = ins->parent()->AddInstruction(HloInstruction::CreateConvert(
                  copy_shape, add_ins));
            }

            HloInstruction* reshape_ins = ins->parent()->AddInstruction(
                HloInstruction::CreateReshape(ins->shape(), add_ins));
            ins->ReplaceAllUsesWith(reshape_ins);
          }

          // Try pattern 2: a * (partition-id / k)
          // TODO(lmzheng): match this pattern
        }
      }
    }
  }

  if (changed) {
    TF_RETURN_IF_ERROR(HloDCE().Run(module).status());
  }

  return changed;
}

}  // namespace spmd
}  // namespace xla
