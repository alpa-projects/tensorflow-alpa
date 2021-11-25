#include "tensorflow/compiler/xla/service/gpu/common_computation_elimination.h"

namespace xla {
namespace gpu {


template <class T>
inline void hash_combine(std::size_t& seed, const T& value) {
    seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2);
}


struct HloComputationPtrHash {
   size_t operator() (const HloComputation *computation) const {
     size_t ret = 0x1234;
     for (const HloInstruction* ins : computation->instructions()) {
       hash_combine(ret, int(ins->opcode()));
     }
     return ret;
   }
};

struct HloComputationPtrEqual {
   size_t operator() (const HloComputation *lhs, const HloComputation *rhs) const {
     return *lhs == *rhs;
   }
};

StatusOr<bool> CommonComputationElimination::Run(HloModule* module) {
  bool changed = false;

  absl::flat_hash_map<HloComputation*, HloComputation*,
                      HloComputationPtrHash, HloComputationPtrEqual> unique;
  absl::flat_hash_map<HloComputation*, HloComputation*> replace_with;

  if (module->computation_count() < 5) {
    return false;
  }

  //std::cerr << "===== Enter =====" << std::endl;
  //std::cerr << module->ToString() << std::endl;

  for (HloComputation* computation : module->computations()) {
    auto iter = unique.find(computation);
    if (iter == unique.end()) {
      unique[computation] = computation;
    } else {
      replace_with[computation] = iter->second;
    }
  }

  HloComputation* entry_computation = module->entry_computation();
  int64_t total_ct = 0, replaced_ct = 0;
  for (HloInstruction* ins : entry_computation->instructions()) {
    if (ins->opcode() == HloOpcode::kFusion) {
      total_ct++;

      HloComputation* src = ins->called_computations().front();
      auto iter = replace_with.find(src);
      if (iter != replace_with.end()) {
        //std::cerr << "replace " << src->name() << " with "  << iter->second->name() << std::endl;
        changed = true;
        replaced_ct++;
        ins->ReplaceCalledComputations([&](HloComputation* call_target){
          CHECK_EQ(call_target, src);
          return iter->second;
        });
        src->SetFusionInstruction(nullptr);
        module->RemoveEmbeddedComputation(src);
      }
    }
  }

  std::cerr << "Total fused ins: " << total_ct << ", Replaced fused ins: " << replaced_ct << std::endl;

  //std::cerr << "===== Exit =====" << std::endl;
  //std::cerr << module->ToString() << std::endl;

  return changed;
};

}  // namespace gpu
}  // namespace xla
