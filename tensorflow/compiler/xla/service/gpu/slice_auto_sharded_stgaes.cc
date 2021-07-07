#include "tensorflow/compiler/xla/service/gpu/auto_sharding.h"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace gpu {

namespace py = pybind11;

enum VisitState { kVisiting, kVisited };

std::vector<HloInstruction*> GetAncestorInstructions(
    HloInstruction* start_ins) {
  std::vector<HloInstruction*> postorder;
  absl::flat_hash_map<HloInstruction*, VisitState> visited;
  std::vector<HloInstruction*> dfs_stack;
  dfs_stack.push_back(start_ins);

  while (!dfs_stack.empty()) {
    auto* cur = dfs_stack.back();
    auto it = visited.find(cur);
    if (it != visited.end()) {
      dfs_stack.pop_back();
      if (it->second == kVisited) {
        continue;
      }
      CHECK_EQ(it->second, kVisiting);
      postorder.push_back(cur);
      it->second = kVisited;
      continue;
    }

    visited.insert({cur, kVisiting});
    for (HloInstruction* operand : cur->operands()) {
      dfs_stack.push_back(operand);
    }
  }
  return postorder;
}

std::unique_ptr<HloModule> CreateStageModule(
    HloModule* full_module,
    const std::vector<HloInstruction*>& stage_instructions,
    std::string suffix) {
  HloModuleConfig config = full_module->config();
  // TODO (zhuohan): Support input/output alias
  config.set_shardable_value_update_pairs({});
  config.mutable_fusion_config()->clear();
  config.mutable_dot_config()->clear();
  config.mutable_layout_config()->clear();

  auto module = absl::make_unique<HloModule>(
      absl::StrCat(full_module->name(), "-", suffix), config);

  std::unique_ptr<HloCloneContext> context_ptr =
      absl::make_unique<HloCloneContext>(module.get(), suffix);
  HloCloneContext* context = context_ptr.get();

  std::vector<std::unique_ptr<HloInstruction>> instructions;

  HloInstruction* stage_start_instruction = stage_instructions.front();
  HloInstruction* stage_end_instruction = stage_instructions.back();
  CHECK(stage_start_instruction->IsCustomCall("xla_pipeline_marker"));
  CHECK(stage_end_instruction->IsCustomCall("xla_pipeline_marker"));

  CHECK(stage_start_instruction->shape().IsTuple());
  size_t n_parameters = stage_start_instruction->shape().tuple_shapes_size();
  std::vector<HloInstruction*> parameters(n_parameters);
  for (size_t i = 0; i < n_parameters; ++i) {
    auto new_param = HloInstruction::CreateParameter(
        i, stage_start_instruction->shape().tuple_shapes(i),
        absl::StrCat("param_", i));
    if (stage_start_instruction->has_sharding()) {
      CHECK(stage_start_instruction->sharding().IsTuple());
      new_param->set_sharding(
          stage_start_instruction->sharding().GetSubSharding(
              stage_start_instruction->shape(), {i}));
    }
    new_param->set_metadata(stage_start_instruction->metadata());
    parameters[i] = new_param.get();
    instructions.push_back(std::move(new_param));
  }

  // std::cerr << "======old instructions=====" << std::endl;
  for (size_t i = 1; i < stage_instructions.size() - 1; ++i) {
    HloInstruction* ins = stage_instructions[i];
    // std::cerr << ins->ToString() << std::endl;
    if (ins->opcode() == HloOpcode::kGetTupleElement &&
        ins->operand(0) == stage_start_instruction) {
      int64 param_no = ins->tuple_index();
      context->MapInstruction(ins, parameters[param_no]);
    } else {
      CHECK_NE(ins->opcode(), HloOpcode::kParameter)
          << "instructions in a stage should not be parameter"
          << ins->ToString();
      std::vector<HloInstruction*> new_operands;
      for (auto operand : ins->operands()) {
        HloInstruction* new_operand = context->FindInstruction(operand);
        if (new_operand == nullptr) {
          std::vector<HloInstruction*> ancestors =
              GetAncestorInstructions(operand);
          for (auto ancestor : ancestors) {
            // Make sure the ancestor is a constant
            // TODO (zhuohan): Might also check that the opcode is not
            // kRngGetAndUpdateState
            CHECK_NE(ancestor->opcode(), HloOpcode::kParameter);
            std::vector<HloInstruction*> new_ancestor_operands;
            for (auto ancestor_operand : ancestor->operands()) {
              new_ancestor_operands.push_back(
                  context->GetInstruction(ancestor_operand));
            }
            instructions.push_back(ancestor->CloneWithNewOperands(
                ancestor->shape(), new_ancestor_operands, context));
          }
          new_operand = context->GetInstruction(operand);
        }
        new_operands.push_back(new_operand);
      }
      instructions.push_back(
          ins->CloneWithNewOperands(ins->shape(), new_operands, context));
    }
  }
  // std::cerr << "======new instructions=====" << std::endl;
  // for (const auto &ins : instructions) {
  //   std::cerr << ins->ToString() << std::endl;
  // }
  HloComputation::Builder builder(
      absl::StrCat(full_module->entry_computation()->name(), "-", suffix));
  for (auto& ins : instructions) {
    builder.AddInstruction(std::move(ins));
  }
  std::unique_ptr<HloComputation> new_computation = builder.Build(
      /*root_instruction=*/context->GetInstruction(
          stage_end_instruction->operand(0)));

  for (size_t i = 1; i < stage_instructions.size() - 1; ++i) {
    HloInstruction* ins = stage_instructions[i];
    HloInstruction* new_ins = context->GetInstruction(ins);
    for (auto successor : ins->control_successors()) {
      TF_CHECK_OK(
          new_ins->AddControlDependencyTo(context->GetInstruction(successor)));
    }
  }

  // std::cerr << "======new computation=====" << std::endl;
  // std::cerr << new_computation->ToString() << std::endl;

  // NOTE: We assume the HLO graph only has one computation.
  module->AddEntryComputationWithLayouts(std::move(new_computation));

  return std::move(module);
}

std::vector<std::unique_ptr<HloModule>> SliceAutoShardedStagesInternal(
    HloModule* module) {
  // ----- Slice the hlo module according to the pipeline marker -----
  // TODO (zhuohan): Move this into a seperate pass
  HloComputation* entry = module->entry_computation();

  std::vector<std::unique_ptr<HloModule>> pipeline_stages;
  std::vector<HloInstruction*> current_stage_instructions;
  std::vector<HloInstruction*> post_order = entry->MakeInstructionPostOrder();
  bool in_stage = false;
  for (HloInstruction* current_ins : post_order) {
    if (current_ins->IsCustomCall("xla_pipeline_marker")) {
      if (in_stage) {
        current_stage_instructions.push_back(current_ins);
        pipeline_stages.push_back(
            CreateStageModule(module, current_stage_instructions,
                              std::to_string(pipeline_stages.size())));
        current_stage_instructions.clear();
        in_stage = false;
      } else {
        in_stage = true;
        current_stage_instructions.push_back(current_ins);
      }
    } else if (in_stage) {
      current_stage_instructions.push_back(current_ins);
    }
  }

  // ----- Put the sharded HLO module back to Python -----
  HloModuleProto module_proto = module->ToProto();
  std::string serilaized_module_proto;
  CHECK(module_proto.SerializeToString(&serilaized_module_proto));

  PyGILState_STATE gstate = PyGILState_Ensure();
  {
    py::object submodule = py::module_::import("parax.auto_sharding");
    py::list stage_modules;
    for (const auto& stage_module : pipeline_stages) {
      HloModuleProto module_proto = stage_module->ToProto();
      std::string serilaized_module_proto;
      CHECK(module_proto.SerializeToString(&serilaized_module_proto));
      py::bytes serilaized_module_proto_bytes(serilaized_module_proto);
      stage_modules.append(serilaized_module_proto_bytes);
    }
    py::object set_auto_sharded_hlo_stages =
        submodule.attr("set_auto_sharded_hlo_stages");
    py::object ret = set_auto_sharded_hlo_stages(stage_modules);
    if (!ret.is_none()) {
      PyGILState_Release(gstate);
      exit(-1);
    }
  }
  PyGILState_Release(gstate);

  return pipeline_stages;
}

StatusOr<bool> SliceAutoShardedStages::Run(HloModule* module) {
  SliceAutoShardedStagesInternal(module);
  return false;
}

}  // namespace gpu
}  // namespace xla
