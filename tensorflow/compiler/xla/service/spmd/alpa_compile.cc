#include "tensorflow/compiler/xla/service/spmd/alpa_compile.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_util.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"
#include "tensorflow/compiler/xla/service/spmd/redundant_slice_eliminator.h"
#include "tensorflow/compiler/xla/service/spmd/slice_auto_sharded_stages.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"

namespace xla {
namespace spmd {

const char kBeforeAutoShardingDumpName[] = "before_run_auto_sharding";
const char kBeforeSpmdPartitionDumpName[] = "before_run_spmd_partitioner";

// TODO(yonghao): Check correctness of compile options and modules
Status PreCompileCheck(const CompileOptions& options) {
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  if (build_options.has_device_assignment()) {
    if (build_options.device_assignment().replica_count() !=
        build_options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).\n%s",
          build_options.device_assignment().replica_count(),
          build_options.num_replicas(),
          build_options.device_assignment().ToString());
    }
    // TODO(yonghao): for TPU, computation count != 1 is unsupported
    if (build_options.device_assignment().computation_count() !=
        build_options.num_partitions()) {
      return InvalidArgument(
          "Mismatched number of partitions for device "
          "assignment and computation (%d vs %d).\n%s",
          build_options.device_assignment().computation_count(),
          build_options.num_partitions(),
          build_options.device_assignment().ToString());
    }
  }

  return Status::OK();
}

StatusOr<HloModuleConfig> CreateHloModuleConfig(const HloModule* hlo_module,
                                                const CompileOptions options) {
  PreCompileCheck(options);

  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const ProgramShape& program_shape =
      hlo_module->entry_computation_layout().ComputeProgramShape();
  const ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromShape(
          program_shape, build_options.debug_options(), &execution_options));
  return module_config;
}

Status RunAutoShardingPass(HloModule* hlo_module,
                           const CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      CreateHloModuleConfig(hlo_module, options));
  hlo_module->set_config(module_config);
  DumpHloModuleIfEnabled(*hlo_module, kBeforeAutoShardingDumpName);

  // TODO(yonghao): TF Profiler Traceme
  const DebugOptions& debug_options = hlo_module->config().debug_options();
  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      // Run some IR cleanup passes before running the SPMD partitioning passes.
      spmd_pipeline.AddInvariantChecker<HloVerifier>(
          /*layout_sensitive=*/false,
          /*allow_mixed_precision=*/false);
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<DotDecomposer>();
      spmd_pipeline.AddPass<ZeroSizedHloElimination>();
      spmd_pipeline.AddPass<ConditionalCanonicalizer>();

      HloPassPipeline& spmd_simplify =
          spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

      AlgebraicSimplifierOptions options;
      options.set_replace_transpose_with_bitcast(false);
      options.set_enable_conv_operand_swap(false);
      // "slow" minmax means we propagate nan.
      options.set_minmax_propagate_nan(
          !debug_options.xla_gpu_enable_fast_min_max());
      options.set_enable_dot_strength_reduction(false);
      spmd_simplify.AddPass<AlgebraicSimplifier>(options);

      spmd_simplify.AddPass<SortSimplifier>();
      spmd_simplify.AddPass<TupleSimplifier>();
      spmd_simplify.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);
      spmd_simplify.AddPass<GatherExpander>(
          GatherExpander::kEliminateSimpleGathers);
      spmd_simplify.AddPass<WhileLoopConstantSinking>();
      spmd_simplify.AddPass<WhileLoopSimplifier>();

      spmd_simplify.AddPass<ReshapeMover>();
      spmd_simplify.AddPass<HloConstantFolding>();
      spmd_simplify.AddPass<ConditionalSimplifier>();
      spmd_simplify.AddPass<TransposeFolding>(
          [](const HloInstruction& dot,
             const TransposeFolding::OperandIndices& candidate_operands) {
            return gpu::IsMatrixMultiplication(dot)
                       ? candidate_operands
                       : TransposeFolding::OperandIndices{};
          });
      spmd_simplify.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
      spmd_simplify.AddPass<HloDCE>();

      spmd_pipeline.AddPass<AutoSharding>();
      spmd_pipeline.AddPass<ShardingPropagation>(
          /*is_spmd=*/true, /*propagate_metadata=*/false,
          /*allow_spmd_sharding_propagation_to_output=*/true);
      spmd_pipeline.AddPass<SliceAutoShardedStages>();
      spmd_pipeline.AddPass<StatefulRngSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
      spmd_pipeline.AddPass<RedundantSliceEliminator>();
      spmd_pipeline.AddPass<AllReduceReassociate>();
      spmd_pipeline.AddPass<GradAccRewrite>();
    } else {
      spmd_pipeline.AddPass<SliceAutoShardedStages>();
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  return Status::OK();
}

Status RunSpmdPartitionerPass(HloModule* hlo_module,
                              const CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      CreateHloModuleConfig(hlo_module, options));
  hlo_module->set_config(module_config);

  DumpHloModuleIfEnabled(*hlo_module, kBeforeSpmdPartitionDumpName);

  // TODO(yonghao): TF Profiler Traceme
  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      spmd_pipeline.AddPass<StatefulRngSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
      spmd_pipeline.AddPass<RedundantSliceEliminator>();
      spmd_pipeline.AddPass<AllReduceReassociate>();
      spmd_pipeline.AddPass<GradAccRewrite>();
    } else {
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  return Status::OK();
}

Status SetHloModuleOutputShardings(
    HloModule* module, const std::vector<OpSharding>& op_shardings) {
  // Run TupleSimplifier pass to remove redundant tuples.
  // Otherwise, these redundant tuples and other custom call markers together
  // will make the propagation generate unexpected results.
  TupleSimplifier tuple_simplifier;
  tuple_simplifier.Run(module);

  // Set the sharding for the output tuple
  HloComputation* entry = module->entry_computation();
  HloInstruction* output_tuple = entry->root_instruction();

  ShapeTree<HloSharding> tuple_sharding(output_tuple->shape(),
                                        HloSharding::Replicate());
  CHECK_EQ(tuple_sharding.leaf_count(), op_shardings.size());

  size_t i = 0;
  for (auto& leaf : tuple_sharding.leaves()) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(op_shardings[i++]));
    leaf.second = hlo_sharding;
  }
  output_tuple->set_sharding(HloSharding::Tuple(tuple_sharding));

  if (output_tuple->IsCustomCall(kPipelineMarker) ||
      output_tuple->IsCustomCall(kIdentityMarker)) {
    // Also set the operand. Otherwise, the propagation generates unexpected
    // results.
    output_tuple->mutable_operand(0)->set_sharding(
        HloSharding::Tuple(tuple_sharding));
  }

  return Status::OK();
}

Status SetHloModuleInputShardings(HloModule* module,
                                  const std::vector<OpSharding>& op_shardings) {
  // Run TupleSimplifier pass to remove redundant tuples.
  // Otherwise, these redundant tuples and other custom call markers together
  // will make the propagation generate unexpected results.
  TupleSimplifier tuple_simplifier;
  tuple_simplifier.Run(module);

  HloComputation* entry = module->entry_computation();
  std::vector<HloInstruction*> input_insts = entry->parameter_instructions();
  CHECK_EQ(input_insts.size(), op_shardings.size());

  size_t i = 0;
  for (auto& inst : input_insts) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding, HloSharding::FromProto(op_shardings[i++]));
    if (IsUndefined(hlo_sharding)) {
      continue;
    }
    inst->set_sharding(HloSharding::Single(inst->shape(), hlo_sharding));
  }

  return Status::OK();
}

};  // namespace spmd
};  // namespace xla
