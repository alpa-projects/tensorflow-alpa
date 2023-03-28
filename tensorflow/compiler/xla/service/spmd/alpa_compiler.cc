#include "tensorflow/compiler/xla/service/spmd/alpa_compiler.h"

#include "tensorflow/compiler/xla/service/spmd/auto_sharding.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"
#include "tensorflow/compiler/xla/service/spmd/redundant_slice_eliminator.h"
#include "tensorflow/compiler/xla/service/spmd/slice_auto_sharded_stages.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"

// Copied from gpu_compiler.cc
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dot_merger.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gather_simplifier.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_shape_verifier.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/hlo/transforms/hlo_constant_splitter.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/scatter_simplifier.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"

namespace xla {

namespace {
// Adds the HloVerifier for GPU to the given pipeline.
void AddHloVerifier(HloPassPipeline* pipeline, HloVerifierOpts&& opts = {},
                    bool debug_only = false) {
  std::unique_ptr<TargetVerifierMetadata> verifier_metadata =
      std::make_unique<GpuVerifierMetadata>(std::move(opts));
  if (debug_only) {
    pipeline->AddInvariantCheckerDebug<HloVerifier>(
        std::move(verifier_metadata), "hlo verifier (debug)");
  } else {
    pipeline->AddInvariantChecker<HloVerifier>(std::move(verifier_metadata),
                                               "hlo verifier");
  }
}

bool ConvIsLowerable(HloInstruction* conv) {
  return gpu::GpuConvRewriter::ConvIsLowerable(conv);
}
}  // namespace

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

  return OkStatus();
}

StatusOr<HloModuleConfig> CreateHloModuleConfig(const HloModule* hlo_module,
                                                const CompileOptions options) {
  TF_RETURN_IF_ERROR(PreCompileCheck(options));

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

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);
  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      !debug_options.xla_gpu_enable_fast_min_max());
  layout_insensitive_algsimp_opts.set_enable_dot_strength_reduction(false);  // Added by Alpa

  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("run-auto-sharding");
    AddHloVerifier(&spmd_pipeline);
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      // Run some IR cleanup passes before running the SPMD partitioning
      // passes.
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<DotDecomposer>();  // Added by Alpa
      spmd_pipeline.AddPass<ZeroSizedHloElimination>();
      spmd_pipeline.AddPass<ConditionalCanonicalizer>();

      HloPassPipeline& spmd_simplify =
          spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

      spmd_simplify.AddPass<AlgebraicSimplifier>(
          layout_insensitive_algsimp_opts);

      spmd_simplify.AddPass<SortSimplifier>();
      spmd_simplify.AddPass<TupleSimplifier>();
      // spmd_simplify.AddPass<ScatterSimplifier>();
      spmd_simplify.AddPass<ScatterExpander>(
          ScatterExpander::kEliminateSimpleScatters);
      // spmd_simplify.AddPass<GatherSimplifier>();
      spmd_simplify.AddPass<GatherExpander>(
          GatherExpander::kEliminateSimpleGathers);
      spmd_simplify.AddPass<WhileLoopConstantSinking>();
      spmd_simplify.AddPass<WhileLoopSimplifier>();

      spmd_simplify.AddPass<ReshapeMover>();
      spmd_simplify.AddPass<HloConstantFolding>();
      spmd_simplify.AddPass<ConditionalSimplifier>();
      spmd_simplify.AddPass<TransposeFolding>(
          gpu::CanFoldTransposeOperandIntoDot);  // Added by Alpa
      spmd_simplify.AddPass<HloCSE>(
          /*is_layout_sensitive=*/false);  // Added by Alpa
      spmd_simplify.AddPass<HloDCE>();

      spmd_pipeline.AddPass<HloConstantSplitter>();

      spmd_pipeline.AddPass<AutoSharding>();
      spmd_pipeline.AddPass<ShardingPropagation>(
          /*is_spmd=*/true, /*propagate_metadata=*/false,
          /*allow_spmd_sharding_propagation_to_output=*/absl::Span<const bool>{true});
      spmd_pipeline.AddPass<SliceAutoShardedStages>();
    } else {
      spmd_pipeline.AddPass<CallInliner>();
      spmd_pipeline.AddPass<SliceAutoShardedStages>();
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  }
  return OkStatus();
}

Status RunSpmdPartitionerPass(HloModule* hlo_module,
                              const CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      CreateHloModuleConfig(hlo_module, options));
  hlo_module->set_config(module_config);

  DumpHloModuleIfEnabled(*hlo_module, kBeforeSpmdPartitionDumpName);

  // TODO(yonghao): TF Profiler Traceme
  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("run-spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      spmd_pipeline.AddPass<ShardingPropagation>(
          /*is_spmd=*/true, /*propagate_metadata=*/false,
          /*allow_spmd_sharding_propagation_to_output=*/absl::Span<const bool>{true});
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
  return OkStatus();
}

Status SetHloModuleOutputShardings(
    HloModule* module, const std::vector<OpSharding>& op_shardings) {
  // Run some simplification passes to remove redundant tuples.
  // Otherwise, these redundant tuples and other custom call markers together
  // will make the propagation generate unexpected results.
  HloPassPipeline pipeline("set-sharding-pipeline");
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

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

  if (IsPassThroughTuple(output_tuple)) {
    // Also set the operand. Otherwise, the propagation generates unexpected
    // results.
    output_tuple->mutable_operand(0)->set_sharding(
        HloSharding::Tuple(tuple_sharding));
  }

  return OkStatus();
}

Status SetHloModuleInputShardings(HloModule* module,
                                  const std::vector<OpSharding>& op_shardings) {
  // Run some simplification passes to remove redundant tuples.
  // Otherwise, these redundant tuples and other custom call markers together
  // will make the propagation generate unexpected results.
  HloPassPipeline pipeline("set-sharding-pipeline");
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();
  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

  HloComputation* entry = module->entry_computation();
  std::vector<HloInstruction*> input_insts = entry->parameter_instructions();
  CHECK_EQ(input_insts.size(), op_shardings.size());

  size_t i = 0;
  for (auto& inst : input_insts) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(op_shardings[i++]));
    if (IsUndefined(hlo_sharding)) {
      continue;
    }
    inst->set_sharding(HloSharding::Single(inst->shape(), hlo_sharding));
  }

  return OkStatus();
}

};  // namespace spmd
};  // namespace xla
