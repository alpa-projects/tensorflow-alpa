#include "tensorflow/compiler/xla/service/spmd/alpa_compile.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_spmd_partitioner.h"  // TODO(yonghao): remove it
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"  // TODO(yonghao): remove it
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
#include "tensorflow/compiler/xla/service/spmd/grad_acc_rewrite.h"
#include "tensorflow/compiler/xla/service/spmd/slice_auto_sharded_stages.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"

namespace xla {

namespace spmd {

const char kBeforeAutoShardingDumpName[] = "before_auto_sharding";
const char kBeforeSpmdPartitionDumpName[] = "before_spmd_partitioning";
// TODO(yonghao): Check correctness of compile options and modules
Status PreCompileCheck(const XlaComputation& computation,
                       CompileOptions options) {
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

StatusOr<std::shared_ptr<xla::HloModule>> CreateHloModule(
    const XlaComputation& computation, CompileOptions options) {
  PreCompileCheck(computation, options);

  const HloModuleProto& module_proto = computation.proto();
  // The device ordinal of the option might be -1, but it will not be used.
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const ProgramShape program_shape(module_proto.host_program_shape());
  const ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromProto(
          module_proto, build_options.debug_options(), &execution_options));

  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      CreateModuleFromProto(
                          module_proto, module_config,
                          options.executable_build_options.run_backend_only()));
  return hlo_module;
}

StatusOr<std::shared_ptr<xla::HloModule>> RunAutoShardingPass(
    const XlaComputation& computation, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      CreateHloModule(computation, options));
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

      spmd_pipeline.AddPass<xla::spmd::AutoSharding>();
      spmd_pipeline.AddPass<xla::spmd::SliceAutoShardedStages>();

      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true);
      spmd_pipeline.AddPass<gpu::GpuSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
      spmd_pipeline.AddPass<xla::spmd::GradAccRewrite>();
    } else {
      spmd_pipeline.AddPass<xla::spmd::SliceAutoShardedStages>();
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module.get()).status());
  }
  return hlo_module;
}

StatusOr<std::shared_ptr<HloModule>> RunSpmdPartitionerPass(
    const XlaComputation& computation, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> hlo_module,
                      CreateHloModule(computation, options));
  DumpHloModuleIfEnabled(*hlo_module, kBeforeSpmdPartitionDumpName);
  // TODO(yonghao): TF Profiler Traceme
  if (hlo_module->config().use_spmd_partitioning()) {
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    const int64_t num_partitions = hlo_module->config().num_partitions();
    if (num_partitions > 1) {
      spmd_pipeline.AddPass<ShardingPropagation>(/*is_spmd=*/true);
      spmd_pipeline.AddPass<gpu::GpuSpmdPartitioner>(
          num_partitions, hlo_module->config().replica_count());
      spmd_pipeline.AddPass<xla::spmd::GradAccRewrite>();
    } else {
      spmd_pipeline.AddPass<xla::spmd::SliceAutoShardedStages>();
      // Remove redundant sharding ops when partition_count == 1.
      spmd_pipeline.AddPass<ShardingRemover>();
      spmd_pipeline.AddPass<HloDCE>();
    }
    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module.get()).status());
  }
  return hlo_module;
}
};  // namespace spmd

};  // namespace xla