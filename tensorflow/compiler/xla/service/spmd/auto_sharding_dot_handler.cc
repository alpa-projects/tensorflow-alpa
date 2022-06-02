#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

void AppendNewStrategy(const HloInstruction* ins, const std::string& name,
                       const HloSharding& output_spec,
                       const std::vector<HloSharding>& input_specs,
                       double compute_cost, double communication_cost,
                       const ClusterEnvironment& cluster_env,
                       const StrategyMap& strategy_map,
                       std::unique_ptr<StrategyVector>& strategies) {
  std::vector<std::vector<double>> resharding_costs;

  for (int i = 0; i < ins->operand_count(); ++i) {
    const HloInstruction* operand = ins->operand(i);
    resharding_costs.push_back(
        ReshardingCostVector(strategy_map.at(operand).get(), operand->shape(),
                             input_specs[i], cluster_env));
  }

  strategies->leaf_vector.push_back(ShardingStrategy({
      name,
      output_spec,
      compute_cost,
      communication_cost,
      GetBytes(ins->shape()) / output_spec.NumTiles(),
      resharding_costs,
      input_specs,
  }));
}

class DotHandler {
 public:
  DotHandler(std::unique_ptr<StrategyVector>& strategies,
             StrategyMap& strategy_map, const HloInstruction* ins,
             const ClusterEnvironment& cluster_env,
             const InstructionBatchDimMap& batch_map,
             const AutoShardingSolverOption& solver_option)
      : strategies(strategies),
        strategy_map(strategy_map),
        ins(ins),
        cluster_env(cluster_env),
        batch_map(batch_map),
        solver_option(solver_option),
        device_mesh(cluster_env.device_mesh),
        device_mesh_1d(cluster_env.device_mesh_1d),
        lhs(ins->operand(0)),
        rhs(ins->operand(1)),
        dot_dnums(ins->dot_dimension_numbers()),
        lhs_con_dims(dot_dnums.lhs_contracting_dimensions()),
        rhs_con_dims(dot_dnums.rhs_contracting_dimensions()),
        lhs_batch_dims(dot_dnums.lhs_batch_dimensions()),
        rhs_batch_dims(dot_dnums.rhs_batch_dimensions()) {
    std::tie(lhs_space_dims, rhs_space_dims) =
        GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);

    CHECK_EQ(lhs_space_dims.size(), 1) << ins->ToString();
    CHECK_EQ(rhs_space_dims.size(), 1) << ins->ToString();
    CHECK_EQ(lhs_con_dims.size(), 1);
    CHECK_EQ(rhs_con_dims.size(), 1);

    // The dimension in the output that corresponds to the lhs space dim or rhs
    // space dim
    out_lhs_space_dim = dot_dnums.lhs_batch_dimensions_size();
    out_rhs_space_dim = out_lhs_space_dim + 1;

    // Only support 2 dimensional device mesh
    CHECK_EQ(device_mesh.num_dimensions(), 2);
  }

  void SplitLhsSpaceRhsSpace(int mesh_dim0, int mesh_dim1) {
    if (ins->shape().dimensions(out_lhs_space_dim) <
            device_mesh.dim(mesh_dim0) ||
        ins->shape().dimensions(out_rhs_space_dim) <
            device_mesh.dim(mesh_dim1)) {
      return;  // Do not allow padding the output tensor
    }

    std::string name =
        absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1);
    HloSharding output_spec =
        Tile(ins->shape(), {out_lhs_space_dim, out_rhs_space_dim},
             {mesh_dim0, mesh_dim1}, device_mesh);
    HloSharding lhs_spec =
        Tile(lhs->shape(), {lhs_space_dims[0]}, {mesh_dim0}, device_mesh);
    HloSharding rhs_spec =
        Tile(rhs->shape(), {rhs_space_dims[0]}, {mesh_dim1}, device_mesh);

    AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                      cluster_env, strategy_map, strategies);
  }

  void SplitLhsSpaceBothContract(int mesh_dim0, int mesh_dim1) {
    if (lhs->shape().dimensions(out_lhs_space_dim) <
        device_mesh.dim(mesh_dim0)) {
      return;  // Do not allow padding the output tensor
    }

    if (device_mesh.dim(mesh_dim0) > 1 && device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim1);
      HloSharding output_spec =
          Tile(ins->shape(), {out_lhs_space_dim}, {mesh_dim0}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_space_dims[0], lhs_con_dims[0]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_con_dims[0]}, {mesh_dim1}, device_mesh);

      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim1);
      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env, strategy_map,
                        strategies);
    }
  }

  void SplitRhsSpaceBothContract(int mesh_dim0, int mesh_dim1) {
    if (ins->shape().dimensions(out_rhs_space_dim) <
        device_mesh.dim(mesh_dim1)) {
      return;  // Do not allow padding the output tensor
    }

    if (device_mesh.dim(mesh_dim0) > 1) {
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim0);
      HloSharding output_spec =
          Tile(ins->shape(), {out_rhs_space_dim}, {mesh_dim1}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_con_dims[0]}, {mesh_dim0}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_con_dims[0], rhs_space_dims[0]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim0);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env, strategy_map,
                        strategies);
    }
  }

  void SplitOneBatchDim() {
    if (device_mesh.dim(0) == 1 || device_mesh.dim(1) == 1) {
      for (int64_t i = 0; i < lhs_batch_dims.size(); ++i) {
        for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
          if (device_mesh.dim(j) == 1 ||
              ins->shape().dimensions(i) < device_mesh.dim(j)) {
            continue;
          }

          std::string name = absl::StrFormat("Sb_%d = Sb x Sb @ {%d}", i, j);
          HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);
          HloSharding lhs_spec =
              Tile(lhs->shape(), {lhs_batch_dims[i]}, {j}, device_mesh);
          HloSharding rhs_spec =
              Tile(rhs->shape(), {rhs_batch_dims[i]}, {j}, device_mesh);

          AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                            cluster_env, strategy_map, strategies);
        }
      }
    }
  }

  void SplitTwoBatchDims(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() == 2 && device_mesh.dim(mesh_dim0) > 1 &&
        device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("Sb = Sb x Sb @ {%d,%d}", mesh_dim0, mesh_dim1);
      HloSharding output_spec =
          Tile(ins->shape(), {0, 1}, {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_batch_dims[0], lhs_batch_dims[1]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_batch_dims[0], rhs_batch_dims[1]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                        cluster_env, strategy_map, strategies);
    }
  }

  void SplitBatchDimLhsSpace(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() > 0 && device_mesh.dim(mesh_dim0) > 1 &&
        device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SbSi = SbSi x SbR @ {%d,%d}", mesh_dim0, mesh_dim1);
      HloSharding output_spec = Tile(ins->shape(), {0, out_lhs_space_dim},
                                     {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_batch_dims[0], lhs_space_dims[0]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_batch_dims[0]}, {mesh_dim0}, device_mesh);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                        cluster_env, strategy_map, strategies);
    }
  }

  void SplitBatchDimRhsSpace(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() > 0 && device_mesh.dim(mesh_dim0) > 1 &&
        device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SbSj = SbR x SbSj @ {%d,%d}", mesh_dim0, mesh_dim1);
      HloSharding output_spec = Tile(ins->shape(), {0, out_rhs_space_dim},
                                     {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_batch_dims[0]}, {mesh_dim0}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_batch_dims[0], rhs_space_dims[0]},
               {mesh_dim0, mesh_dim1}, device_mesh);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                        cluster_env, strategy_map, strategies);
    }
  }

  void SplitBatchDimBothContract(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() > 0 && device_mesh.dim(mesh_dim0) > 1 &&
        device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SbR = SbSk x SbSk @ {%d,%d} (allreduce @ %d}",
                          mesh_dim0, mesh_dim1, mesh_dim1);
      HloSharding output_spec =
          Tile(ins->shape(), {0}, {mesh_dim0}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_batch_dims[0], lhs_con_dims[0]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_batch_dims[0], rhs_con_dims[0]},
               {mesh_dim0, mesh_dim1}, device_mesh);
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim1);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env, strategy_map,
                        strategies);
    }
  }

  void RecomputeSplitBothContract(int mesh_dim0, int mesh_dim1) {
    if (device_mesh.dim(mesh_dim0) > 1 && device_mesh.dim(mesh_dim1) > 1) {
      std::string name = absl::StrFormat("RR = RS x SR @ {%d} (allreduce @ %d)",
                                         mesh_dim0, mesh_dim0);
      HloSharding output_spec = HloSharding::Replicate();
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_con_dims[0]}, {mesh_dim0}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_con_dims[0]}, {mesh_dim0}, device_mesh);
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double compute_cost =
          cluster_env.DotCost(lhs->shape(), rhs->shape(), dot_dnums);
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim0);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec},
                        compute_cost, communication_cost, cluster_env,
                        strategy_map, strategies);
    }
  }

  void Add1DDataParallel() {
    if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
      int mesh_dim = 0;
      int64_t num_devices = device_mesh_1d.dim(mesh_dim);

      // Si = Si x R @ 0
      if (lhs->shape().dimensions(lhs_space_dims[0]) % num_devices == 0) {
        std::string name = absl::StrFormat("Si = Si x R @ %d", mesh_dim);
        HloSharding output_spec =
            Tile(ins->shape(), {out_lhs_space_dim}, {mesh_dim}, device_mesh_1d);
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_space_dims[0]}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec = HloSharding::Replicate();
        AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env, strategy_map, strategies);
      }

      // R = Sk x Sk @ (allreduce @ 0)
      if (lhs->shape().dimensions(lhs_con_dims[0]) % num_devices == 0 &&
          rhs->shape().dimensions(rhs_con_dims[0]) % num_devices == 0) {
        std::string name = absl::StrFormat("R = Sk x Sk @ %d (allreduce @ %d)",
                                           mesh_dim, mesh_dim);
        HloSharding output_spec = HloSharding::Replicate();
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_con_dims[0]}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_con_dims[0]}, {mesh_dim}, device_mesh_1d);
        double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        double communication_cost = cluster_env.AllReduceCost(memory_cost, 0) +
                                    cluster_env.AllReduceCost(memory_cost, 1);

        AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                          communication_cost, cluster_env, strategy_map,
                          strategies);
      }
    }
  }

  void Add1DBatchSplit() {
    if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
      int mesh_dim = 0;
      for (int64_t i = 0; i < lhs_batch_dims.size(); ++i) {
        std::string name =
            absl::StrFormat("Sb_%d = Sb x Sb @ {%d} 1d", i, mesh_dim);
        HloSharding output_spec =
            Tile(ins->shape(), {i}, {mesh_dim}, device_mesh_1d);
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dims[i]}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec =
            Tile(rhs->shape(), {rhs_batch_dims[i]}, {mesh_dim}, device_mesh_1d);
        AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env, strategy_map, strategies);
      }
    }
  }

  Status RegisterStrategies() {
    // SS = SR x RS
    // Split lhs space dim and rhs space dim.
    SplitLhsSpaceRhsSpace(0, 1);
    SplitLhsSpaceRhsSpace(1, 0);

    // SR = SS x SR
    // Split lhs space dim and both contracting dims.
    SplitLhsSpaceBothContract(0, 1);
    SplitLhsSpaceBothContract(1, 0);

    // RS = RS x SS
    // Split rhs space dim and both contracting dims.
    SplitRhsSpaceBothContract(0, 1);
    SplitRhsSpaceBothContract(1, 0);

    // RR = RS x SR
    // This is a special case where we allow spliting only one dim in the
    // 2d-mesh case. This allows some recomputation (e.g., the dense layer in
    // the LM_head of BERT).
    RecomputeSplitBothContract(0, 1);
    RecomputeSplitBothContract(1, 0);

    // Add 1d data parallel in 2d mesh
    if (solver_option.allow_mixed_mesh_shape) {
      Add1DDataParallel();
    }

    if (solver_option.batch_matmul_always_split_batch &&
        lhs_batch_dims.size() > 0 &&
        cluster_env.non_zero_mesh_dims.size() > 1) {
      // If there is a batch dim and the device mesh is 2d, always split on
      // batch dim. Clear all old strategies.
      strategies->leaf_vector.clear();
    }

    // Sb = Sb x Sb
    // Split one batch dim. Only used for 1d mesh
    SplitOneBatchDim();

    // SbSi = SbSi x SbR
    // Split batch dim and lhs space dim
    SplitBatchDimLhsSpace(0, 1);
    SplitBatchDimLhsSpace(1, 0);

    // SbSj = SbR x SbSj
    // Split batch dim and lhs space dim
    SplitBatchDimRhsSpace(0, 1);
    SplitBatchDimRhsSpace(1, 0);

    // SbSj = SbR x SbSj
    // Split batch dim and lhs space dim
    SplitBatchDimBothContract(0, 1);
    SplitBatchDimBothContract(1, 0);

    if (solver_option.batch_matmul_always_split_batch &&
        lhs_batch_dims.size() == 2 && device_mesh.dim(0) > 1 &&
        device_mesh.dim(1) > 1) {
      // If there are two batch dims, always split on these two dims.
      // Clear all old strategies.
      strategies->leaf_vector.clear();
    }

    // Sb = Sb x Sb
    // Split batch dims.
    SplitTwoBatchDims(0, 1);
    SplitTwoBatchDims(1, 0);

    if (solver_option.allow_mixed_mesh_shape) {
      Add1DBatchSplit();
    }

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map.count(ins)) {
      TF_RETURN_IF_ERROR(FilterStrategy(ins, strategies, cluster_env, batch_map,
                                        solver_option));
    }

    return Status::OK();
  }

  std::unique_ptr<StrategyVector>& strategies;
  StrategyMap& strategy_map;
  const HloInstruction* ins;
  const ClusterEnvironment& cluster_env;
  const InstructionBatchDimMap& batch_map;
  const AutoShardingSolverOption& solver_option;

  const Array<int64_t>& device_mesh;
  const Array<int64_t>& device_mesh_1d;
  const HloInstruction* lhs;
  const HloInstruction* rhs;

  // Dimension information
  const DotDimensionNumbers& dot_dnums;
  const tensorflow::protobuf::RepeatedField<int64_t>& lhs_con_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& rhs_con_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& lhs_batch_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& rhs_batch_dims;
  std::vector<int64_t> lhs_space_dims, rhs_space_dims;
  int64_t out_lhs_space_dim, out_rhs_space_dim;
};

// Register strategies for dot instructions.
Status HandleDot(std::unique_ptr<StrategyVector>& strategies,
                 LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                 const HloInstruction* ins, size_t instruction_id,
                 const ClusterEnvironment& cluster_env,
                 const InstructionBatchDimMap& batch_map,
                 const AutoShardingSolverOption& solver_option) {
  strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                        leaf_strategies);

  DotHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                     solver_option);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return Status::OK();
}

class ConvHandler {
 public:
  ConvHandler(std::unique_ptr<StrategyVector>& strategies,
              StrategyMap& strategy_map, const HloInstruction* ins,
              const ClusterEnvironment& cluster_env,
              const InstructionBatchDimMap& batch_map,
              const AutoShardingSolverOption& solver_option)
      : strategies(strategies),
        strategy_map(strategy_map),
        ins(ins),
        cluster_env(cluster_env),
        batch_map(batch_map),
        solver_option(solver_option),
        device_mesh(cluster_env.device_mesh),
        device_mesh_1d(cluster_env.device_mesh_1d),
        lhs(ins->operand(0)),
        rhs(ins->operand(1)),
        conv_dnums(ins->convolution_dimension_numbers()) {
    lhs_batch_dim = conv_dnums.input_batch_dimension();
    lhs_in_channel_dim = conv_dnums.input_feature_dimension();
    rhs_in_channel_dim = conv_dnums.kernel_input_feature_dimension();
    rhs_out_channel_dim = conv_dnums.kernel_output_feature_dimension();
    out_batch_dim = conv_dnums.output_batch_dimension();
    out_out_channel_dim = conv_dnums.output_feature_dimension();

    // Only support 2 dimensional device mesh
    CHECK_EQ(device_mesh.num_dimensions(), 2);
  }

  void SplitLhsBatchRhsOutchannel(int mesh_dim0, int mesh_dim1) {
    std::string name =
        absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1);
    HloSharding output_spec =
        Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
             {mesh_dim0, mesh_dim1}, device_mesh);
    HloSharding lhs_spec =
        Tile(lhs->shape(), {lhs_batch_dim}, {mesh_dim0}, device_mesh);
    HloSharding rhs_spec =
        Tile(rhs->shape(), {rhs_out_channel_dim}, {mesh_dim1}, device_mesh);

    AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                      cluster_env, strategy_map, strategies);
  }

  void SplitLhsBatchBothInchannel(int mesh_dim0, int mesh_dim1) {
    if (device_mesh.dim(mesh_dim0) > 1 && device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim1);
      HloSharding output_spec =
          Tile(ins->shape(), {out_batch_dim}, {mesh_dim0}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
               {mesh_dim0, mesh_dim1}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_in_channel_dim}, {mesh_dim1}, device_mesh);

      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim1);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env, strategy_map,
                        strategies);
    }
  }

  void SplitRhsOutchannelBothInchannel(int mesh_dim0, int mesh_dim1) {
    if (device_mesh.dim(mesh_dim0) > 1) {
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim0);
      HloSharding output_spec =
          Tile(ins->shape(), {out_out_channel_dim}, {mesh_dim1}, device_mesh);
      HloSharding lhs_spec =
          Tile(lhs->shape(), {lhs_in_channel_dim}, {mesh_dim0}, device_mesh);
      HloSharding rhs_spec =
          Tile(rhs->shape(), {rhs_in_channel_dim, rhs_out_channel_dim},
               {mesh_dim0, mesh_dim1}, device_mesh);

      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim0);

      AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env, strategy_map,
                        strategies);
    }
  }

  void Add1DDataParallel() {
    if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
      int mesh_dim = 0;
      int64_t num_devices = device_mesh_1d.dim(mesh_dim);

      // Si = Si x R @ 0
      if (lhs->shape().dimensions(lhs_batch_dim) % num_devices == 0) {
        std::string name = absl::StrFormat("Si = Si x R @ 0");
        HloSharding output_spec =
            Tile(ins->shape(), {out_batch_dim}, {mesh_dim}, device_mesh_1d);
        HloSharding lhs_spec =
            Tile(lhs->shape(), {lhs_batch_dim}, {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec = HloSharding::Replicate();

        AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env, strategy_map, strategies);
      }

      // R = Sk x Sk @ (allreduce @ 0)
      if (lhs->shape().dimensions(lhs_in_channel_dim) % num_devices == 0 &&
          rhs->shape().dimensions(rhs_in_channel_dim) % num_devices == 0) {
        std::string name = absl::StrFormat("R = Sk x Sk @ %d (allreduce @ %d)",
                                           mesh_dim, mesh_dim);
        HloSharding output_spec = HloSharding::Replicate();
        HloSharding lhs_spec = Tile(lhs->shape(), {lhs_in_channel_dim},
                                    {mesh_dim}, device_mesh_1d);
        HloSharding rhs_spec = Tile(rhs->shape(), {rhs_in_channel_dim},
                                    {mesh_dim}, device_mesh_1d);
        double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        double communication_cost = cluster_env.AllReduceCost(memory_cost, 0) +
                                    cluster_env.AllReduceCost(memory_cost, 1);

        AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0,
                          communication_cost, cluster_env, strategy_map,
                          strategies);
      }
    }
  }

  void SplitDepthwise(int mesh_dim0, int mesh_dim1, bool forward) {
    std::string name =
        absl::StrFormat("SS = SS x RS @ {%d,%d}", mesh_dim0, mesh_dim1);
    HloSharding output_spec =
        Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
             {mesh_dim0, mesh_dim1}, device_mesh);
    HloSharding lhs_spec =
        forward ? Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                       {mesh_dim0, mesh_dim1}, device_mesh)
                : Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                       {mesh_dim1, mesh_dim0}, device_mesh);

    HloSharding rhs_spec =
        Tile(rhs->shape(), {rhs_out_channel_dim}, {mesh_dim1}, device_mesh);

    AppendNewStrategy(ins, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                      cluster_env, strategy_map, strategies);
  }

  Status RegisterStrategies() {
    if ((ins->feature_group_count() ==
             lhs->shape().dimensions(lhs_in_channel_dim) &&
         ins->feature_group_count() ==
             rhs->shape().dimensions(rhs_out_channel_dim))) {
      // for depthwise conv
      // SS = SS x S
      // Split batch dim and channel dim
      SplitDepthwise(0, 1, true);
      SplitDepthwise(1, 0, true);
    } else if ((ins->batch_group_count() ==
                    lhs->shape().dimensions(lhs_batch_dim) &&
                ins->batch_group_count() ==
                    rhs->shape().dimensions(rhs_out_channel_dim))) {
      // for depthwise conv filter_backward
      // SS = SS x S
      // Split batch dim and channel dim
      SplitDepthwise(0, 1, false);
      SplitDepthwise(1, 0, false);
    }

    // SS = SR x RS
    // Split lhs batch dim and rhs out_channel dim.
    SplitLhsBatchRhsOutchannel(0, 1);
    SplitLhsBatchRhsOutchannel(1, 0);

    // SR = SS x SR
    // Split lhs batch dim and both in_channel dims.
    SplitLhsBatchBothInchannel(0, 1);
    SplitLhsBatchBothInchannel(1, 0);

    // RS = RS x SS
    // Split rhs out_channel dim and both in_channel dims.
    SplitRhsOutchannelBothInchannel(0, 1);
    SplitRhsOutchannelBothInchannel(1, 0);

    // Add 1d data parallel in 2d mesh
    if (solver_option.allow_mixed_mesh_shape) {
      Add1DDataParallel();
    }

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map.count(ins)) {
      TF_RETURN_IF_ERROR(FilterStrategy(ins, strategies, cluster_env, batch_map,
                                        solver_option));
    }

    return Status::OK();
  }

  std::unique_ptr<StrategyVector>& strategies;
  StrategyMap& strategy_map;
  const HloInstruction* ins;
  const ClusterEnvironment& cluster_env;
  const InstructionBatchDimMap& batch_map;
  const AutoShardingSolverOption& solver_option;

  const Array<int64_t>& device_mesh;
  const Array<int64_t>& device_mesh_1d;
  const HloInstruction* lhs;
  const HloInstruction* rhs;

  // Dimension information
  const ConvolutionDimensionNumbers& conv_dnums;
  int64_t lhs_batch_dim, lhs_in_channel_dim;
  int64_t rhs_in_channel_dim, rhs_out_channel_dim;
  int64_t out_batch_dim, out_out_channel_dim;
};

// Register strategies for dot instructions.
Status HandleConv(std::unique_ptr<StrategyVector>& strategies,
                  LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                  const HloInstruction* ins, size_t instruction_id,
                  const ClusterEnvironment& cluster_env,
                  const InstructionBatchDimMap& batch_map,
                  const AutoShardingSolverOption& solver_option) {
  strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                        leaf_strategies);

  ConvHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                      solver_option);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return Status::OK();
}

}  // namespace spmd
}  // namespace xla
