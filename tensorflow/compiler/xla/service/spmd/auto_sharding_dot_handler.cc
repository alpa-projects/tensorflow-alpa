#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

// Filter strategies according to the solver_option.force_batch_dim_to_mesh_dim.
// This can be used to forcibly generate data-parallel strategies.
void FilterStrategy(const HloInstruction* ins,
                    std::unique_ptr<StrategyVector>& strategies,
                    const ClusterEnvironment& cluster_env,
                    const InstructionBatchDimMap& batch_map,
                    const AutoShardingSolverOption& solver_option) {
  int mesh_dim = solver_option.force_batch_dim_to_mesh_dim;
  int batch_dim = batch_map.at(ins);
  CHECK_GE(ins->shape().dimensions(batch_dim),
           cluster_env.device_mesh.dim(mesh_dim));

  std::vector<ShardingStrategy> new_leaf_vector;
  for (auto& stra : strategies->leaf_vector) {
    std::vector<int> tensor_dim_to_mesh_dim =
        cluster_env.GetTensorDimToMeshDim(ins->shape(), stra.output_sharding);
    if (tensor_dim_to_mesh_dim[batch_dim] == mesh_dim) {
      new_leaf_vector.push_back(std::move(stra));
    }
  }
  CHECK(!new_leaf_vector.empty());
  strategies->leaf_vector = std::move(new_leaf_vector);
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
        lhs(ins->operand(0)),
        rhs(ins->operand(1)),
        dot_dnums(ins->dot_dimension_numbers()),
        space_base_dim(dot_dnums.lhs_batch_dimensions_size()),
        lhs_con_dims(ins->dot_dimension_numbers().lhs_contracting_dimensions()),
        rhs_con_dims(ins->dot_dimension_numbers().rhs_contracting_dimensions()),
        lhs_batch_dims(ins->dot_dimension_numbers().lhs_batch_dimensions()),
        rhs_batch_dims(ins->dot_dimension_numbers().rhs_batch_dimensions()) {
    std::tie(lhs_space_dims, rhs_space_dims) =
        GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);

    CHECK_EQ(lhs_space_dims.size(), 1);
    CHECK_EQ(rhs_space_dims.size(), 1);
    CHECK_EQ(lhs_con_dims.size(), 1);
    CHECK_EQ(rhs_con_dims.size(), 1);

    // Only support 2 dimensional device mesh
    CHECK_EQ(device_mesh.num_dimensions(), 2);
  }

  void SplitLhsSpaceRhsSpace(int mesh_dim0, int mesh_dim1) {
    // if (ins->shape().dimensions(space_base_dim) < device_mesh.dim(mesh_dim0)
    // ||
    //    ins->shape().dimensions(space_base_dim + 1) <
    //    device_mesh.dim(mesh_dim1)) {
    //  return;  // The dimension length is to small to be parallelzied.
    //}

    HloSharding output_spec =
        Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
             {mesh_dim0, mesh_dim1}, device_mesh);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                  Tile(lhs->shape(), {lhs_space_dims[0]},
                                       {mesh_dim0}, device_mesh),
                                  cluster_env),
             ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                  Tile(rhs->shape(), {rhs_space_dims[0]},
                                       {mesh_dim1}, device_mesh),
                                  cluster_env),
         }}));
  }

  void SplitLhsSpaceBothContract(int mesh_dim0, int mesh_dim1) {
    // if (lhs->shape().dimensions(lhs_space_dims[0]) <
    // device_mesh.dim(mesh_dim0) ||
    //    lhs->shape().dimensions(lhs_con_dims[0]) < device_mesh.dim(mesh_dim1))
    //    {
    //  return;  // The dimension length is to small to be parallelzied.
    //}

    if (device_mesh.dim(mesh_dim0) > 1 && device_mesh.dim(mesh_dim1) > 1) {
      HloSharding output_spec = Undefined();
      std::string name;
      double communication_cost;
      double memory_cost;

      if (false && solver_option.prefer_reduce_scatter) {  // Deprecated branch
        name = absl::StrFormat("SS = SS x SR @ {%d,%d} (reduce-scatter @ %d)",
                               mesh_dim0, mesh_dim1, mesh_dim1);
        output_spec = Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                           {mesh_dim0, mesh_dim1}, device_mesh);
        memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        communication_cost = cluster_env.ReduceScatterCost(
            memory_cost * device_mesh.dim(mesh_dim1), mesh_dim1);
      } else {
        name = absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)",
                               mesh_dim0, mesh_dim1, mesh_dim1);
        output_spec =
            Tile(ins->shape(), {space_base_dim}, {mesh_dim0}, device_mesh);
        memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        communication_cost = cluster_env.AllReduceCost(memory_cost, mesh_dim1);
      }

      strategies->leaf_vector.push_back(ShardingStrategy(
          {name,
           output_spec,
           0,
           communication_cost,
           memory_cost,
           {
               ReshardingCostVector(
                   strategy_map.at(lhs).get(), lhs->shape(),
                   Tile(lhs->shape(), {lhs_space_dims[0], lhs_con_dims[0]},
                        {mesh_dim0, mesh_dim1}, device_mesh),
                   cluster_env),
               ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                    Tile(rhs->shape(), {rhs_con_dims[0]},
                                         {mesh_dim1}, device_mesh),
                                    cluster_env),
           }}));
    }
  }

  void SplitRhsSpaceBothContract(int mesh_dim0, int mesh_dim1) {
    // if (rhs->shape().dimensions(rhs_con_dims[0]) < device_mesh.dim(mesh_dim0)
    // ||
    //    rhs->shape().dimensions(rhs_space_dims[0]) <
    //    device_mesh.dim(mesh_dim1)) {
    //  return;  // The dimension length is to small to be parallelzied.
    //}

    if (device_mesh.dim(mesh_dim0) > 1) {
      HloSharding output_spec = Undefined();
      std::string name;
      double communication_cost;
      double memory_cost;

      if (false && solver_option.prefer_reduce_scatter) {  // Deprecated branch
        name = absl::StrFormat("SS = RS x SS @ {%d,%d} (reduce-scatter @ %d)",
                               mesh_dim0, mesh_dim1, mesh_dim0);
        output_spec = Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                           {mesh_dim0, mesh_dim1}, device_mesh);
        memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        communication_cost = cluster_env.ReduceScatterCost(
            memory_cost * device_mesh.dim(mesh_dim0), mesh_dim0);
      } else {
        name = absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)",
                               mesh_dim0, mesh_dim1, mesh_dim0),
        output_spec =
            Tile(ins->shape(), {space_base_dim + 1}, {mesh_dim1}, device_mesh);
        memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        communication_cost = cluster_env.AllReduceCost(memory_cost, mesh_dim0);
      }

      strategies->leaf_vector.push_back(ShardingStrategy(
          {name,
           output_spec,
           0,
           communication_cost,
           memory_cost,
           {
               ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                    Tile(lhs->shape(), {lhs_con_dims[0]},
                                         {mesh_dim0}, device_mesh),
                                    cluster_env),
               ReshardingCostVector(
                   strategy_map.at(rhs).get(), rhs->shape(),
                   Tile(rhs->shape(), {rhs_con_dims[0], rhs_space_dims[0]},
                        {mesh_dim0, mesh_dim1}, device_mesh),
                   cluster_env),
           }}));
    }
  }

  void SplitBatchDims() {
    // Split one batch dim
    for (int64_t i = 0; i < lhs_batch_dims.size(); ++i) {
      for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
        if (device_mesh.dim(j) == 1 ||
            ins->shape().dimensions(i) < device_mesh.dim(j)) {
          continue;
        }

        HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);
        strategies->leaf_vector.push_back(ShardingStrategy(
            {absl::StrFormat("Sb_%d = Sb x Sb @ {%d}", i, j),
             output_spec,
             0,
             0,
             GetBytes(ins->shape()) / output_spec.NumTiles(),
             {
                 ReshardingCostVector(
                     strategy_map.at(lhs).get(), lhs->shape(),
                     Tile(lhs->shape(), {lhs_batch_dims[i]}, {j}, device_mesh),
                     cluster_env),
                 ReshardingCostVector(
                     strategy_map.at(rhs).get(), rhs->shape(),
                     Tile(rhs->shape(), {rhs_batch_dims[i]}, {j}, device_mesh),
                     cluster_env),
             }}));
      }
    }

    // Split two batch dims
    // TODO(lmzheng): Register the mirror strategies for device mapping {1,0}.
    if (lhs_batch_dims.size() == 2 && device_mesh.dim(0) > 1 &&
        device_mesh.dim(1) > 1) {
      strategies->leaf_vector.clear();

      HloSharding output_spec = Tile(ins->shape(), {0, 1}, {0, 1}, device_mesh);
      strategies->leaf_vector.push_back(ShardingStrategy(
          {"Sb = Sb x Sb @ {0,1}",
           output_spec,
           0,
           0,
           GetBytes(ins->shape()) / output_spec.NumTiles(),
           {
               ReshardingCostVector(
                   strategy_map.at(lhs).get(), lhs->shape(),
                   Tile(lhs->shape(), {lhs_batch_dims[0], lhs_batch_dims[1]},
                        {0, 1}, device_mesh),
                   cluster_env),
               ReshardingCostVector(
                   strategy_map.at(rhs).get(), rhs->shape(),
                   Tile(rhs->shape(), {rhs_batch_dims[0], rhs_batch_dims[1]},
                        {0, 1}, device_mesh),
                   cluster_env),
           }}));
    }
  }

  void SplitBatchDimLhsSpace(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() < 1) {
      return;
    }

    HloSharding output_spec = Tile(ins->shape(), {0, space_base_dim},
                                   {mesh_dim0, mesh_dim1}, device_mesh);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SbSi = SbSi x SbR @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_batch_dims[0], lhs_space_dims[0]},
                      {mesh_dim0, mesh_dim1}, device_mesh),
                 cluster_env),
             ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                  Tile(rhs->shape(), {rhs_batch_dims[0]},
                                       {mesh_dim0}, device_mesh),
                                  cluster_env),
         }}));
  }

  void SplitBatchDimRhsSpace(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() < 1) {
      return;
    }

    HloSharding output_spec = Tile(ins->shape(), {0, space_base_dim + 1},
                                   {mesh_dim0, mesh_dim1}, device_mesh);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SbSj = SbR x SbSj @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                  Tile(rhs->shape(), {lhs_batch_dims[0]},
                                       {mesh_dim0}, device_mesh),
                                  cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(lhs->shape(), {rhs_batch_dims[0], rhs_space_dims[0]},
                      {mesh_dim0, mesh_dim1}, device_mesh),
                 cluster_env),
         }}));
  }

  void SplitBatchDimBothContract(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims.size() < 1) {
      return;
    }

    HloSharding output_spec = Tile(ins->shape(), {0}, {mesh_dim0}, device_mesh);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SbR = SbSk x SbSk @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(rhs->shape(), {lhs_batch_dims[0], lhs_con_dims[0]},
                      {mesh_dim0, mesh_dim1}, device_mesh),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(lhs->shape(), {rhs_batch_dims[0], rhs_con_dims[0]},
                      {mesh_dim0, mesh_dim1}, device_mesh),
                 cluster_env),
         }}));
  }

  void RecomputeSplitBothContract(int mesh_dim0, int mesh_dim1) {
    if (device_mesh.dim(mesh_dim0) > 1 && device_mesh.dim(mesh_dim1) > 1) {
      HloSharding output_spec = HloSharding::Replicate();
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

      strategies->leaf_vector.push_back(ShardingStrategy(
          {absl::StrFormat("RR = RS x SR @ {%d} (allreduce @ %d)", mesh_dim0,
                           mesh_dim0),
           output_spec,
           cluster_env.DotCost(lhs->shape(), rhs->shape(), dot_dnums),
           cluster_env.AllReduceCost(memory_cost, mesh_dim0),
           memory_cost,
           {
               ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                    Tile(lhs->shape(), {lhs_con_dims[0]},
                                         {mesh_dim0}, device_mesh),
                                    cluster_env),
               ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                    Tile(rhs->shape(), {rhs_con_dims[0]},
                                         {mesh_dim0}, device_mesh),
                                    cluster_env),
           }}));
    }
  }

  void Add1DDataParallel() {
    if (!(device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1)) {
      return;
    }
    Array<int64_t> device_mesh = cluster_env.device_mesh;
    device_mesh.Reshape({device_mesh.num_elements(), 1});

    int mesh_dim = 0;

    // Si = Si x R @ 0
    HloSharding output_spec =
        Tile(ins->shape(), {space_base_dim}, {mesh_dim}, device_mesh);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("Si = Si x R @ %d", mesh_dim),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                  Tile(lhs->shape(), {lhs_space_dims[0]},
                                       {mesh_dim}, device_mesh),
                                  cluster_env),
             ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                  HloSharding::Replicate(), cluster_env),
         }}));

    // R = Sk x Sk @ (allreduce @ 0)
    output_spec = HloSharding::Replicate();
    double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
    double communication_cost = cluster_env.AllReduceCost(memory_cost, 0) +
                                cluster_env.AllReduceCost(memory_cost, 1);

    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("R = Sk x Sk @ %d (allreduce @ %d)", mesh_dim,
                         mesh_dim),
         output_spec,
         0,
         communication_cost,
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_con_dims[0]}, {0}, device_mesh),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0]}, {0}, device_mesh),
                 cluster_env),
         }}));
  }

  void RegisterStrategies() {
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

    if (lhs_batch_dims.size() > 0 &&
        solver_option.batch_matmul_always_split_batch) {
      // Always split on batch dim if there is a batch dim
      strategies->leaf_vector.clear();
    }

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

    // Sb = Sb x Sb
    // Split batch dims.
    SplitBatchDims();

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map.count(ins)) {
      FilterStrategy(ins, strategies, cluster_env, batch_map, solver_option);
    }
  }

  std::unique_ptr<StrategyVector>& strategies;
  StrategyMap& strategy_map;
  const HloInstruction* ins;
  const ClusterEnvironment& cluster_env;
  const InstructionBatchDimMap& batch_map;
  const AutoShardingSolverOption& solver_option;

  const Array<int64_t>& device_mesh;
  const HloInstruction* lhs;
  const HloInstruction* rhs;

  // Dimension information
  const DotDimensionNumbers& dot_dnums;
  int64_t space_base_dim;
  std::vector<int64_t> lhs_space_dims, rhs_space_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& lhs_con_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& rhs_con_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& lhs_batch_dims;
  const tensorflow::protobuf::RepeatedField<int64_t>& rhs_batch_dims;
};

// Register strategies for dot instructions.
void HandleDot(std::unique_ptr<StrategyVector>& strategies,
               LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
               const HloInstruction* ins, size_t instruction_id,
               const ClusterEnvironment& cluster_env,
               const InstructionBatchDimMap& batch_map,
               const AutoShardingSolverOption& solver_option) {
  strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
  SetInNodesWithInstruction(strategies, ins, strategy_map);

  DotHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                     solver_option);
  handler.RegisterStrategies();
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
    HloSharding output_spec =
        Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
             {mesh_dim0, mesh_dim1}, device_mesh);

    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_batch_dim}, {mesh_dim0}, device_mesh),
                 cluster_env),
             ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                  Tile(rhs->shape(), {rhs_out_channel_dim},
                                       {mesh_dim1}, device_mesh),
                                  cluster_env),
         }}));
  }

  void SplitLhsBatchBothInchannel(int mesh_dim0, int mesh_dim1) {
    if (device_mesh.dim(mesh_dim0) > 1 && device_mesh.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim1);
      HloSharding output_spec =
          Tile(ins->shape(), {out_batch_dim}, {mesh_dim0}, device_mesh);
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim1);

      strategies->leaf_vector.push_back(ShardingStrategy(
          {name,
           output_spec,
           0,
           communication_cost,
           memory_cost,
           {
               ReshardingCostVector(
                   strategy_map.at(lhs).get(), lhs->shape(),
                   Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                        {mesh_dim0, mesh_dim1}, device_mesh),
                   cluster_env),
               ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                    Tile(rhs->shape(), {rhs_in_channel_dim},
                                         {mesh_dim1}, device_mesh),
                                    cluster_env),
           }}));
    }
  }

  void SplitRhsOutchannelBothInchannel(int mesh_dim0, int mesh_dim1) {
    if (device_mesh.dim(mesh_dim0) > 1) {
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim0);
      HloSharding output_spec =
          Tile(ins->shape(), {out_out_channel_dim}, {mesh_dim1}, device_mesh);
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env.AllReduceCost(memory_cost, mesh_dim0);

      strategies->leaf_vector.push_back(ShardingStrategy(
          {name,
           output_spec,
           0,
           communication_cost,
           memory_cost,
           {
               ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                    Tile(lhs->shape(), {lhs_in_channel_dim},
                                         {mesh_dim0}, device_mesh),
                                    cluster_env),
               ReshardingCostVector(
                   strategy_map.at(rhs).get(), rhs->shape(),
                   Tile(rhs->shape(), {rhs_in_channel_dim, rhs_out_channel_dim},
                        {mesh_dim0, mesh_dim1}, device_mesh),
                   cluster_env),
           }}));
    }
  }

  void SplitDepthwise(int mesh_dim0, int mesh_dim1, bool forward) {
    HloSharding output_spec =
        Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
             {mesh_dim0, mesh_dim1}, device_mesh);

    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SS = SS x RS @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 forward
                     ? Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                            {mesh_dim0, mesh_dim1}, device_mesh)
                     : Tile(lhs->shape(), {lhs_batch_dim, lhs_in_channel_dim},
                            {mesh_dim1, mesh_dim0}, device_mesh),
                 cluster_env),
             ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                  Tile(rhs->shape(), {rhs_out_channel_dim},
                                       {mesh_dim1}, device_mesh),
                                  cluster_env),
         }}));
  }

  void RegisterStrategies() {
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

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map.count(ins)) {
      FilterStrategy(ins, strategies, cluster_env, batch_map, solver_option);
    }
  }

  std::unique_ptr<StrategyVector>& strategies;
  StrategyMap& strategy_map;
  const HloInstruction* ins;
  const ClusterEnvironment& cluster_env;
  const InstructionBatchDimMap& batch_map;
  const AutoShardingSolverOption& solver_option;

  const Array<int64_t>& device_mesh;
  const HloInstruction* lhs;
  const HloInstruction* rhs;

  // Dimension information
  const ConvolutionDimensionNumbers& conv_dnums;
  int64_t lhs_batch_dim, lhs_in_channel_dim;
  int64_t rhs_in_channel_dim, rhs_out_channel_dim;
  int64_t out_batch_dim, out_out_channel_dim;
};

// Register strategies for dot instructions.
void HandleConv(std::unique_ptr<StrategyVector>& strategies,
                LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                const HloInstruction* ins, size_t instruction_id,
                const ClusterEnvironment& cluster_env,
                const InstructionBatchDimMap& batch_map,
                const AutoShardingSolverOption& solver_option) {
  strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
  SetInNodesWithInstruction(strategies, ins, strategy_map);

  ConvHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                      solver_option);
  handler.RegisterStrategies();
}

}  // namespace spmd
}  // namespace xla
