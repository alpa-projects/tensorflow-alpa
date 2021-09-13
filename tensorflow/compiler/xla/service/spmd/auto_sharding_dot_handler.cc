#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

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
             {mesh_dim0, mesh_dim1}, cluster_env);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1),
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(strategy_map.at(lhs).get(), lhs->shape(),
                                  Tile(lhs->shape(), {lhs_space_dims[0]},
                                       {mesh_dim0}, cluster_env),
                                  cluster_env),
             ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                  Tile(rhs->shape(), {rhs_space_dims[0]},
                                       {mesh_dim1}, cluster_env),
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
                           {mesh_dim0, mesh_dim1}, cluster_env);
        memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        communication_cost = cluster_env.ReduceScatterCost(
            memory_cost * device_mesh.dim(mesh_dim1), mesh_dim1);
      } else {
        name = absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)",
                               mesh_dim0, mesh_dim1, mesh_dim1);
        output_spec =
            Tile(ins->shape(), {space_base_dim}, {mesh_dim0}, cluster_env);
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
                        {mesh_dim0, mesh_dim1}, cluster_env),
                   cluster_env),
               ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                    Tile(rhs->shape(), {rhs_con_dims[0]},
                                         {mesh_dim1}, cluster_env),
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
                               mesh_dim0, mesh_dim1, mesh_dim0),
        output_spec = Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                           {mesh_dim0, mesh_dim1}, cluster_env);
        memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
        communication_cost = cluster_env.ReduceScatterCost(
            memory_cost * device_mesh.dim(mesh_dim0), mesh_dim0);
      } else {
        name = absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)",
                               mesh_dim0, mesh_dim1, mesh_dim0),
        output_spec =
            Tile(ins->shape(), {space_base_dim + 1}, {mesh_dim1}, cluster_env);
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
                                         {mesh_dim0}, cluster_env),
                                    cluster_env),
               ReshardingCostVector(
                   strategy_map.at(rhs).get(), rhs->shape(),
                   Tile(rhs->shape(), {rhs_con_dims[0], rhs_space_dims[0]},
                        {mesh_dim0, mesh_dim1}, cluster_env),
                   cluster_env),
           }}));
    }
  }

  void SplitBatchDims() {
    if (lhs_batch_dims.size() > 0 &&
        solver_option.batch_matmul_always_split_batch) {
      strategies->leaf_vector.clear();
    }

    // Split one batch dim
    for (int64_t i = 0; i < lhs_batch_dims.size(); ++i) {
      for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
        if (device_mesh.dim(j) == 1 ||
            ins->shape().dimensions(i) < device_mesh.dim(j)) {
          continue;
        }

        HloSharding output_spec = Tile(ins->shape(), {i}, {j}, cluster_env);
        strategies->leaf_vector.push_back(ShardingStrategy(
            {absl::StrFormat("Sb_%d = Sb x Sb @ {%d}", i, j),
             output_spec,
             0,
             0,
             GetBytes(ins->shape()) / output_spec.NumTiles(),
             {
                 ReshardingCostVector(
                     strategy_map.at(lhs).get(), lhs->shape(),
                     Tile(lhs->shape(), {lhs_batch_dims[i]}, {j}, cluster_env),
                     cluster_env),
                 ReshardingCostVector(
                     strategy_map.at(rhs).get(), rhs->shape(),
                     Tile(rhs->shape(), {rhs_batch_dims[i]}, {j}, cluster_env),
                     cluster_env),
             }}));
      }
    }

    // Split two batch dims
    if (lhs_batch_dims.size() == 2 && device_mesh.dim(0) > 1 &&
        device_mesh.dim(1) > 1) {
      strategies->leaf_vector.clear();

      HloSharding output_spec = Tile(ins->shape(), {0, 1}, {0, 1}, cluster_env);
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
                        {0, 1}, cluster_env),
                   cluster_env),
               ReshardingCostVector(
                   strategy_map.at(rhs).get(), rhs->shape(),
                   Tile(rhs->shape(), {rhs_batch_dims[0], rhs_batch_dims[1]},
                        {0, 1}, cluster_env),
                   cluster_env),
           }}));
    }

    // TODO(lmzheng): Register the mirror strategies for device mapping {1,0}.
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
                                         {mesh_dim0}, cluster_env),
                                    cluster_env),
               ReshardingCostVector(strategy_map.at(rhs).get(), rhs->shape(),
                                    Tile(rhs->shape(), {rhs_con_dims[0]},
                                         {mesh_dim0}, cluster_env),
                                    cluster_env),
           }}));
    }
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

    // Sb = Sb x Sb
    // Split batch dims.
    SplitBatchDims();

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies.
    if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map.count(ins)) {
      int mesh_dim = solver_option.force_batch_dim_to_mesh_dim;
      int batch_dim = batch_map.at(ins);
      CHECK_GE(ins->shape().dimensions(batch_dim), device_mesh.dim(mesh_dim));

      std::vector<ShardingStrategy> new_leaf_vector;
      for (auto& stra : strategies->leaf_vector) {
        std::vector<int> tensor_dim_to_mesh_dim =
            cluster_env.GetTensorDimToMeshDim(ins->shape(),
                                              stra.output_sharding);
        if (tensor_dim_to_mesh_dim[batch_dim] == mesh_dim) {
          new_leaf_vector.push_back(std::move(stra));
        }
      }
      CHECK(!new_leaf_vector.empty());
      strategies->leaf_vector = std::move(new_leaf_vector);
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

}  // namespace spmd
}  // namespace xla
