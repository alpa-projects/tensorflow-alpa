#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

void HandleDot(std::unique_ptr<StrategyVector>& strategies,
               LeafStrategies& leaf_strategies,
               StrategyMap& strategy_map,
               const HloInstruction* ins,
               size_t instruction_id,
               const ClusterEnvironment& cluster_env) {
  const Array<int64>& device_mesh = cluster_env.device_mesh;

  strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
  SetInNodesWithInstruction(strategies, ins, strategy_map);

  // Parse dimensions
  const HloInstruction* lhs = ins->operand(0);
  const HloInstruction* rhs = ins->operand(1);
  const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
  int64 space_base_dim = dot_dnums.lhs_batch_dimensions_size();
  std::vector<int64> lhs_space_dims, rhs_space_dims;
  std::tie(lhs_space_dims, rhs_space_dims) =
      GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);
  const auto& lhs_con_dims = dot_dnums.lhs_contracting_dimensions();
  const auto& rhs_con_dims = dot_dnums.rhs_contracting_dimensions();
  const auto& lhs_batch_dims = dot_dnums.lhs_batch_dimensions();
  const auto& rhs_batch_dims = dot_dnums.rhs_batch_dimensions();

  CHECK_EQ(lhs_space_dims.size(), 1);
  CHECK_EQ(rhs_space_dims.size(), 1);
  CHECK_EQ(lhs_con_dims.size(), 1);
  CHECK_EQ(rhs_con_dims.size(), 1);

  // Only support 2 dimensional device mesh
  CHECK_EQ(device_mesh.num_dimensions(), 2);

  // Split lhs space dim + rhs space dim
  // @ {0, 1}
  HloSharding output_spec =
      Tile(ins->shape(), {space_base_dim, space_base_dim + 1}, {0, 1},
           cluster_env);
  strategies->leaf_vector.push_back(ShardingStrategy(
      {"SS = SR x RS @ {0, 1}",
       output_spec,
       0,
       0,
       GetBytes(ins->shape()) / output_spec.NumTiles(),
       {
           ReshardingCostVector(
               strategy_map.at(lhs).get(), lhs->shape(),
               Tile(lhs->shape(), {lhs_space_dims[0]}, {0}, cluster_env),
               cluster_env),
           ReshardingCostVector(
               strategy_map.at(rhs).get(), rhs->shape(),
               Tile(rhs->shape(), {rhs_space_dims[0]}, {1}, cluster_env),
               cluster_env),
       }}));

  // @ {1, 0}
  output_spec = Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                     {1, 0}, cluster_env);
  strategies->leaf_vector.push_back(ShardingStrategy(
      {"SS = SR x RS @ {1, 0}",
       output_spec,
       0,
       0,
       GetBytes(ins->shape()) / output_spec.NumTiles(),
       {
           ReshardingCostVector(
               strategy_map.at(lhs).get(), lhs->shape(),
               Tile(lhs->shape(), {lhs_space_dims[0]}, {1}, cluster_env),
               cluster_env),
           ReshardingCostVector(
               strategy_map.at(rhs).get(), rhs->shape(),
               Tile(rhs->shape(), {rhs_space_dims[0]}, {0}, cluster_env),
               cluster_env),
       }}));

  // Split lhs space dim + contracting dim
  // @ {0, 1}
  if (device_mesh.dim(1) > 1) {
    HloSharding output_spec =
        Tile(ins->shape(), {space_base_dim}, {0}, cluster_env);
    double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
    strategies->leaf_vector.push_back(ShardingStrategy(
        {"SR = SS x SR @ {0, 1} (allreduce @ 1)",
         output_spec,
         0,
         cluster_env.AllReduceCost(memory_cost, 1),
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_space_dims[0], lhs_con_dims[0]},
                      {0, 1}, cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0]}, {1}, cluster_env),
                 cluster_env),
         }}));
  }

  // @ {1, 0}
  if (device_mesh.dim(0) > 1) {
    HloSharding output_spec =
        Tile(ins->shape(), {space_base_dim}, {1}, cluster_env);
    double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
    strategies->leaf_vector.push_back(ShardingStrategy(
        {"SR = SS x SR @ {1, 0} (allreduce @ 0)",
         output_spec,
         0,
         cluster_env.AllReduceCost(memory_cost, 0),
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_space_dims[0], lhs_con_dims[0]},
                      {1, 0}, cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0]}, {0}, cluster_env),
                 cluster_env),
         }}));
  }

  // Split rhs space dim + contracting dim
  // @ {0, 1}
  if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
    HloSharding output_spec =
        Tile(ins->shape(), {space_base_dim + 1}, {1}, cluster_env);
    double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
    strategies->leaf_vector.push_back(ShardingStrategy(
        {"RS = RS x SS @ {0, 1} (allreduce @ 0)",
         output_spec,
         0,
         cluster_env.AllReduceCost(memory_cost, 0),
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_con_dims[0]}, {0}, cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0], rhs_space_dims[0]},
                      {0, 1}, cluster_env),
                 cluster_env),
         }}));
  }

  // @ {1, 0}
  if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
    HloSharding output_spec =
        Tile(ins->shape(), {space_base_dim + 1}, {0}, cluster_env);
    double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
    strategies->leaf_vector.push_back(ShardingStrategy(
        {"RS = RS x SS @ {1, 0} (allreduce @ 1)",
         output_spec,
         0,
         cluster_env.AllReduceCost(memory_cost, 1),
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_con_dims[0]}, {1}, cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0], rhs_space_dims[0]},
                      {1, 0}, cluster_env),
                 cluster_env),
         }}));
  }

  // RR = RS x SR
  // This is a special case where we allow spliting only one dim in the 2d-mesh case.
  // This allows some recomputation (e.g., the dense layer in the LM_head of BERT).
  if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
    HloSharding output_spec = HloSharding::Replicate();
    double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

    strategies->leaf_vector.push_back(ShardingStrategy(
        {"RR = RS x SR @ {0} (allreduce @ 0)",
         output_spec,
         cluster_env.DotCost(lhs->shape(), rhs->shape(), dot_dnums),
         cluster_env.AllReduceCost(memory_cost, 0),
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_con_dims[0]}, {0}, cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0]}, {0}, cluster_env),
                 cluster_env),
         }}));

    strategies->leaf_vector.push_back(ShardingStrategy(
        {"RR = RS x SR @ {1} (allreduce @ 1)",
         output_spec,
         cluster_env.DotCost(lhs->shape(), rhs->shape(), dot_dnums),
         cluster_env.AllReduceCost(memory_cost, 1),
         memory_cost,
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(), {lhs_con_dims[0]}, {1}, cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(), {rhs_con_dims[0]}, {1}, cluster_env),
                 cluster_env),
         }}));
  }


  // Split one batch dim
  for (int64 i = 0; i < lhs_batch_dims.size(); ++i) {
    for (int64 j = 0; j < device_mesh.num_dimensions(); ++j) {
      if (device_mesh.dim(j) == 1 ||
          ins->shape().dimensions(i) < device_mesh.dim(j)) {
        continue;
      }

      HloSharding output_spec = Tile(ins->shape(), {i}, {j}, cluster_env);
      std::string name = "Sb_" + std::to_string(i) + " = Sb x Sb @ {" +
                         std::to_string(j) + "}";
      strategies->leaf_vector.push_back(ShardingStrategy(
          {name,
           output_spec,
           0,
           0,
           GetBytes(ins->shape()) / output_spec.NumTiles(),
           {
               ReshardingCostVector(
                   strategy_map.at(lhs).get(), lhs->shape(),
                   Tile(lhs->shape(), {lhs_batch_dims[i]}, {j},
                        cluster_env),
                   cluster_env),
               ReshardingCostVector(
                   strategy_map.at(rhs).get(), rhs->shape(),
                   Tile(rhs->shape(), {rhs_batch_dims[i]}, {j},
                        cluster_env),
                   cluster_env),
           }}));
    }
  }

  // Split two batch dims
  if (lhs_batch_dims.size() == 2 && device_mesh.dim(0) > 1 &&
      device_mesh.dim(1) > 1) {
    strategies->leaf_vector.clear();

    HloSharding output_spec =
        Tile(ins->shape(), {0, 1}, {0, 1}, cluster_env);
    strategies->leaf_vector.push_back(ShardingStrategy(
        {"Sb = Sb x Sb @ {0, 1}",
         output_spec,
         0,
         0,
         GetBytes(ins->shape()) / output_spec.NumTiles(),
         {
             ReshardingCostVector(
                 strategy_map.at(lhs).get(), lhs->shape(),
                 Tile(lhs->shape(),
                      {lhs_batch_dims[0], lhs_batch_dims[1]}, {0, 1},
                      cluster_env),
                 cluster_env),
             ReshardingCostVector(
                 strategy_map.at(rhs).get(), rhs->shape(),
                 Tile(rhs->shape(),
                      {rhs_batch_dims[0], rhs_batch_dims[1]}, {0, 1},
                      cluster_env),
                 cluster_env),
         }}));
  }
}

}  // namespace spmd
}  // namespace xla
