#include "tensorflow/compiler/xla/service/spmd/auto_sharding.h"

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"

namespace xla {
namespace spmd {

// Create a HloSharding that tiles some tensor dims on some device mesh dims.
HloSharding Tile(const Shape& shape, const std::vector<int64_t> tensor_dims,
                 const std::vector<int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh) {
  CHECK_EQ(tensor_dims.size(), mesh_dims.size());
  CHECK(shape.IsArray());

  std::vector<int64_t> tile_assignment_dimensions(shape.rank(), 1);

  // Split on certain mesh dimensions
  int64_t split_prod = 1;
  for (size_t i = 0; i < tensor_dims.size(); ++i) {
    tile_assignment_dimensions[tensor_dims[i]] = device_mesh.dim(mesh_dims[i]);
    split_prod *= device_mesh.dim(mesh_dims[i]);
  }

  // Replicate on reminding mesh dimensions
  bool replicate_on_last_tile_dim = false;
  if (split_prod < device_mesh.num_elements()) {
    tile_assignment_dimensions.push_back(device_mesh.num_elements() /
                                         split_prod);
    replicate_on_last_tile_dim = true;
  }

  // Map device ids from device_mesh to tile_assignment_devices
  std::vector<int64_t> tile_assignment_devices;
  tile_assignment_devices.reserve(device_mesh.num_elements());

  std::vector<int64_t> tmp_indices(device_mesh.num_dimensions(), 0);
  std::function<void(int64_t, std::vector<int64_t>)>
      generate_tile_assignment_devices;
  generate_tile_assignment_devices = [&](int64_t tensor_dim,
                                         std::vector<int64_t> mesh_indices) {
    if (tensor_dim == shape.rank() - 1) {
      AppendFlattenElements(&tile_assignment_devices, device_mesh, mesh_indices,
                            -1, tmp_indices);
    } else {
      int64_t next_tensor_dim = tensor_dim + 1;
      int64_t next_mesh_dim = -1;

      int64_t index = GetIndex(tensor_dims, next_tensor_dim);
      if (index >= 0) {
        next_mesh_dim = mesh_dims[index];
      }

      for (int64_t i = 0; i < tile_assignment_dimensions[next_tensor_dim];
           ++i) {
        if (next_mesh_dim != -1) {
          mesh_indices[next_mesh_dim] = i;
        }
        generate_tile_assignment_devices(next_tensor_dim, mesh_indices);
      }
    }
  };

  std::vector<int64_t> mesh_indices(device_mesh.num_dimensions(), -1);
  generate_tile_assignment_devices(-1, mesh_indices);

  // Make HloSharding
  Array<int64_t> tile_assignment(tile_assignment_dimensions);
  tile_assignment.SetValues(tile_assignment_devices);

  return replicate_on_last_tile_dim
             ? HloSharding::PartialTile(std::move(tile_assignment))
             : HloSharding::Tile(std::move(tile_assignment));
}

// Compute the resharding cost vector from multiple possible strategies
// to a desired sharding spec.
std::vector<double> ReshardingCostVector(
    const StrategyVector* strategies, const Shape& shape,
    const HloSharding& required_sharding,
    const ClusterEnvironment& cluster_env) {
  // Only works with strategy vector
  CHECK(!strategies->is_tuple);
  std::vector<double> ret;
  for (const auto& x : strategies->leaf_vector) {
    ret.push_back(cluster_env.ReshardingCost(shape, x.output_sharding,
                                             required_sharding));
  }
  return ret;
}

// Create the resharding cost vector for a follow strategy.
std::vector<double> FollowInsCostVector(int64_t source_len, int64_t index) {
  std::vector<double> ret(source_len, INFINITY_COST);
  ret[index] = 0;
  return ret;
}

// Factory functions for StrategyVector.
std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, LeafStrategies& leaf_strategies) {
  std::unique_ptr<StrategyVector> strategies =
      absl::make_unique<StrategyVector>();
  strategies->is_tuple = false;
  strategies->id = leaf_strategies.size();
  leaf_strategies.push_back(strategies.get());
  strategies->instruction_id = instruction_id;
  for (int64_t i = 0; i < ins->operand_count(); ++i) {
    strategies->in_nodes.push_back(strategy_map.at(ins->operand(i)).get());
  }
  return strategies;
}

std::unique_ptr<StrategyVector> CreateTupleStrategyVector(
    size_t instruction_id) {
  std::unique_ptr<StrategyVector> strategies =
      absl::make_unique<StrategyVector>();
  strategies->is_tuple = true;
  strategies->id = -1;
  strategies->instruction_id = instruction_id;
  return strategies;
}

std::unique_ptr<StrategyVector> FollowInsStrategyVector(
    const StrategyVector* src_strategies, const Shape& shape,
    size_t instruction_id, bool have_memory_cost,
    LeafStrategies& leaf_strategies) {
  std::unique_ptr<StrategyVector> strategies;
  if (src_strategies->is_tuple) {
    CHECK(shape.IsTuple());
    CHECK_EQ(shape.tuple_shapes_size(), src_strategies->childs.size());
    strategies = CreateTupleStrategyVector(instruction_id);
    strategies->childs.reserve(src_strategies->childs.size());
    for (size_t i = 0; i < src_strategies->childs.size(); ++i) {
      strategies->childs.push_back(FollowInsStrategyVector(
          src_strategies->childs[i].get(), shape.tuple_shapes(i),
          instruction_id, have_memory_cost, leaf_strategies));
    }
  } else {
    CHECK(shape.IsArray());
    strategies = absl::make_unique<StrategyVector>();
    strategies->is_tuple = false;
    strategies->id = leaf_strategies.size();
    leaf_strategies.push_back(strategies.get());
    strategies->instruction_id = instruction_id;
    strategies->in_nodes.push_back(src_strategies);
    strategies->following = src_strategies;
    strategies->leaf_vector.reserve(src_strategies->leaf_vector.size());
    for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
      HloSharding output_spec =
          src_strategies->leaf_vector[sid].output_sharding;
      std::string name = ToStringSimple(output_spec);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost =
          have_memory_cost ? GetBytes(shape) / output_spec.NumTiles() : 0;
      std::vector<std::vector<double>> resharding_costs = {
          FollowInsCostVector(src_strategies->leaf_vector.size(), sid)};
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {}}));
    }
  }
  return strategies;
}

// Add "Replicate()" strategy
void AddReplicatedStrategy(const HloInstruction* ins,
                           const ClusterEnvironment& cluster_env,
                           const StrategyMap& strategy_map,
                           std::unique_ptr<StrategyVector>& strategies,
                           double replicated_penalty) {
  HloSharding output_spec = HloSharding::Replicate();

  std::vector<std::vector<double>> resharding_costs;
  for (int64_t k = 0; k < ins->operand_count(); ++k) {
    resharding_costs.push_back(ReshardingCostVector(
        strategy_map.at(ins->operand(k)).get(), ins->operand(k)->shape(),
        output_spec, cluster_env));
  }

  strategies->leaf_vector.push_back(
      ShardingStrategy({"R",
                        HloSharding::Replicate(),
                        replicated_penalty,
                        0,
                        GetBytes(ins->shape()),
                        std::move(resharding_costs),
                        {}}));
}

// Enumerate all 1d partition strategies.
void EnumerateAll1DPartition(const HloInstruction* ins,
                             const Array<int64_t>& device_mesh,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyVector>& strategies,
                             bool only_allow_divisible,
                             const std::string& suffix) {
  // Split one dim
  for (int64_t i = 0; i < ins->shape().rank(); ++i) {
    for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
      if (device_mesh.dim(j) == 1 ||
          ins->shape().dimensions(i) < device_mesh.dim(j)) {
        continue;
      }

      if (only_allow_divisible &&
          ins->shape().dimensions(i) % device_mesh.dim(j) != 0) {
        continue;
      }

      std::string name = absl::StrFormat("S%d @ %d", i, j) + suffix;
      HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

      std::vector<std::vector<double>> resharding_costs;
      for (int64_t k = 0; k < ins->operand_count(); ++k) {
        const HloInstruction* operand = ins->operand(k);
        if (operand->shape().rank() == 0) {
          resharding_costs.push_back(std::vector<double>(
              strategy_map.at(operand).get()->leaf_vector.size(), 0.0));
        } else {
          resharding_costs.push_back(ReshardingCostVector(
              strategy_map.at(operand).get(), ins->operand(k)->shape(),
              output_spec, cluster_env));
        }
      }
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {}}));
    }
  }
}

// Enumerate 2D partition
void EnumerateAll2DPartition(const HloInstruction* ins,
                             const Array<int64_t>& device_mesh,
                             const ClusterEnvironment& cluster_env,
                             const StrategyMap& strategy_map,
                             std::unique_ptr<StrategyVector>& strategies,
                             bool only_allow_divisible) {
  // Fully tile the buffer to 2-d mesh
  for (int64_t i = 0; i < ins->shape().rank(); ++i) {
    for (int64_t j = 0; j < ins->shape().rank(); ++j) {
      if (i == j) {
        continue;
      }
      if (ins->shape().dimensions(i) < device_mesh.dim(0) ||
          ins->shape().dimensions(j) < device_mesh.dim(1)) {
        continue;
      }

      if (only_allow_divisible &&
          (ins->shape().dimensions(i) % device_mesh.dim(0) != 0 ||
           ins->shape().dimensions(j) % device_mesh.dim(1) != 0)) {
        continue;
      }

      std::string name = absl::StrFormat("S{%d,%d} @ {0,1}", i, j);
      HloSharding output_spec = Tile(ins->shape(), {i, j}, {0, 1}, device_mesh);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
      std::vector<std::vector<double>> resharding_costs;
      for (int64_t k = 0; k < ins->operand_count(); ++k) {
        const HloInstruction* operand = ins->operand(k);
        if (operand->shape().rank() == 0) {
          resharding_costs.push_back(std::vector<double>(
              strategy_map.at(operand).get()->leaf_vector.size(), 0.0));
        } else {
          resharding_costs.push_back(
              ReshardingCostVector(strategy_map.at(operand).get(),
                                   operand->shape(), output_spec, cluster_env));
        }
      }
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {}}));
    }
  }
}

// Enumerate all 1d partition strategies.
void EnumerateAll1DPartitionReshape(const HloInstruction* ins,
                                    const Array<int64_t>& device_mesh,
                                    const ClusterEnvironment& cluster_env,
                                    const StrategyMap& strategy_map,
                                    std::unique_ptr<StrategyVector>& strategies,
                                    const std::string& suffix) {
  const HloInstruction* operand = ins->operand(0);

  for (int64_t i = 0; i < ins->shape().rank(); ++i) {
    for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
      if (device_mesh.dim(j) == 1 ||
          ins->shape().dimensions(i) < device_mesh.dim(j)) {
        continue;
      }
      HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);

      absl::optional<HloSharding> input_spec =
          hlo_sharding_util::ReshapeSharding(ins->shape(), operand->shape(),
                                             output_spec);
      if (!input_spec.has_value()) {  // invalid reshape
        continue;
      }

      std::string name = absl::StrFormat("S%d @ %d", i, j) + suffix;
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

      std::vector<std::vector<double>> resharding_costs{
          ReshardingCostVector(strategy_map.at(operand).get(), operand->shape(),
                               *input_spec, cluster_env)};
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {*input_spec}}));
    }
  }
}

// Enumerate 2D partition for reshape. Batch dim is always partitioned.
void Enumerate2DPartitionReshape(const HloInstruction* ins,
                                 const Array<int64_t>& device_mesh,
                                 const ClusterEnvironment& cluster_env,
                                 const StrategyMap& strategy_map,
                                 const InstructionBatchDimMap& batch_dim_map,
                                 std::unique_ptr<StrategyVector>& strategies) {
  auto iter = batch_dim_map.find(ins);
  if (iter == batch_dim_map.end()) {
    return;
  }

  int batch_dim = iter->second;
  const HloInstruction* operand = ins->operand(0);

  // Split batch dim + another dim
  for (int64_t i = 0; i < ins->shape().rank(); ++i) {
    for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
      if (device_mesh.dim(j) == 1 ||
          ins->shape().dimensions(i) < device_mesh.dim(j)) {
        continue;
      }
      if (batch_dim == i || 0 == j) {
        continue;
      }

      HloSharding output_spec =
          Tile(ins->shape(), {batch_dim, i}, {0, j}, device_mesh);
      absl::optional<HloSharding> input_spec =
          hlo_sharding_util::ReshapeSharding(ins->shape(), operand->shape(),
                                             output_spec);
      if (!input_spec.has_value()) {  // invalid reshape
        continue;
      }

      std::string name = absl::StrFormat("S%d%d @ {%d,%d}", batch_dim, i, 0, j);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

      std::vector<std::vector<double>> resharding_costs{
          ReshardingCostVector(strategy_map.at(operand).get(), operand->shape(),
                               *input_spec, cluster_env)};
      strategies->leaf_vector.push_back(
          ShardingStrategy({name,
                            output_spec,
                            compute_cost,
                            communication_cost,
                            memory_cost,
                            std::move(resharding_costs),
                            {*input_spec}}));
    }
  }
}

// Return the maximum number of tiles among all strategies of an instruction.
int64_t MaxNumTiles(const StrategyMap& strategy_map,
                    const HloInstruction* ins) {
  const StrategyVector* strategies = strategy_map.at(ins).get();
  // TODO(lmzheng): optimize with path compression.
  while (strategies->following != nullptr) {
    strategies = strategies->following;
  }
  int64_t max_num_tiles = -1;

  std::function<void(const StrategyVector*)> visit_all;
  visit_all = [&](const StrategyVector* stra) {
    if (stra->is_tuple) {
      for (const auto& child : stra->childs) {
        visit_all(child.get());
      }
    } else {
      for (size_t i = 0; i < stra->leaf_vector.size(); ++i) {
        max_num_tiles = std::max(
            max_num_tiles, stra->leaf_vector[i].output_sharding.NumTiles());
      }
    }
  };

  visit_all(strategies);

  return max_num_tiles;
}

// Choose an operand to follow.
// We choose to follow the operand with the highest priority.
// priority(operand) = max(x.output_spec.num_tiles for x in operand.strategies)
//
// Return `tie == True` if there are two operands with very close priorities and
// we cannot decide which one to follow.
std::pair<int64_t, bool> ChooseOperandToFollow(
    const StrategyMap& strategy_map, const InstructionDepthMap& depth_map,
    const AliasMap& alias_map,
    const absl::flat_hash_set<const HloInstruction*>& undefined_set,
    int64_t max_depth, const HloInstruction* ins) {
  int64_t follow_idx = -1;
  bool tie = false;
  double max_priority = -1e20;
  double depth_normalizer = 0.1 / max_depth;
  double range_delta = 4 * depth_normalizer;

  for (int64_t i = 0; i < ins->operand_count(); ++i) {
    const HloInstruction* operand = ins->operand(i);
    if (!undefined_set.count(operand)) {
      double priority = MaxNumTiles(strategy_map, operand) +
                        depth_map.at(operand) * depth_normalizer;
      if (priority > max_priority + range_delta) {
        follow_idx = i;
        tie = false;
        max_priority = priority;
      } else if (priority >= max_priority - range_delta) {
        tie = true;
      }
    }

    // If an alias constraint is set, always follow its alias source.
    auto it = alias_map.find(ins);
    if (it != alias_map.end() && it->second == operand) {
      break;
    }
  }
  CHECK_GE(follow_idx, 0);

  return std::make_pair(follow_idx, tie);
}

// Return whether an instruciton can follow one of its operand when
// more than one operand have the same priority.
bool AllowTieFollowing(const HloInstruction* ins) {
  if (ins->opcode() == HloOpcode::kCompare ||
      ins->opcode() == HloOpcode::kAnd) {
    // This is used to resolve tricky cases where an iota and a parameter
    // has the same priority. This happens for embedding, onehot or
    // make_attention_mask.
    return false;
  }
  if (ins->operand_count() == 3) {
    return false;
  }
  return true;
}

// Build possible sharding strategies and their costs for all instructions.
StatusOr<std::tuple<StrategyMap, LeafStrategies, AssociativeDotPairs>>
BuildStrategyAndCost(const HloInstructionSequence& sequence,
                     const InstructionDepthMap& depth_map,
                     const InstructionBatchDimMap& batch_dim_map,
                     const AliasMap& alias_map,
                     const ClusterEnvironment& cluster_env,
                     AutoShardingSolverOption& solver_option) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d;
  StrategyMap strategy_map;
  LeafStrategies leaf_strategies;
  AssociativeDotPairs associative_dot_pairs;
  absl::flat_hash_set<const HloInstruction*> undefined_set;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Count the non-one mesh dimension.
  int mesh_nn_dims = 0;
  for (int dim : device_mesh.dimensions()) {
    if (dim > 1) {
      mesh_nn_dims++;
    }
  }

  // Gather all output values
  absl::flat_hash_set<const HloInstruction*> output_set;
  for (size_t i = 0; i < instructions.back()->operand_count(); ++i) {
    output_set.insert(instructions.back()->operand(i));
  }

  // Add penalty for replicated tensors
  double replicated_penalty = std::round(cluster_env.AllReduceCost(1, 0) +
                                         cluster_env.AllReduceCost(1, 1));

  int64_t max_depth = -1;
  for (auto iter : depth_map) {
    max_depth = std::max(max_depth, iter.second);
  }

  int64_t disallowed_follow = 0;

  // Register strategies and their costs for each instruction.
  for (size_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
    const HloInstruction* ins = instructions[instruction_id];
    std::unique_ptr<StrategyVector> strategies;
    HloOpcode opcode = ins->opcode();

    switch (opcode) {
      case HloOpcode::kParameter:
      case HloOpcode::kRng: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        // Split 1 dim
        EnumerateAll1DPartition(ins, device_mesh, cluster_env, strategy_map,
                                strategies, true, "");

        // Split 2 dims
        if (cluster_env.non_zero_mesh_dims.size() > 1) {
          // Add penalty for 1d partial tiled layout
          for (size_t i = 0; i < strategies->leaf_vector.size(); ++i) {
            strategies->leaf_vector[i].compute_cost += replicated_penalty * 0.8;
          }

          if (batch_dim_map.count(ins)) {
            // This is a pruning heuristic: only allow 2d partition
            // for parameters with a batch dim. These parameters are
            // typically input data and intermediate activations.
            EnumerateAll2DPartition(ins, device_mesh, cluster_env, strategy_map,
                                    strategies, true);
          }

          if (solver_option.allow_mixed_mesh_shape) {
            // Split 1 dim, but for 1d mesh
            EnumerateAll1DPartition(ins, device_mesh_1d, cluster_env,
                                    strategy_map, strategies, true, " 1d");
          }
        }

        if (solver_option.allow_replicated_parameters) {
          AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies,
                                replicated_penalty);
        }

        RemoveDuplicatedStrategy(strategies);

        // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
        // and only keep the data parallel strategies.
        if (solver_option.force_batch_dim_to_mesh_dim >= 0 &&
            batch_dim_map.count(ins)) {
          TF_RETURN_IF_ERROR(FilterStrategy(ins, strategies, cluster_env,
                                            batch_dim_map, solver_option));
        }
        break;
      }
      case HloOpcode::kConstant: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies, 0);
        break;
      }
      case HloOpcode::kBroadcast: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.count(operand)) {
          break;
        }

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec = BroadcastSharding(
              src_strategies->leaf_vector[sid].output_sharding, ins->shape(),
              ins->dimensions());

          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name,
               output_spec,
               compute_cost,
               communication_cost,
               memory_cost,
               {FollowInsCostVector(src_strategies->leaf_vector.size(), sid)},
               {}}));
        }

        // If the operand is a scalar, following it only generates "Replicated"
        // strategy. So we should register new strategies instead of following
        // it.
        if (operand->shape().rank() == 0) {
          if (!output_set.count(ins) &&
              operand->opcode() == HloOpcode::kConstant) {
            // one execption: always replicate intermidiate broadcasted
            // constants.
            break;
          }

          strategies->following = nullptr;
          strategies->leaf_vector.clear();

          // Split one dim
          for (int64_t i = 0; i < ins->shape().rank(); ++i) {
            for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
              if (device_mesh.dim(j) == 1 ||
                  ins->shape().dimensions(i) < device_mesh.dim(j)) {
                continue;
              }

              std::string name = absl::StrFormat("S%d @ %d", i, j);
              HloSharding output_spec =
                  Tile(ins->shape(), {i}, {j}, device_mesh);
              double compute_cost = 0, communication_cost = 0;
              double memory_cost =
                  GetBytes(ins->shape()) / output_spec.NumTiles();
              strategies->leaf_vector.push_back(ShardingStrategy(
                  {name,
                   output_spec,
                   compute_cost,
                   communication_cost,
                   memory_cost,
                   {std::vector<double>(src_strategies->leaf_vector.size(),
                                        0.0)},
                   {}}));
            }
          }

          // Replicate
          AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies,
                                replicated_penalty);

          RemoveDuplicatedStrategy(strategies);
        }

        break;
      }
      case HloOpcode::kReshape: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        const HloInstruction* operand = ins->operand(0);

        // Create follow strategies
        if (!undefined_set.count(operand) &&
            ((ins->users().size() == 1 && !IsBatchDimSwitchReshape(ins)) ||
             (mesh_nn_dims >= 2 && !solver_option.allow_mixed_mesh_shape))) {
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          CHECK(!src_strategies->is_tuple);
          strategies->following = src_strategies;

          for (int64_t sid = 0; sid < src_strategies->leaf_vector.size();
               ++sid) {
            absl::optional<HloSharding> output_spec =
                hlo_sharding_util::ReshapeSharding(
                    operand->shape(), ins->shape(),
                    src_strategies->leaf_vector[sid].output_sharding);

            if (!output_spec.has_value()) {
              continue;
            }

            if (!IsValidTileAssignment(*output_spec)) {
              continue;
            }

            std::string name = ToStringSimple(*output_spec);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec->NumTiles();
            strategies->leaf_vector.push_back(ShardingStrategy(
                {name,
                 *output_spec,
                 compute_cost,
                 communication_cost,
                 memory_cost,
                 {FollowInsCostVector(src_strategies->leaf_vector.size(), sid)},
                 {}}));
          }
        }

        // Fail to create follow strategies, enumerate all possible cases
        if (strategies->leaf_vector.empty()) {
          strategies->leaf_vector.clear();
          strategies->following = nullptr;

          // Split 1 dim
          EnumerateAll1DPartitionReshape(ins, device_mesh, cluster_env,
                                         strategy_map, strategies, "");

          if (solver_option.allow_mixed_mesh_shape &&
              cluster_env.non_zero_mesh_dims.size() > 1) {
            // Split 1 dim, but for 1d mesh
            EnumerateAll1DPartitionReshape(ins, device_mesh_1d, cluster_env,
                                           strategy_map, strategies, " 1d");

            // Split 2 dim, one is always the batch dim
            Enumerate2DPartitionReshape(ins, device_mesh, cluster_env,
                                        strategy_map, batch_dim_map,
                                        strategies);
          }

          // Replicate
          AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies,
                                replicated_penalty);

          RemoveDuplicatedStrategy(strategies);
        }
        break;
      }
      case HloOpcode::kTranspose:
      case HloOpcode::kReverse: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.count(operand)) {
          break;
        }

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec = Undefined();

          if (opcode == HloOpcode::kTranspose) {
            output_spec = hlo_sharding_util::TransposeSharding(
                src_strategies->leaf_vector[sid].output_sharding,
                ins->dimensions());
          } else {
            output_spec = hlo_sharding_util::ReverseSharding(
                src_strategies->leaf_vector[sid].output_sharding,
                ins->dimensions());
          }

          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name,
               output_spec,
               compute_cost,
               communication_cost,
               memory_cost,
               {FollowInsCostVector(src_strategies->leaf_vector.size(), sid)},
               {}}));
        }
        break;
      }
      case HloOpcode::kPad:
      case HloOpcode::kSlice:
      case HloOpcode::kConcatenate:  // TODO(lmzheng): revisit concatenate
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSelectAndScatter: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        // Choose an operand to follow
        int64_t follow_idx;
        bool tie;
        std::tie(follow_idx, tie) = ChooseOperandToFollow(
            strategy_map, depth_map, alias_map, undefined_set, max_depth, ins);

        // Create follow strategies
        const HloInstruction* operand = ins->operand(follow_idx);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          absl::optional<HloSharding> output_spec;

          switch (opcode) {
            case HloOpcode::kPad:
            case HloOpcode::kSlice:
            case HloOpcode::kConcatenate:
            case HloOpcode::kDynamicSlice:
            case HloOpcode::kDynamicUpdateSlice:
              output_spec = PropagateDimwiseSharding(
                  src_strategies->leaf_vector[sid].output_sharding,
                  operand->shape(), ins->shape());
              break;
            case HloOpcode::kReduceWindow:
            case HloOpcode::kSelectAndScatter:
              output_spec = PropagateReduceWindowSharding(
                  src_strategies->leaf_vector[sid].output_sharding,
                  operand->shape(), ins->window());
              break;
            default:
              LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
          }

          if (!output_spec.has_value()) {
            continue;
          }

          std::string name = ToStringSimple(*output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec->NumTiles();
          std::vector<std::vector<double>> resharding_costs;
          for (int64_t k = 0; k < ins->operand_count(); ++k) {
            if (k == follow_idx) {
              resharding_costs.push_back(
                  FollowInsCostVector(src_strategies->leaf_vector.size(), sid));
            } else {
              operand = ins->operand(k);
              if (operand->shape().rank() > 0) {
                resharding_costs.push_back(ReshardingCostVector(
                    strategy_map.at(operand).get(), operand->shape(),
                    *output_spec, cluster_env));
              } else {
                resharding_costs.push_back(std::vector<double>(
                    strategy_map.at(operand)->leaf_vector.size(), 0.0));
              }
            }
          }
          strategies->leaf_vector.push_back(
              ShardingStrategy({name,
                                *output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                std::move(resharding_costs),
                                {}}));
        }
        break;
      }
      case HloOpcode::kGather: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        auto dnums = ins->gather_dimension_numbers();

        // Split one update_window_dims
        for (size_t i = 0; i < dnums.offset_dims().size(); ++i) {
          for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
            if (device_mesh.dim(j) == 1 ||
                ins->shape().dimensions(i) < device_mesh.dim(j)) {
              continue;
            }

            HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);

            int operand_dim = dnums.offset_dims(i);

            CHECK_EQ(ins->shape().dimensions(operand_dim),
                     ins->operand(0)->shape().dimensions(operand_dim));

            std::vector<HloSharding> operand_specs{
                Tile(ins->operand(0)->shape(), {operand_dim}, {j}, device_mesh),
                HloSharding::Replicate(),
            };

            std::string name = ToStringSimple(output_spec);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec.NumTiles();
            std::vector<std::vector<double>> resharding_costs;
            for (int64_t k = 0; k < ins->operand_count(); ++k) {
              resharding_costs.push_back(ReshardingCostVector(
                  strategy_map.at(ins->operand(k)).get(),
                  ins->operand(k)->shape(), operand_specs[k], cluster_env));
            }

            strategies->leaf_vector.push_back(ShardingStrategy(
                {name, output_spec, compute_cost, communication_cost,
                 memory_cost, std::move(resharding_costs),
                 std::move(operand_specs)}));
          }
        }

        // Replicate all
        AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies, 0);
        break;
      }
      case HloOpcode::kScatter: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        auto dnums = ins->scatter_dimension_numbers();

        // Split one update_window_dims
        for (size_t i = 0; i < dnums.update_window_dims().size(); ++i) {
          for (int64_t j = 0; j < device_mesh.num_dimensions(); ++j) {
            if (device_mesh.dim(j) == 1 ||
                ins->shape().dimensions(i) < device_mesh.dim(j)) {
              continue;
            }

            HloSharding output_spec = Tile(ins->shape(), {i}, {j}, device_mesh);

            int operand_dim = dnums.update_window_dims(i);
            int update_dim = operand_dim;

            CHECK_EQ(ins->shape().dimensions(operand_dim),
                     ins->operand(0)->shape().dimensions(operand_dim));
            CHECK_EQ(ins->shape().dimensions(operand_dim),
                     ins->operand(2)->shape().dimensions(update_dim));

            std::vector<HloSharding> operand_specs{
                Tile(ins->operand(0)->shape(), {operand_dim}, {j}, device_mesh),
                HloSharding::Replicate(),
                Tile(ins->operand(2)->shape(), {update_dim}, {j}, device_mesh),
            };

            std::string name = ToStringSimple(output_spec);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec.NumTiles();
            std::vector<std::vector<double>> resharding_costs;
            for (int64_t k = 0; k < ins->operand_count(); ++k) {
              resharding_costs.push_back(ReshardingCostVector(
                  strategy_map.at(ins->operand(k)).get(),
                  ins->operand(k)->shape(), operand_specs[k], cluster_env));
            }

            strategies->leaf_vector.push_back(ShardingStrategy(
                {name, output_spec, compute_cost, communication_cost,
                 memory_cost, std::move(resharding_costs),
                 std::move(operand_specs)}));
          }
        }

        // Replicate all
        AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies, 0);
        break;
      }
      // Unary elementwise operations.
      case HloOpcode::kAbs:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kCeil:
      case HloOpcode::kClz:
      case HloOpcode::kConvert:
      case HloOpcode::kBitcast:
      case HloOpcode::kBitcastConvert:
      case HloOpcode::kCopy:
      case HloOpcode::kCos:
      case HloOpcode::kExp:
      case HloOpcode::kExpm1:
      case HloOpcode::kFloor:
      case HloOpcode::kImag:
      case HloOpcode::kIsFinite:
      case HloOpcode::kLog:
      case HloOpcode::kLog1p:
      case HloOpcode::kNot:
      case HloOpcode::kNegate:
      case HloOpcode::kPopulationCount:
      case HloOpcode::kReal:
      case HloOpcode::kReducePrecision:
      case HloOpcode::kRsqrt:
      case HloOpcode::kLogistic:
      case HloOpcode::kSign:
      case HloOpcode::kSin:
      case HloOpcode::kSqrt:
      case HloOpcode::kCbrt:
      case HloOpcode::kTanh:
      // Binary elementwise operations
      case HloOpcode::kAdd:
      case HloOpcode::kAtan2:
      case HloOpcode::kCompare:
      case HloOpcode::kComplex:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kMultiply:
      case HloOpcode::kPower:
      case HloOpcode::kRemainder:
      case HloOpcode::kSubtract:
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightArithmetic:
      case HloOpcode::kShiftRightLogical:
      // Ternary elementwise operations.
      case HloOpcode::kSelect:
      case HloOpcode::kClamp: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        // Choose an operand to follow
        int64_t follow_idx;
        bool tie;
        std::tie(follow_idx, tie) = ChooseOperandToFollow(
            strategy_map, depth_map, alias_map, undefined_set, max_depth, ins);

        if (!tie || AllowTieFollowing(ins)) {
          strategies->following =
              strategy_map.at(ins->operand(follow_idx)).get();
        } else {
          strategies->following = nullptr;
          disallowed_follow++;
        }

        // Get all possible sharding specs from operands
        for (int64_t i = 0; i < ins->operand_count(); ++i) {
          const StrategyVector* src_strategies =
              strategy_map.at(ins->operand(i)).get();
          CHECK(!src_strategies->is_tuple);

          if (strategies->following != nullptr &&
              strategies->following != src_strategies) {
            // If ins follows one operand, do not consider sharding specs from
            // other operands.
            continue;
          }

          for (int64_t sid = 0; sid < src_strategies->leaf_vector.size();
               ++sid) {
            HloSharding output_spec =
                src_strategies->leaf_vector[sid].output_sharding;

            std::string name = ToStringSimple(output_spec);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec.NumTiles();
            std::vector<std::vector<double>> resharding_costs;
            for (int64_t k = 0; k < ins->operand_count(); ++k) {
              if (strategies->following == src_strategies && k == follow_idx) {
                resharding_costs.push_back(FollowInsCostVector(
                    src_strategies->leaf_vector.size(), sid));
              } else {
                resharding_costs.push_back(ReshardingCostVector(
                    strategy_map.at(ins->operand(k)).get(),
                    ins->operand(k)->shape(), output_spec, cluster_env));
              }
            }

            strategies->leaf_vector.push_back(
                ShardingStrategy({name,
                                  output_spec,
                                  compute_cost,
                                  communication_cost,
                                  memory_cost,
                                  std::move(resharding_costs),
                                  {}}));
          }
        }

        if (strategies->following == nullptr) {
          RemoveDuplicatedStrategy(strategies);
        }

        if (ins->opcode() == HloOpcode::kAdd) {
          // Adjust the resharding costs for AllReduceReassociate pass.
          // The AllReduceReassociate pass can simplify
          // allreduce(x) + allreduce(y) to allreduce(x + y),
          // so we adjust the resharidng costs to reflect this optimization.

          // TODO(lmzheng): The current implementation only works for
          // x = a + b. We also need to cover cases where there are
          // more than two operands (i.e., x = a + b + c).
          if (ins->operand(0)->opcode() == HloOpcode::kDot &&
              ins->operand(1)->opcode() == HloOpcode::kDot) {
            associative_dot_pairs.push_back(
                {strategy_map.at(ins->operand(0)).get(),
                 strategy_map.at(ins->operand(1)).get()});
          }
        }
        break;
      }
      case HloOpcode::kReduce: {
        // Choose an operand to follow
        int64_t follow_idx;
        bool tie;
        std::tie(follow_idx, tie) = ChooseOperandToFollow(
            strategy_map, depth_map, alias_map, undefined_set, max_depth, ins);
        CHECK_LT(follow_idx, ins->operand_count() / 2);

        const HloInstruction* operand = ins->operand(follow_idx);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);

        // Create main child strategy for the 0-th output
        auto main_stra = CreateLeafStrategyVector(
            instruction_id, ins, strategy_map, leaf_strategies);
        main_stra->following = src_strategies;

        // Map old dims to new dims
        const auto& dimensions = ins->dimensions();
        std::vector<int64_t> old_dim_to_new_dim;
        old_dim_to_new_dim.reserve(operand->shape().rank());
        int64_t pt = 0;
        for (int64_t old_dim = 0; old_dim < operand->shape().rank();
             ++old_dim) {
          if (absl::c_find(dimensions, old_dim) != dimensions.end()) {
            old_dim_to_new_dim.push_back(-1);
          } else {
            old_dim_to_new_dim.push_back(pt);
            pt += 1;
          }
        }

        Shape one_output_shape;
        if (ins->shape().IsTuple()) {
          one_output_shape = ins->shape().tuple_shapes(0);
        } else {
          one_output_shape = ins->shape();
        }
        CHECK_EQ(pt, one_output_shape.rank());

        // Create follow strategies
        for (size_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          const HloSharding& input_spec =
              src_strategies->leaf_vector[sid].output_sharding;
          const auto& tensor_dim_to_mesh_dim =
              cluster_env.GetTensorDimToMeshDim(operand->shape(), input_spec);

          std::vector<int64_t> tile_tensor_dims, tile_mesh_dims,
              all_reduce_dims;

          for (int64_t tensor_dim = 0; tensor_dim < operand->shape().rank();
               ++tensor_dim) {
            int64_t mesh_dim = tensor_dim_to_mesh_dim[tensor_dim];
            if (absl::c_find(dimensions, tensor_dim) != dimensions.end()) {
              if (mesh_dim == -1) {  // Reduce on a replicated dim
                continue;
              } else {  // Reduce on a split dim. Require an allreduce
                all_reduce_dims.push_back(mesh_dim);
              }
            } else {
              if (mesh_dim == -1) {  // Follow a replicated dim
                continue;
              } else {  // Follow a split dim
                tile_tensor_dims.push_back(old_dim_to_new_dim[tensor_dim]);
                tile_mesh_dims.push_back(mesh_dim);
              }
            }
          }

          // Check validity
          bool valid = true;
          for (int64_t x : tile_mesh_dims) {
            if (x >= device_mesh.num_dimensions()) {
              valid = false;
            }
          }
          for (int64_t x : all_reduce_dims) {
            if (x >= device_mesh.num_dimensions()) {
              valid = false;
            }
          }
          if (!valid) {
            continue;
          }

          std::string name, suffix;
          HloSharding output_spec = Undefined();
          if (solver_option.allow_mixed_mesh_shape &&
              cluster_env.non_zero_mesh_dims.size() > 1) {
            std::vector<int> operand_tensor_dim_to_mesh_dim;
            int n_dim;
            std::tie(operand_tensor_dim_to_mesh_dim, n_dim) =
                GetTensorDimToMeshDimInternal(operand->shape(), input_spec);

            if (n_dim == 1) {
              output_spec = Tile(one_output_shape, tile_tensor_dims,
                                 tile_mesh_dims, device_mesh_1d);
              suffix = " 1d";
            } else {
              output_spec = Tile(one_output_shape, tile_tensor_dims,
                                 tile_mesh_dims, device_mesh);
            }
          } else {
            output_spec = Tile(one_output_shape, tile_tensor_dims,
                               tile_mesh_dims, device_mesh);
          }

          name += ToStringSimple(output_spec);
          if (!all_reduce_dims.empty()) {
            name += " (allreduce @ " + ToString(all_reduce_dims) + ")";
          }
          name += suffix;

          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          for (auto mesh_dim : all_reduce_dims) {
            communication_cost +=
                cluster_env.AllReduceCost(memory_cost, mesh_dim);
          }

          std::vector<std::vector<double>> resharding_costs;

          for (int64_t k = 0; k < ins->operand_count(); ++k) {
            if (k == follow_idx) {
              resharding_costs.push_back(
                  FollowInsCostVector(src_strategies->leaf_vector.size(), sid));
            } else if (k < ins->operand_count() / 2) {
              resharding_costs.push_back(ReshardingCostVector(
                  strategy_map.at(ins->operand(k)).get(),
                  ins->operand(k)->shape(), input_spec, cluster_env));
            } else {
              resharding_costs.push_back(
                  ReshardingCostVector(strategy_map.at(ins->operand(k)).get(),
                                       ins->operand(k)->shape(),
                                       HloSharding::Replicate(), cluster_env));
            }
          }

          main_stra->leaf_vector.push_back(ShardingStrategy(
              {name, output_spec, compute_cost, communication_cost, memory_cost,
               std::move(resharding_costs)}));
        }

        if (ins->shape().IsTuple()) {  // Variadic Reduce
          // Create following child strategies for other outputs
          // All costs are considered in the main_strategy, so we don't
          // need to compute costs for other outputs. We only need
          // to set the correct outptut_sharding for them and let them
          // follow the main strategy.
          strategies = CreateTupleStrategyVector(instruction_id);
          strategies->childs.reserve(ins->operand_count() / 2);
          const StrategyVector* main_stra_ptr = main_stra.get();
          for (size_t i = 0; i < ins->operand_count() / 2; ++i) {
            if (i == follow_idx) {
              strategies->childs.push_back(std::move(main_stra));
            } else {
              strategies->childs.push_back(std::move(FollowInsStrategyVector(
                  main_stra_ptr, ins->operand(i)->shape(), instruction_id, true,
                  leaf_strategies)));
            }
          }
        } else {
          strategies = std::move(main_stra);
        }
        break;
      }
      case HloOpcode::kDot: {
        TF_RETURN_IF_ERROR(HandleDot(strategies, leaf_strategies, strategy_map,
                                     ins, instruction_id, cluster_env,
                                     batch_dim_map, solver_option));
        break;
      }
      case HloOpcode::kConvolution: {
        TF_RETURN_IF_ERROR(HandleConv(strategies, leaf_strategies, strategy_map,
                                      ins, instruction_id, cluster_env,
                                      batch_dim_map, solver_option));
        break;
      }
      case HloOpcode::kRngGetAndUpdateState: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);
        AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies, 0);
        break;
      }
      case HloOpcode::kIota: {
        strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                              leaf_strategies);

        // Split 2 dims
        EnumerateAll2DPartition(ins, device_mesh, cluster_env, strategy_map,
                                strategies, false);

        if (solver_option.allow_mixed_mesh_shape &&
            cluster_env.non_zero_mesh_dims.size() > 1) {
          // Split 1 dim, but for 1d mesh
          EnumerateAll1DPartition(ins, device_mesh_1d, cluster_env,
                                  strategy_map, strategies, false, " 1d");
        }

        if (strategies->leaf_vector.empty() || IsFollowedByBroadcast(ins)) {
          // Replicate
          AddReplicatedStrategy(ins, cluster_env, strategy_map, strategies,
                                replicated_penalty * 5);
        }

        RemoveDuplicatedStrategy(strategies);
        break;
      }
      case HloOpcode::kSort: {
        // Choose an operand to follow
        int64_t follow_idx;
        bool tie;
        std::tie(follow_idx, tie) = ChooseOperandToFollow(
            strategy_map, depth_map, alias_map, undefined_set, max_depth, ins);
        const HloInstruction* operand = ins->operand(follow_idx);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();

        // Create main child strategy for the 0-th output
        auto main_stra = CreateLeafStrategyVector(
            instruction_id, ins, strategy_map, leaf_strategies);
        main_stra->following = src_strategies;

        for (int64_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec =
              src_strategies->leaf_vector[sid].output_sharding;

          std::string name = ToStringSimple(output_spec);
          double compute_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();

          bool need_all_gather = false;
          for (int64_t dim : ins->dimensions()) {
            if (output_spec.tile_assignment().dim(dim) > 1) {
              need_all_gather = true;
              break;
            }
          }

          double communication_cost =
              need_all_gather
                  ? cluster_env.ReshardingCost(operand->shape(), output_spec,
                                               HloSharding::Replicate())
                  : 0;
          HloSharding input_spec =
              need_all_gather ? HloSharding::Replicate() : output_spec;

          std::vector<std::vector<double>> resharding_costs;
          for (int64_t k = 0; k < ins->operand_count(); ++k) {
            if (k == follow_idx) {
              resharding_costs.push_back(
                  FollowInsCostVector(src_strategies->leaf_vector.size(), sid));
            } else {
              resharding_costs.push_back(ReshardingCostVector(
                  strategy_map.at(ins->operand(k)).get(),
                  ins->operand(k)->shape(), input_spec, cluster_env));
            }
          }

          main_stra->leaf_vector.push_back(
              ShardingStrategy({name,
                                output_spec,
                                compute_cost,
                                communication_cost,
                                memory_cost,
                                std::move(resharding_costs),
                                {}}));
        }

        if (ins->shape().IsTuple()) {
          // Create following child strategies for other outputs.
          // All costs are considered in the main_strategy, so we don't
          // need to compute costs for other outputs. We only need
          // to set the correct outptut_sharding for them and let them
          // follow the main strategy.
          strategies = CreateTupleStrategyVector(instruction_id);
          strategies->childs.reserve(ins->operand_count());
          const StrategyVector* main_stra_ptr = main_stra.get();
          for (size_t i = 0; i < ins->operand_count(); ++i) {
            if (i == follow_idx) {
              strategies->childs.push_back(std::move(main_stra));
            } else {
              strategies->childs.push_back(std::move(FollowInsStrategyVector(
                  main_stra_ptr, ins->operand(i)->shape(), instruction_id, true,
                  leaf_strategies)));
            }
          }
        } else {
          strategies = std::move(main_stra);
        }

        break;
      }
      case HloOpcode::kTuple: {
        strategies = CreateTupleStrategyVector(instruction_id);
        strategies->childs.reserve(ins->operand_count());
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          strategies->childs.push_back(FollowInsStrategyVector(
              src_strategies, operand->shape(), instruction_id,
              /* have_memory_cost= */ false, leaf_strategies));
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* operand = ins->operand(0);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(src_strategies->is_tuple);
        strategies = FollowInsStrategyVector(
            src_strategies->childs[ins->tuple_index()].get(), ins->shape(),
            instruction_id,
            /* have_memory_cost= */ false, leaf_strategies);
        break;
      }
      case HloOpcode::kCustomCall: {
        if (IsCustomCallMarker(ins)) {
          const HloInstruction* operand = ins->operand(0);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          CHECK(src_strategies->is_tuple);
          strategies = FollowInsStrategyVector(
              src_strategies, ins->shape(), instruction_id,
              /* have_memory_cost= */ false, leaf_strategies);
        } else {
          LOG(FATAL) << "Unknown CustomCall instruction: " + ins->ToString();
        }
        break;
      }
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
    }

    // Debug options: forcibly set the the strategy of some instructions.
    if (pass_context::GetBool("auto_sharding::force_strategy", false)) {
      std::vector<int64_t> inst_indices = pass_context::GetIntVector(
          "auto_sharding::force_strategy_inst_indices");
      std::vector<std::string> stra_names = pass_context::GetStringVector(
          "auto_sharding::force_strategy_stra_names");
      CHECK_EQ(inst_indices.size(), stra_names.size());
      auto it = absl::c_find(inst_indices, strategies->id);

      if (it != inst_indices.end()) {
        CHECK(!strategies->is_tuple);
        std::vector<ShardingStrategy> new_leaf_vector;
        int64_t idx = it - inst_indices.begin();

        for (const auto& stra : strategies->leaf_vector) {
          if (stra.name == stra_names[idx]) {
            new_leaf_vector.push_back(stra);
          }
        }

        strategies->leaf_vector = std::move(new_leaf_vector);
      }
    }

    CHECK(strategies->is_tuple || !strategies->leaf_vector.empty())
        << ins->ToString() << " does not have any valid strategies.";
    strategy_map[ins] = std::move(strategies);
  }

  // If gradient accumulation is used, adjust the cost of all-reduce for
  // gradient synchronization.
  if (solver_option.grad_acc_num_micro_batches > 1) {
    // find gradientt-computatin instructions
    std::vector<const HloInstruction*> grad_insts =
        GetGradientComputationInstructions(instructions);
    for (const HloInstruction* inst : grad_insts) {
      StrategyVector* stra_vector = strategy_map[inst].get();
      CHECK(!stra_vector->is_tuple);

      for (auto& stra : stra_vector->leaf_vector) {
        if (stra.name.find("allreduce") != std::string::npos) {
          stra.communication_cost /= solver_option.grad_acc_num_micro_batches;
        }
      }
    }
  }

  return std::make_tuple(std::move(strategy_map), std::move(leaf_strategies),
                         std::move(associative_dot_pairs));
}

// Handle alias: alias pairs must have the same HloSharding.
// To deal with alias, we do special process both before and after
// BuildStrategyAndCost. Because it is easier to handle elementwise instructions
// before BuildStrategyAndCost and it is easier to handle dot/conv instructions
// after BuildStrategyAndCost.
// Before BuildStrategyAndCost, we build an AliasMap to guide the generation of
// strategies. After BuildStrategyAndCost, we use AliasSet to add alias
// constraints in the ILP problem.

AliasMap BuildAliasMap(const HloModule* module,
                       const HloDataflowAnalysis& dataflow_analysis) {
  AliasMap alias_map;

  const HloInputOutputAliasConfig& alias_config =
      module->input_output_alias_config();

  HloComputation* entry = module->entry_computation();
  const std::vector<HloInstruction*>& parameter_instructions =
      entry->parameter_instructions();
  const HloInstruction* output_tuple = entry->root_instruction();

  if (IsCustomCallMarker(output_tuple)) {
    output_tuple = output_tuple->operand(0);
  }

  alias_config.ForEachAlias([&](const ShapeIndex& output_index,
                                const HloInputOutputAliasConfig::Alias& alias) {
    CHECK_EQ(alias.parameter_index.size(), 0) << "Do not support tuple alias";
    CHECK_EQ(output_index.size(), 1) << "Do not support tuple alias";

    HloInstruction* src_ins = parameter_instructions[alias.parameter_number];
    const HloInstruction* dst_ins = output_tuple->operand(output_index.front());
    alias_map[dst_ins] = src_ins;
  });

  return alias_map;
}

AliasSet BuildAliasSet(const HloModule* module,
                       const HloDataflowAnalysis& dataflow_analysis,
                       const StrategyMap& strategy_map) {
  // Adjust the edge cost for aliases (donated buffer).
  // Typically, old weights and new weights are aliases, so we should
  // let them have the same sharding spec.
  const HloInputOutputAliasConfig& alias_config =
      module->input_output_alias_config();

  HloComputation* entry = module->entry_computation();
  const std::vector<HloInstruction*>& parameter_instructions =
      entry->parameter_instructions();
  const HloInstruction* output_tuple = entry->root_instruction();

  AliasSet alias_set;
  std::function<void(const StrategyVector*, const StrategyVector*)>
      traverse_tuple_alias;
  traverse_tuple_alias = [&](const StrategyVector* src_strategies,
                             const StrategyVector* dst_strategies) {
    if (src_strategies->is_tuple) {
      CHECK(dst_strategies->is_tuple);
      CHECK_EQ(src_strategies->childs.size(), dst_strategies->childs.size());
      for (size_t i = 0; i < src_strategies->childs.size(); ++i) {
        traverse_tuple_alias(src_strategies->childs[i].get(),
                             dst_strategies->childs[i].get());
      }
    } else {
      alias_set.insert(std::make_pair(src_strategies->id, dst_strategies->id));
    }
  };
  alias_config.ForEachAlias([&](const ShapeIndex& output_index,
                                const HloInputOutputAliasConfig::Alias& alias) {
    CHECK_EQ(alias.parameter_index.size(), 0) << "Do not support tuple alias";
    CHECK_EQ(output_index.size(), 1) << "Do not support tuple alias";

    const HloInstruction* src_ins =
        parameter_instructions[alias.parameter_number];

    traverse_tuple_alias(strategy_map.at(src_ins).get(),
                         strategy_map.at(output_tuple)
                             .get()
                             ->childs[output_index.front()]
                             .get());
  });

  return alias_set;
}

// Serialize parameters of the ILP problem as numpy arrays and call the python
// solver.
std::tuple<std::vector<int64_t>, std::vector<int64_t>, double> CallSolver(
    const HloInstructionSequence& sequence, const LivenessSet& liveness_set,
    const StrategyMap& strategy_map, const LeafStrategies& leaf_strategies,
    const CostGraph& cost_graph, const AliasSet& alias_set) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Serialize edges and edge costs to 1d numpy arrays
  int64_t N = leaf_strategies.size();
  int64_t M =
      pass_context::GetInt("auto_sharding::memory_budget_per_device", -1);
  std::vector<int> s_len_np = cost_graph.node_lens;
  const std::vector<int>& s_follow_np = cost_graph.follow_idx;
  std::vector<int> E_np;
  std::vector<double> r_np;
  for (const auto& iter : cost_graph.edge_costs) {
    int src = iter.first.first;
    int dst = iter.first.second;
    Matrix edge_cost = iter.second;

    E_np.push_back(src);
    E_np.push_back(dst);

    CHECK_EQ(edge_cost.n, s_len_np[src]);
    CHECK_EQ(edge_cost.m, s_len_np[dst]);

    for (size_t i = 0; i < edge_cost.n; i++) {
      for (size_t j = 0; j < edge_cost.m; j++) {
        r_np.push_back(edge_cost(i, j));
      }
    }
  }

  // Serialize node costs
  std::vector<double> c_np, d_np, m_np;
  for (size_t i = 0; i < N; ++i) {
    const StrategyVector* strategies = leaf_strategies[i];
    if (s_follow_np[i] < 0) {
      for (size_t j = 0; j < strategies->leaf_vector.size(); ++j) {
        c_np.push_back(strategies->leaf_vector[j].compute_cost);
        d_np.push_back(strategies->leaf_vector[j].communication_cost +
                       cost_graph.extra_node_costs[i][j]);
        m_np.push_back(strategies->leaf_vector[j].memory_cost);
      }
    } else {
      std::vector<int> reindexing = cost_graph.reindexing_vector.at(i);
      s_len_np[i] = reindexing.size();
      for (size_t k = 0; k < reindexing.size(); ++k) {
        size_t j = reindexing[k];
        c_np.push_back(strategies->leaf_vector[j].compute_cost);
        d_np.push_back(strategies->leaf_vector[j].communication_cost +
                       cost_graph.extra_node_costs[i][j]);
        m_np.push_back(strategies->leaf_vector[j].memory_cost);
      }
    }
  }

  // Serialize special edges that forces a alias pair have the same sharding
  // spec
  std::vector<int> A_np;
  std::vector<double> v_np;
  for (const auto& pair : alias_set) {
    const StrategyVector* src_strategies = leaf_strategies[pair.first];
    const StrategyVector* dst_strategies = leaf_strategies[pair.second];

    Matrix raw_cost(src_strategies->leaf_vector.size(),
                    dst_strategies->leaf_vector.size());
    bool compatible = false;
    for (size_t i = 0; i < src_strategies->leaf_vector.size(); ++i) {
      for (size_t j = 0; j < dst_strategies->leaf_vector.size(); ++j) {
        if (src_strategies->leaf_vector[i].output_sharding ==
            dst_strategies->leaf_vector[j].output_sharding) {
          compatible = true;
          raw_cost(i, j) = 0.0;
        } else {
          raw_cost(i, j) = 1.0;
        }
      }
    }
    CHECK(compatible) << "Incompatible alias pairs: "
                      << "(" << src_strategies->instruction_id << ", "
                      << dst_strategies->instruction_id << ")";

    int idx_a = pair.first;
    int idx_b = pair.second;
    std::vector<int> row_indices;
    std::vector<int> col_indices;

    if (s_follow_np[idx_a] >= 0) {
      row_indices = cost_graph.reindexing_vector.at(idx_a);
      idx_a = s_follow_np[idx_a];
    } else {
      row_indices.assign(s_len_np[idx_a], 0);
      std::iota(row_indices.begin(), row_indices.end(), 0);
    }

    if (s_follow_np[idx_b] >= 0) {
      col_indices = cost_graph.reindexing_vector.at(idx_b);
      idx_b = s_follow_np[idx_b];
    } else {
      col_indices.assign(s_len_np[idx_b], 0);
      std::iota(col_indices.begin(), col_indices.end(), 0);
    }

    CHECK_EQ(s_len_np[idx_a], row_indices.size());
    CHECK_EQ(s_len_np[idx_b], col_indices.size());

    A_np.push_back(idx_a);
    A_np.push_back(idx_b);
    for (int i : row_indices) {
      for (int j : col_indices) {
        v_np.push_back(raw_cost(i, j));
      }
    }
  }

  // Serialize liveness_set
  auto filter_func = [&instructions](size_t i) {
    HloOpcode opcode = instructions[i]->opcode();
    if (opcode == HloOpcode::kDot || opcode == HloOpcode::kConvolution) {
      return true;
    } else {
      return false;
    }
  };

  std::vector<int> L_np;
  std::vector<std::vector<int>> liveness_set_indices(N);
  for (size_t i = 0; i < N; ++i) {
    if (filter_func(leaf_strategies[i]->instruction_id)) {
      std::vector<int>& current_liveness_set_indices = liveness_set_indices[i];
      std::function<void(const StrategyVector*)> traverse_live_instructions;
      traverse_live_instructions = [&](const StrategyVector* strategies) {
        if (strategies->is_tuple) {
          for (const auto& child : strategies->childs) {
            traverse_live_instructions(child.get());
          }
        } else {
          current_liveness_set_indices.push_back(strategies->id);
        }
      };
      for (const HloValue* value :
           liveness_set[leaf_strategies[i]->instruction_id]) {
        traverse_live_instructions(strategy_map.at(value->instruction()).get());
      }
      L_np.push_back(current_liveness_set_indices.size());
    } else {
      L_np.push_back(0);
    }
  }

  for (const auto& indices : liveness_set_indices) {
    L_np.insert(L_np.end(), indices.begin(), indices.end());
  }

  // Call the solver function in python
  size_t num_edges = E_np.size() / 2;
  std::vector<int64_t> s_val, e_val;
  double objective;

  PyGILState_STATE gstate = PyGILState_Ensure();
  {
    py::object submodule =
        py::module_::import("alpa.shard_parallel.auto_sharding");
    py::object call_solver_serialized_args =
        submodule.attr("call_solver_serialized_args");
    py::object ret = call_solver_serialized_args(
        N, M,
        py::array(s_len_np.size(), s_len_np.data()),  // TODO: avoid this copy
        py::array(s_follow_np.size(), s_follow_np.data()),
        py::array(E_np.size(), E_np.data()),
        py::array(A_np.size(), A_np.data()),
        py::array(L_np.size(), L_np.data()),
        py::array(c_np.size(), c_np.data()),
        py::array(d_np.size(), d_np.data()),
        py::array(m_np.size(), m_np.data()),
        py::array(r_np.size(), r_np.data()),
        py::array(v_np.size(), v_np.data()));
    if (ret.is_none()) {
      PyGILState_Release(gstate);
      exit(-1);
    }
    py::tuple tuple_ret = ret;

    py::object s_val_obj = tuple_ret[0], e_val_obj = tuple_ret[1];
    objective = py::cast<double>(tuple_ret[2]);
    py::array_t<int> s_val_array = s_val_obj, e_val_array = e_val_obj;
    auto s_val_unckecked = s_val_array.unchecked<1>();
    auto e_val_unckecked = e_val_array.unchecked<1>();
    for (size_t i = 0; i < N; ++i) {
      s_val.push_back(s_val_unckecked(i));
    }
    for (size_t i = 0; i < num_edges; ++i) {
      e_val.push_back(e_val_unckecked(i));
    }
  }
  PyGILState_Release(gstate);

  return std::make_tuple(std::move(s_val), std::move(e_val), objective);
}

// Set the HloSharding for all instructions according to the ILP solution.
void SetHloSharding(const HloInstructionSequence& sequence,
                    const StrategyMap& strategy_map,
                    const CostGraph& cost_graph,
                    const std::vector<int64_t>& s_val,
                    const ClusterEnvironment& cluster_env,
                    const AutoShardingSolverOption& solver_option) {
  // Set the HloSharding for every instruction
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;

  for (HloInstruction* inst : instructions) {
    if (inst->has_sharding()) {  // its sharding has been forcibly set
      continue;
    }
    auto iter = strategy_map.find(inst);
    if (iter == strategy_map.end()) {
      continue;
    }

    const StrategyVector* strategies = iter->second.get();
    if (strategies->is_tuple) {
      const Shape& out_shape = inst->shape();
      ShapeTree<HloSharding> tuple_sharding(out_shape, Undefined());
      std::vector<HloSharding> flattened_shardings;

      if (inst->opcode() == HloOpcode::kTuple || IsCustomCallMarker(inst)) {
        // Directly copy operands' shardings to output.
        //
        // This special handling is required after GenerateReduceScatter.
        // In GenerateReduceScatter, we may forcibly set the sharding of one
        // operand without correctly updating the sharding in `strategies`
        // (because this update is very hard to do). This leads to inconsistency
        // between one operand's sharding and the sharding in
        // `strategies->childs`, so we should read the shardings from operands
        // instead of `strategies->childs`.
        std::function<void(const HloInstruction*)> get_flattened_shardings;
        get_flattened_shardings = [&](const HloInstruction* cur) {
          for (const HloInstruction* operand : cur->operands()) {
            if (operand->shape().IsTuple()) {
              get_flattened_shardings(operand);
            } else {
              CHECK(operand->has_sharding());
              flattened_shardings.push_back(operand->sharding());
            }
          }
        };
        get_flattened_shardings(inst);
      } else {
        // Otherwise, copy shardings from `strategies.childs`
        std::function<void(const StrategyVector*)> get_flattened_shardings;
        get_flattened_shardings = [&](const StrategyVector* cur) {
          if (cur->is_tuple) {
            for (const auto& child : strategies->childs) {
              get_flattened_shardings(child.get());
            }
          } else {
            int node_idx = cur->id;
            int stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
            flattened_shardings.push_back(
                cur->leaf_vector[stra_idx].output_sharding);
          }
        };
        get_flattened_shardings(strategies);
      }

      // Create Tuple HloSharding
      // TODO(lmzheng, zhuohan): Rewrite this with ShapeTree and support
      // nested tuple in the output.
      CHECK_EQ(tuple_sharding.leaf_count(), flattened_shardings.size());
      size_t i = 0;
      for (auto& leaf : tuple_sharding.leaves()) {
        leaf.second = flattened_shardings[i++];
      }
      inst->set_sharding(HloSharding::Tuple(tuple_sharding));
    } else {
      const HloSharding& sharding_spec =
          GetShardingStrategy(inst).output_sharding;
      if (IsUndefined(sharding_spec)) {
        continue;
      }
      inst->set_sharding(sharding_spec);
    }
  }

  // Post process: fix some corner cases.
  ReshardingCache resharding_cache_entity;
  ReshardingCache* resharding_cache = &resharding_cache_entity;

  for (HloInstruction* inst : sequence.instructions()) {
    // For some dot instructions and resharding cases, our formulation thinks
    // they are valid. But the the spmd partitioner cannot infer the correct
    // dot algorithms or resharding algorithm from the input/output sharding.
    // It then generates bad fallback code.
    // Here we insert some extra annotated identity instructions to help the
    // spmd partitioner generate correct code.

    if (inst->opcode() == HloOpcode::kDot) {
      const ShardingStrategy& stra = GetShardingStrategy(inst);
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const HloSharding& lhs_sharding = lhs->sharding();
      const HloSharding& rhs_sharding = rhs->sharding();
      const DotDimensionNumbers& dot_dnums = inst->dot_dimension_numbers();
      const auto& lhs_con_dims = dot_dnums.lhs_contracting_dimensions();
      const auto& rhs_con_dims = dot_dnums.rhs_contracting_dimensions();

      const auto& lhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDim(lhs->shape(), lhs_sharding);
      const auto& rhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDim(rhs->shape(), rhs_sharding);

      if (stra.name.find("allreduce") != std::string::npos &&
          lhs_tensor_dim_to_mesh_dim[lhs_con_dims[0]] == -1 &&
          rhs_tensor_dim_to_mesh_dim[rhs_con_dims[0]] == -1) {
        ;  // Allow duplicatd dot computation in this case to reduce
           // communication
      } else {
        FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0],
                                    device_mesh, resharding_cache);
        FixMixedMeshShapeResharding(inst, 1, stra.input_shardings[1],
                                    device_mesh, resharding_cache);
      }
    } else if (inst->opcode() == HloOpcode::kConvolution) {
      const ShardingStrategy& stra = GetShardingStrategy(inst);
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const HloSharding& lhs_sharding = lhs->sharding();
      const HloSharding& rhs_sharding = rhs->sharding();
      const ConvolutionDimensionNumbers& conv_dnums =
          inst->convolution_dimension_numbers();
      const int lhs_in_channel_dim = conv_dnums.input_feature_dimension();
      const int rhs_in_channel_dim =
          conv_dnums.kernel_input_feature_dimension();

      const auto& lhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDim(lhs->shape(), lhs_sharding);
      const auto& rhs_tensor_dim_to_mesh_dim =
          cluster_env.GetTensorDimToMeshDim(rhs->shape(), rhs_sharding);

      if (stra.name.find("allreduce") != std::string::npos &&
          lhs_tensor_dim_to_mesh_dim[lhs_in_channel_dim] == -1 &&
          rhs_tensor_dim_to_mesh_dim[rhs_in_channel_dim] == -1) {
        ;  // Allow duplicatd conv computation in this case to reduce
           // communication
      } else {
        FixMixedMeshShapeResharding(inst, 0, stra.input_shardings[0],
                                    device_mesh, resharding_cache);
        FixMixedMeshShapeResharding(inst, 1, stra.input_shardings[1],
                                    device_mesh, resharding_cache);
      }
    } else {
      if (!strategy_map.at(inst).get()->is_tuple) {
        const ShardingStrategy& stra = GetShardingStrategy(inst);
        if (!stra.input_shardings.empty()) {
          CHECK_EQ(stra.input_shardings.size(), inst->operand_count());
          for (size_t i = 0; i < stra.input_shardings.size(); ++i) {
            FixMixedMeshShapeResharding(inst, i, stra.input_shardings[i],
                                        device_mesh, resharding_cache);
          }
        }
      }
    }
  }

  for (HloInstruction* inst : sequence.instructions()) {
    if (inst->opcode() == HloOpcode::kIota) {
      if (inst->sharding().IsReplicated()) {
        // For fully replicated iota, leave its sharding annotation to the
        // ShardingPropagation pass, which can typically do a better job.
        inst->clear_sharding();
      }
    }
  }
}

// Print liveness set for debugging.
std::string PrintLivenessSet(const LivenessSet& liveness_set) {
  std::ostringstream os;
  os << "Liveness Set" << std::endl;
  for (size_t i = 0; i < liveness_set.size(); ++i) {
    std::vector<std::string> names;
    for (const HloValue* value : liveness_set[i]) {
      names.push_back(value->instruction()->name() + value->index().ToString());
    }
    std::sort(names.begin(), names.end());

    std::string line;
    for (const std::string& name : names) {
      absl::StrAppendFormat(&line, "%s, ", name);
    }
    os << "Time " << i << ": " << line << std::endl;
  }
  return os.str();
}

// Print strategy map for debugging.
std::string PrintStrategyVector(const StrategyVector* strategies,
                                size_t indention = 0) {
  std::ostringstream os;
  if (strategies->is_tuple) {
    for (size_t i = 0; i < strategies->childs.size(); ++i) {
      os << std::string(indention, ' ') << "Tuple element #" << i << ":\n";
      os << PrintStrategyVector(strategies->childs[i].get(), indention + 2);
    }
  } else {
    for (const auto& strategy : strategies->leaf_vector) {
      os << std::string(indention, ' ') << "Strategy " << strategy.name << ", "
         << strategy.compute_cost << ", " << strategy.communication_cost << ", "
         << strategy.memory_cost << ", {";

      for (const auto& cost_vector : strategy.resharding_costs) {
        os << "[";
        for (double cost : cost_vector) {
          os << cost << ", ";
        }
        os << "], ";
      }
      os << "}\n";
    }
  }
  return os.str();
}

// Print strategy map for debugging.
std::string PrintStrategyMap(const StrategyMap& strategy_map,
                             const HloInstructionSequence& sequence) {
  std::ostringstream os;
  os << "Strategy Map" << std::endl;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << "Instruction " << i << ": " << instructions[i]->ToString() << "\n";
    os << PrintStrategyVector(strategy_map.at(instructions[i]).get());
  }
  return os.str();
}

// Print the choosen sharding strategy for debugging.
std::string PrintAutoShardingSolution(
    const HloInstructionSequence& sequence, const LivenessSet& liveness_set,
    const InstructionDepthMap depth_map, const StrategyMap& strategy_map,
    const LeafStrategies& leaf_strategies, const CostGraph& cost_graph,
    const std::vector<int64_t>& s_val, double objective) {
  std::ostringstream os;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  size_t N = leaf_strategies.size();

  absl::flat_hash_map<const HloInstruction*, int> tuple_elem_counter;

  // Print the choosen strategy
  os << "=== Auto sharding strategy ===\n";
  for (size_t i = 0; i < N; ++i) {
    int stra_idx = cost_graph.RemapIndex(i, s_val[i]);
    const ShardingStrategy& stra = leaf_strategies[i]->leaf_vector[stra_idx];
    const HloInstruction* ins =
        instructions[leaf_strategies[i]->instruction_id];

    int ct = tuple_elem_counter[ins]++;
    if (ct == 0) {
      os << i << " " << ins->ToString(HloPrintOptions::ShortParsable()) << "  ";
    } else {
      // Only print tuple once
      os << i << " tuple." << ct << " "
         << ins->shape().tuple_shapes(ct).ToString() << "  ";
    }

    os << " depth: " << depth_map.at(ins)
       << " max_num_tiles: " << MaxNumTiles(strategy_map, ins) << " ";

    if (cost_graph.follow_idx[i] < 0) {
      os << stra.name << " ";
    } else {
      os << stra.name << " follow " << cost_graph.follow_idx[i] << " ";
    }
    if (stra.communication_cost > 0) {
      os << "comm cost: " << std::fixed << std::setprecision(2)
         << stra.communication_cost;
    }
    os << "\n";
  }

  // Print edge costs
  for (size_t a = 0; a < N; ++a) {
    for (auto b : cost_graph.adjacency.at(a)) {
      if (a >= b) {
        continue;
      }

      double cost = cost_graph.edge_costs.at({a, b})(s_val[a], s_val[b]);
      if (cost > 1e-6) {
        os << "- edge cost (" << a << ", " << b << "): " << std::fixed
           << std::setprecision(3) << cost << "\n";
      }
    }
  }

  os << "objective : " << std::fixed << std::setprecision(2) << objective
     << "\n";

  return os.str();
}

// 1. Disable mixed mesh shape if the batch dim is not divisible by the
// number of devices.
// 2. Disable force_batch_dim_to_mesh_dim if the batch dim is 1. In this case,
// the batch dim analysis can be wrong because the batch dim might be dropped.
void DisableIncompatibleMixedMeshShapeAndForceBatchDim(
    const InstructionBatchDimMap& batch_dim_map, int num_devices,
    AutoShardingSolverOption& solver_option) {
  int64_t batch_size = (((int64_t)1) << 31) - 1;
  for (auto iter : batch_dim_map) {
    int64_t tmp_batch_size;
    if (iter.first->shape().IsTuple()) {
      tmp_batch_size =
          iter.first->shape().tuple_shapes(0).dimensions(iter.second);
    } else {
      tmp_batch_size = iter.first->shape().dimensions(iter.second);
    }
    batch_size = std::min(batch_size, tmp_batch_size);
  }

  if (batch_size % num_devices != 0) {
    if (solver_option.allow_mixed_mesh_shape) {
      solver_option.allow_mixed_mesh_shape = false;
      LOG(WARNING)
          << "Mixed mesh shape is disabled due to indivisible batch size.";
    }
  }

  if (batch_size == 1) {
    solver_option.force_batch_dim_to_mesh_dim = -1;
  }
}

StatusOr<bool> AutoSharding::Run(HloModule* module) {
  if (!pass_context::GetBool("auto_sharding::enable", true)) {
    return false;
  }

  // ----- Read options of this pass -----
  AutoShardingSolverOption solver_option;
  solver_option.override_all_gather_cost = false;
  solver_option.override_all_reduce_cost = false;
  solver_option.override_reduce_scatter_cost = false;
  solver_option.override_all_to_all_cost = false;

  if (pass_context::GetBool("auto_sharding::force_all_gather_cost", false)) {
    solver_option.override_all_gather_cost = true;
    solver_option.all_gather_cost =
        pass_context::GetDouble("auto_sharding::all_gather_cost");
  }
  if (pass_context::GetBool("auto_sharding::force_all_to_all_cost", false)) {
    solver_option.override_all_to_all_cost = true;
    solver_option.all_to_all_cost =
        pass_context::GetDouble("auto_sharding::all_to_all_cost");
  }
  solver_option.force_batch_dim_to_mesh_dim =
      pass_context::GetInt("auto_sharding::force_batch_dim_to_mesh_dim", -1);
  solver_option.allow_replicated_parameters =
      pass_context::GetBool("auto_sharding::allow_replicated_parameters", true);
  solver_option.prefer_reduce_scatter =
      pass_context::GetBool("auto_sharding::prefer_reduce_scatter", false);
  solver_option.reduce_scatter_grad_acc_friendly = pass_context::GetBool(
      "auto_sharding::reduce_scatter_grad_acc_friendly", false);
  solver_option.reduce_scatter_aggressive_partition = pass_context::GetBool(
      "auto_sharding::reduce_scatter_aggressive_partition", false);
  solver_option.batch_matmul_always_split_batch = pass_context::GetBool(
      "auto_sharding::batch_matmul_always_split_batch", false);
  solver_option.allow_recompute_heavy_op =
      pass_context::GetBool("auto_sharding::allow_recompute_heavy_op", false);
  solver_option.allow_mixed_mesh_shape =
      pass_context::GetBool("auto_sharding::allow_mixed_mesh_shape", false);
  solver_option.grad_acc_num_micro_batches =
      pass_context::GetInt("auto_sharding::grad_acc_num_micro_batches", 1);
  solver_option.load_solution_vector =
      pass_context::GetBool("auto_sharding::load_solution_vector", false);
  solver_option.force_simple_heuristic =
      pass_context::GetString("auto_sharding::force_simple_heuristic", "");

  // ----- Read parameters of device mesh -----
  Array<int64_t> device_mesh(
      pass_context::GetIntVector("auto_sharding::device_mesh_shape"));
  device_mesh.SetValues(
      pass_context::GetIntVector("auto_sharding::device_mesh_ids"));
  ProfilingResult prof_result(
      pass_context::GetPyObject("auto_sharding::device_mesh_prof_result"));
  ClusterEnvironment cluster_env(
      device_mesh,
      pass_context::GetDoubleVector("auto_sharding::device_mesh_alpha"),
      pass_context::GetDoubleVector("auto_sharding::device_mesh_beta"),
      prof_result, solver_option);

  // std::cerr << "===== Enter AutoSharding =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  // ----- Pre-process to normalize the dot dimensions -----
  TF_ASSIGN_OR_RETURN(bool changed, NormalizeDotDimension(module));

  // ----- Get a sequential schedule and do liveness analysis -----
  auto size_fn = [](const BufferValue& buffer) {
    return GetBytes(buffer.shape());
  };
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      ScheduleModule(module, size_fn,
                                     ComputationSchedulerToModuleScheduler(
                                         DFSMemoryScheduler)));
  const HloComputation* entry_computation = module->entry_computation();
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
      HloAliasAnalysis::Run(module).ConsumeValueOrDie();
  AliasMap alias_map =
      BuildAliasMap(module, alias_analysis->dataflow_analysis());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(schedule, *alias_analysis, entry_computation));
  absl::flat_hash_map<const HloValue*, HloLiveRange::TimeBound>&
      buffer_live_ranges = hlo_live_range->buffer_live_ranges();
  LivenessSet liveness_set(hlo_live_range->schedule_end_time() + 1);
  for (const auto& iter : buffer_live_ranges) {
    for (int64_t i = iter.second.start; i <= iter.second.end; ++i) {
      liveness_set[i].push_back(iter.first);
    }
  }

  if (solver_option.force_simple_heuristic != "") {
    AnnotateShardingWithSimpleHeuristic(
        module, solver_option.force_simple_heuristic, alias_map, cluster_env);
    return true;
  }

  // ----- Analyze the batch dim -----
  const HloInstructionSequence& sequence =
      hlo_live_range->flattened_instruction_sequence();
  InstructionBatchDimMap batch_dim_map;
  batch_dim_map = BuildInstructionBatchDimMap(sequence);
  if (solver_option.force_batch_dim_to_mesh_dim >= 0) {
    DisableIncompatibleMixedMeshShapeAndForceBatchDim(
        batch_dim_map, device_mesh.num_elements(), solver_option);
  }

  // ----- Analyze depth -----
  InstructionDepthMap ins_depth_map;
  ins_depth_map = BuildInstructionDepthMap(sequence, batch_dim_map);

  // ----- Build strategies and costs -----
  StrategyMap strategy_map;
  LeafStrategies leaf_strategies;
  AssociativeDotPairs associative_dot_pairs;

  TF_ASSIGN_OR_RETURN(
      std::tie(strategy_map, leaf_strategies, associative_dot_pairs),
      BuildStrategyAndCost(sequence, ins_depth_map, batch_dim_map, alias_map,
                           cluster_env, solver_option));
  AliasSet alias_set =
      BuildAliasSet(module, alias_analysis->dataflow_analysis(), strategy_map);
  // std::cerr << PrintStrategyMap(strategy_map, sequence);

  // ----- Build cost graph and merge unimporant nodes -----
  CostGraph cost_graph(leaf_strategies, associative_dot_pairs);
  cost_graph.Simplify();

  // ----- Call the ILP solver -----
  std::vector<int64_t> s_val, e_val;
  double objective = -1.0;
  if (!solver_option.load_solution_vector) {
    std::tie(s_val, e_val, objective) =
        CallSolver(sequence, liveness_set, strategy_map, leaf_strategies,
                   cost_graph, alias_set);
  } else {
    s_val = pass_context::GetIntVector("auto_sharding::solution_vector");
  }

  if (pass_context::GetBool("auto_sharding::print_strategy", false)) {
    std::cerr << PrintAutoShardingSolution(
        sequence, liveness_set, ins_depth_map, strategy_map, leaf_strategies,
        cost_graph, s_val, objective);
  }

  // ----- Substitute all-reduce with reduce-scatter -----
  if (solver_option.prefer_reduce_scatter) {
    GenerateReduceScatter(sequence, alias_map, ins_depth_map, strategy_map,
                          cost_graph, s_val, cluster_env, solver_option);
  }

  // ----- Set sharding for all instructions -----
  SetHloSharding(sequence, strategy_map, cost_graph, s_val, cluster_env,
                 solver_option);

  // std::cerr << "===== Exit AutoSharding =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

}  // namespace spmd
}  // namespace xla
