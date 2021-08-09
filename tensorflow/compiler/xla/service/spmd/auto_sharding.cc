#include "tensorflow/compiler/xla/service/spmd/auto_sharding.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"

#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace spmd {

// Create a HloSharding that tiles some tensor dims on some device mesh dims.
HloSharding Tile(const Shape& shape, const std::vector<int64> tensor_dims,
                 const std::vector<int64> mesh_dims,
                 const ClusterEnvironment& cluster_env) {
  CHECK_EQ(tensor_dims.size(), mesh_dims.size());
  CHECK(shape.IsArray());

  std::vector<int64> tile_assignment_dimensions(shape.rank(), 1);

  // Split on certain mesh dimensions
  int64 split_prod = 1;
  for (size_t i = 0; i < tensor_dims.size(); ++i) {
    tile_assignment_dimensions[tensor_dims[i]] =
        cluster_env.device_mesh.dim(mesh_dims[i]);
    split_prod *= cluster_env.device_mesh.dim(mesh_dims[i]);
  }

  // Replicate on reminding mesh dimensions
  bool replicate_on_last_tile_dim = false;
  if (split_prod < cluster_env.total_devices) {
    tile_assignment_dimensions.push_back(cluster_env.total_devices /
                                         split_prod);
    replicate_on_last_tile_dim = true;
  }

  // Map device ids from device_mesh to tile_assignment_devices
  std::vector<int64> tile_assignment_devices;
  tile_assignment_devices.reserve(cluster_env.total_devices);

  std::vector<int64> tmp_indices(cluster_env.device_mesh.num_dimensions(), 0);
  std::function<void(int64, std::vector<int64>)>
      generate_tile_assignment_devices;
  generate_tile_assignment_devices = [&](int64 tensor_dim,
                                         std::vector<int64> mesh_indices) {
    if (tensor_dim == shape.rank() - 1) {
      AppendFlattenElements(&tile_assignment_devices, cluster_env.device_mesh,
                            mesh_indices, -1, tmp_indices);
    } else {
      int64 next_tensor_dim = tensor_dim + 1;
      int64 next_mesh_dim = -1;

      int64 index = GetIndex(tensor_dims, next_tensor_dim);
      if (index >= 0) {
        next_mesh_dim = mesh_dims[index];
      }

      for (int64 i = 0; i < tile_assignment_dimensions[next_tensor_dim]; ++i) {
        if (next_mesh_dim != -1) {
          mesh_indices[next_mesh_dim] = i;
        }
        generate_tile_assignment_devices(next_tensor_dim, mesh_indices);
      }
    }
  };

  std::vector<int64> mesh_indices(cluster_env.device_mesh.num_dimensions(), -1);
  generate_tile_assignment_devices(-1, mesh_indices);

  // Make HloSharding
  Array<int64> tile_assignment(tile_assignment_dimensions);
  tile_assignment.SetValues(tile_assignment_devices);

  return replicate_on_last_tile_dim ? HloSharding::PartialTile(std::move(tile_assignment))
                                    : HloSharding::Tile(std::move(tile_assignment));
}

// Compute the resharding cost vector from multiple possible strategies
// to a desired sharding spec
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

std::vector<double> FollowInsCostVector(int64 source_len, int64 index) {
  std::vector<double> ret(source_len, INFINITY_COST);
  ret[index] = 0;
  return ret;
}

// Factory functions for StrategyVector
std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, LeafStrategies& leaf_strategies) {
  std::unique_ptr<StrategyVector> strategies =
      absl::make_unique<StrategyVector>();
  strategies->is_tuple = false;
  strategies->id = leaf_strategies.size();
  leaf_strategies.push_back(strategies.get());
  strategies->instruction_id = instruction_id;
  return std::move(strategies);
}

std::unique_ptr<StrategyVector> CreateTupleStrategyVector(
    size_t instruction_id) {
  std::unique_ptr<StrategyVector> strategies =
      absl::make_unique<StrategyVector>();
  strategies->is_tuple = true;
  strategies->id = -1;
  strategies->instruction_id = instruction_id;
  return std::move(strategies);
}

// TODO(lmzheng,zhuohan): merge this into CreateLeafStrategyVector
void SetInNodesWithInstruction(std::unique_ptr<StrategyVector>& strategies,
                               const HloInstruction* ins,
                               const StrategyMap& strategy_map) {
  for (int64 i = 0; i < ins->operand_count(); ++i) {
    strategies->in_nodes.push_back(strategy_map.at(ins->operand(i)).get());
  }
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
    strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
    strategies->in_nodes.push_back(src_strategies);
    strategies->following = src_strategies;
    strategies->leaf_vector.reserve(src_strategies->leaf_vector.size());
    for (int64 sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
      HloSharding output_spec =
          src_strategies->leaf_vector[sid].output_sharding;
      std::string name = SimpleToString(output_spec);
      double compute_cost = 0, communication_cost = 0;
      double memory_cost =
          have_memory_cost ? GetBytes(shape) / output_spec.NumTiles() : 0;
      std::vector<std::vector<double>> resharding_costs = {
          FollowInsCostVector(src_strategies->leaf_vector.size(), sid)};
      strategies->leaf_vector.push_back(
          ShardingStrategy({name, output_spec, compute_cost, communication_cost,
                            memory_cost, resharding_costs}));
    }
  }
  return std::move(strategies);
}

// Build possible sharding strategies and their costs for all instructions
std::tuple<StrategyMap, LeafStrategies, AssociativeDotPairs> BuildStrategyAndCost(
    const HloInstructionSequence& sequence,
    const InstructionDepthMap& depth_map,
    const AliasMap& alias_map,
    const ClusterEnvironment& cluster_env,
    const AutoShardingSolverOption& solver_option) {
  const Array<int64>& device_mesh = cluster_env.device_mesh;
  StrategyMap strategy_map;
  LeafStrategies leaf_strategies;
  AssociativeDotPairs associative_dot_pairs;
  absl::flat_hash_set<const HloInstruction*> undefined_set;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Analyze the batch dim if we want to forcely use data-parallel
  InstructionBatchDimMap batch_dim_map;
  if (solver_option.force_batch_dim_to_mesh_dim >= 0) {
    batch_dim_map = BuildInstructionBatchDimMap(sequence);
  }

  // Gather all output values
  absl::flat_hash_set<const HloInstruction*> output_set;
  for (size_t i = 0; i < instructions.back()->operand_count(); ++i) {
    output_set.insert(instructions.back()->operand(i));
  }

  // Register strategies and their costs for each instruction.
  for (size_t instruction_id = 0; instruction_id < instructions.size();
       ++instruction_id) {
    const HloInstruction* ins = instructions[instruction_id];
    std::unique_ptr<StrategyVector> strategies;
    switch (ins->opcode()) {
      case HloOpcode::kParameter: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);

        // Split one dim
        for (int64 i = 0; i < ins->shape().rank(); ++i) {
          for (int64 j = 0; j < device_mesh.num_dimensions(); ++j) {
            if (device_mesh.dim(j) == 1 ||
                ins->shape().dimensions(i) < device_mesh.dim(j)) {
              continue;
            }

            std::string name = absl::StrFormat("S%d @ %d", i, j);
            HloSharding output_spec = Tile(ins->shape(), {i}, {j}, cluster_env);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec.NumTiles();
            strategies->leaf_vector.push_back(
                ShardingStrategy({name,
                                  output_spec,
                                  compute_cost,
                                  communication_cost,
                                  memory_cost,
                                  {}}));
          }
        }

        // Replicate
        strategies->leaf_vector.push_back(ShardingStrategy(
            {"R", HloSharding::Replicate(), 2, 0, GetBytes(ins->shape()), {}}));
        break;
      }
      case HloOpcode::kConstant: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        strategies->leaf_vector.push_back(ShardingStrategy(
            {"R", HloSharding::Replicate(), 0, 0, GetBytes(ins->shape()), {}}));
        break;
      }
      case HloOpcode::kBroadcast: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.count(operand)) {
          break;
        }

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64 sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec = BroadcastSharding(
              src_strategies->leaf_vector[sid].output_sharding, ins->shape(),
              ins->dimensions());

          std::string name = SimpleToString(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name,
               output_spec,
               compute_cost,
               communication_cost,
               memory_cost,
               {FollowInsCostVector(src_strategies->leaf_vector.size(),
                                    sid)}}));
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
          for (int64 i = 0; i < ins->shape().rank(); ++i) {
            for (int64 j = 0; j < device_mesh.num_dimensions(); ++j) {
              if (device_mesh.dim(j) == 1 ||
                  ins->shape().dimensions(i) < device_mesh.dim(j)) {
                continue;
              }

              std::string name = absl::StrFormat("S%d @ %d", i, j);
              HloSharding output_spec =
                  Tile(ins->shape(), {i}, {j}, cluster_env);
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
                                        0.0)}}));
            }
          }

          // Replicate
          strategies->leaf_vector.push_back(ShardingStrategy(
              {"R",
               HloSharding::Replicate(),
               2,
               0,
               GetBytes(ins->shape()),
               {std::vector<double>(src_strategies->leaf_vector.size(),
                                    0.0)}}));
        }

        break;
      }
      case HloOpcode::kReshape: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.count(operand)) {
          break;
        }

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64 sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          absl::optional<HloSharding> output_spec =
              hlo_sharding_util::ReshapeSharding(
                  operand->shape(), ins->shape(),
                  src_strategies->leaf_vector[sid].output_sharding);

          if (!output_spec.has_value()) {
            continue;
          }

          std::string name = SimpleToString(*output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec->NumTiles();
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name,
               *output_spec,
               compute_cost,
               communication_cost,
               memory_cost,
               {FollowInsCostVector(src_strategies->leaf_vector.size(),
                                    sid)}}));
        }
        break;
      }
      case HloOpcode::kTranspose: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.count(operand)) {
          break;
        }

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64 sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec = hlo_sharding_util::TransposeSharding(
              src_strategies->leaf_vector[sid].output_sharding,
              ins->dimensions());

          std::string name = SimpleToString(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name,
               output_spec,
               compute_cost,
               communication_cost,
               memory_cost,
               {FollowInsCostVector(src_strategies->leaf_vector.size(),
                                    sid)}}));
        }
        break;
      }
      case HloOpcode::kPad:
      case HloOpcode::kSlice:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kConcatenate:  // TODO(lmzheng): revisit concatenate
      case HloOpcode::kDynamicUpdateSlice: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        const HloInstruction* operand = ins->operand(0);
        if (undefined_set.count(operand)) {
          break;
        }

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64 sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          absl::optional<HloSharding> output_spec = PropagateDimwiseSharding(
              src_strategies->leaf_vector[sid].output_sharding,
              operand->shape(), ins->shape());

          if (!output_spec.has_value()) {
            continue;
          }

          std::string name = SimpleToString(*output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec->NumTiles();
          std::vector<std::vector<double>> resharding_costs;
          resharding_costs.push_back(
              FollowInsCostVector(src_strategies->leaf_vector.size(), sid));
          for (int64 k = 1; k < ins->operand_count(); ++k) {
            resharding_costs.push_back(ReshardingCostVector(
                strategy_map.at(ins->operand(k)).get(),
                ins->operand(k)->shape(), *output_spec, cluster_env));
          }
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name, *output_spec, compute_cost, communication_cost,
               memory_cost, resharding_costs}));
        }
        break;
      }
      // Unary elementwise operations.
      case HloOpcode::kAbs:
      case HloOpcode::kRoundNearestAfz:
      case HloOpcode::kCeil:
      case HloOpcode::kClz:
      case HloOpcode::kConvert:
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
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        // Follow the operand with the max depth
        int64 follow_idx = -1;
        int64 max_depth = -1 << 30;
        for (int64 i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          if (!undefined_set.count(operand) && depth_map.at(operand) > max_depth) {
            follow_idx = i;
            max_depth = depth_map.at(operand);
          }
          // If an alias constraint is set, always follow its alias source.
          auto it = alias_map.find(ins);
          if (it != alias_map.end() && it->second == operand) {
            follow_idx = i;
            break;
          }
        }
        CHECK_GE(follow_idx, 0);

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(
          ins->operand(follow_idx)).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (int64 sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          HloSharding output_spec =
              src_strategies->leaf_vector[sid].output_sharding;

          std::string name = SimpleToString(output_spec);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          std::vector<std::vector<double>> resharding_costs;
          for (int64 k = 0; k < ins->operand_count(); ++k) {
            if (k == follow_idx) {
              resharding_costs.push_back(
                  FollowInsCostVector(src_strategies->leaf_vector.size(), sid));
            } else {
              resharding_costs.push_back(ReshardingCostVector(
                  strategy_map.at(ins->operand(k)).get(),
                  ins->operand(k)->shape(), output_spec, cluster_env));
            }
          }

          strategies->leaf_vector.push_back(ShardingStrategy(
              {name, output_spec, compute_cost, communication_cost, memory_cost,
               resharding_costs}));
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
              {strategy_map.at(ins->operand(0)).get(), strategy_map.at(ins->operand(1)).get()});
          }
        }
        break;
      }
      case HloOpcode::kReduce: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        const HloInstruction* operand = ins->operand(0);
        const HloInstruction* unit = ins->operand(1);
        if (undefined_set.count(operand)) {
          break;
        }

        // Map old dims to new dim
        const auto& dimensions = ins->dimensions();
        std::vector<int64> old_dim_to_new_dim;
        old_dim_to_new_dim.reserve(operand->shape().rank());
        int64 pt = 0;
        for (int64 old_dim = 0; old_dim < operand->shape().rank(); ++old_dim) {
          if (absl::c_find(dimensions, old_dim) != dimensions.end()) {
            old_dim_to_new_dim.push_back(-1);
          } else {
            old_dim_to_new_dim.push_back(pt);
            pt += 1;
          }
        }
        CHECK_EQ(pt, ins->shape().rank());

        // Create follow strategies
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        strategies->following = src_strategies;

        for (size_t sid = 0; sid < src_strategies->leaf_vector.size(); ++sid) {
          const auto& tensor_dim_to_mesh = cluster_env.GetTensorDimToMeshDim(
              operand->shape(),
              src_strategies->leaf_vector[sid].output_sharding);

          std::vector<int64> tile_tensor_dims, tile_mesh_dims, all_reduce_dims;

          for (int64 tensor_dim = 0; tensor_dim < operand->shape().rank();
               ++tensor_dim) {
            int64 mesh_dim = tensor_dim_to_mesh[tensor_dim];
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

          HloSharding output_spec =
              Tile(ins->shape(), tile_tensor_dims, tile_mesh_dims, cluster_env);
          double compute_cost = 0, communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
          for (auto mesh_dim : all_reduce_dims) {
            communication_cost +=
                cluster_env.AllReduceCost(memory_cost, mesh_dim);
          }

          std::string name = SimpleToString(output_spec);
          if (!all_reduce_dims.empty()) {
            name += " (allreduce @ " + ToString(all_reduce_dims) + ")";
          }
          strategies->leaf_vector.push_back(ShardingStrategy(
              {name,
               output_spec,
               compute_cost,
               communication_cost,
               memory_cost,
               {FollowInsCostVector(src_strategies->leaf_vector.size(), sid),
                ReshardingCostVector(strategy_map.at(unit).get(), unit->shape(),
                                     HloSharding::Replicate(), cluster_env)}}));
        }

        break;
      }
      case HloOpcode::kDot: {
        HandleDot(strategies, leaf_strategies, strategy_map,
                  ins, instruction_id, cluster_env,
                  batch_dim_map, solver_option);
        break;
      }
      case HloOpcode::kTuple: {
        strategies = CreateTupleStrategyVector(instruction_id);
        strategies->childs.reserve(ins->operand_count());
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          strategies->childs.push_back(std::move(FollowInsStrategyVector(
              src_strategies, operand->shape(), instruction_id,
              /* have_memory_cost= */ false, leaf_strategies)));
        }
        break;
      }
      case HloOpcode::kRngGetAndUpdateState: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        strategies->leaf_vector.push_back(ShardingStrategy(
            {"R", HloSharding::Replicate(), 0, 0, GetBytes(ins->shape()), {}}));
        break;
      }
      case HloOpcode::kIota: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);

        // Fully tile the buffer to 2-d mesh
        for (int64 i = 0; i < ins->shape().rank(); ++i) {
          for (int64 j = 0; j < ins->shape().rank(); ++j) {
            if (i == j) {
              continue;
            }

            if (ins->shape().dimensions(i) < device_mesh.dim(0) ||
                ins->shape().dimensions(j) < device_mesh.dim(1)) {
              continue;
            }

            std::string name = absl::StrFormat("S{%d,%d} @ {0,1}", i, j);
            HloSharding output_spec = Tile(ins->shape(), {i, j}, {0, 1}, cluster_env);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost =
                GetBytes(ins->shape()) / output_spec.NumTiles();
            strategies->leaf_vector.push_back(
                ShardingStrategy({name,
                                  output_spec,
                                  compute_cost,
                                  communication_cost,
                                  memory_cost,
                                  {}}));
          }
        }

        // Replicate
        strategies->leaf_vector.push_back(ShardingStrategy(
            {"R", HloSharding::Replicate(), 4, 0, GetBytes(ins->shape()), {}}));
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
        if (ins->IsCustomCall("xla_pipeline_marker")) {
          const HloInstruction* operand = ins->operand(0);
          const StrategyVector* src_strategies = strategy_map.at(operand).get();
          CHECK(src_strategies->is_tuple);
          // TODO (zhuohan): The memory cost of the marker should eventually be
          // 0.
          strategies = FollowInsStrategyVector(
              src_strategies, ins->shape(), instruction_id,
              /* have_memory_cost= */ true, leaf_strategies);
        } else {
          LOG(FATAL) << "Unknown CustomCall instruction: " + ins->name();
        }
        break;
      }
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->name();
    }

    // For instructions without any registered strategies,
    // set its strategy as "undefined".
    // Its sharding spec will be annotaed later by the ShardingPropagation pass.
    if (!strategies->is_tuple && strategies->leaf_vector.empty()) {
      std::vector<std::vector<double>> resharding_costs;
      for (size_t i = 0; i < ins->operand_count(); ++i) {
        const HloInstruction* operand = ins->operand(i);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        resharding_costs.push_back(
            std::vector<double>(src_strategies->leaf_vector.size(), 0));
      }
      strategies->leaf_vector.push_back(ShardingStrategy(
          {"undefined", Undefined(), 0, 0, 0, resharding_costs}));
      undefined_set.insert(ins);
    }

    // Debug options: forcibly set the the strategy of some instructions.
    if (pass_context::GetBool("auto_sharding::force_strategy", false)) {;
      std::vector<int64> inst_indices =
          pass_context::GetIntVector("auto_sharding::force_strategy_inst_indices");
      std::vector<std::string> stra_names =
          pass_context::GetStringVector("auto_sharding::force_strategy_stra_names");
      CHECK_EQ(inst_indices.size(), stra_names.size());
      auto it = absl::c_find(inst_indices, strategies->id);

      if (it != inst_indices.end()) {
        CHECK(!strategies->is_tuple);
        std::vector<ShardingStrategy> new_leaf_vector;
        int64 idx = it - inst_indices.begin();

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

  return std::make_tuple(std::move(strategy_map), std::move(leaf_strategies),
                         std::move(associative_dot_pairs));
}

// Note for dealing with alias:
// To deal with alias, we do special process both before and after BuildStrategyAndCost.
// Because it is easier to handle elementwise instructions before BuildStrategyAndCost and
// it is easier to handle dot/conv instructions after BuildStrategyAndCost.

AliasMap BuildAliasMap(const HloModule* module,
                       const HloDataflowAnalysis& dataflow_analysis) {
  AliasMap alias_map;

  const HloInputOutputAliasConfig& alias_config =
      module->input_output_alias_config();

  HloComputation* entry = module->entry_computation();
  const std::vector<HloInstruction*>& parameter_instructions =
      entry->parameter_instructions();
  const HloInstruction* output_tuple = entry->root_instruction();

  alias_config.ForEachAlias([&](const ShapeIndex& output_index,
                                const HloInputOutputAliasConfig::Alias& alias) {
    HloInstruction* src_ins = parameter_instructions[alias.parameter_number];
    CHECK_EQ(alias.parameter_index.size(), 0) << "Do not support tuple alias";
    const HloInstruction* dst_ins =
        dataflow_analysis.GetUniqueValueAt(output_tuple, output_index)
            .instruction();

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
    const HloInstruction* src_ins =
        parameter_instructions[alias.parameter_number];
    CHECK_EQ(alias.parameter_index.size(), 0) << "Do not support tuple alias";
    const HloInstruction* dst_ins =
        dataflow_analysis.GetUniqueValueAt(output_tuple, output_index)
            .instruction();
    traverse_tuple_alias(strategy_map.at(src_ins).get(),
                         strategy_map.at(dst_ins).get());
  });

  return alias_set;
}

// A graph data structure to simplify the edge cost graph.
// It merges nodes and does path compression.
class CostGraph {
 public:
  CostGraph(const LeafStrategies& leaf_strategies,
            const AssociativeDotPairs& associative_dot_pairs) {
    node_lens.reserve(leaf_strategies.size());
    adjacency.assign(leaf_strategies.size(), absl::flat_hash_set<int>());

    // Build the cost graph
    for (const auto& strategies : leaf_strategies) {
      node_lens.push_back(strategies->leaf_vector.size());

      for (const auto& strategy : strategies->leaf_vector) {
        CHECK_EQ(strategy.resharding_costs.size(), strategies->in_nodes.size());
      }

      for (size_t i = 0; i < strategies->in_nodes.size(); ++i) {
        size_t src_idx = strategies->in_nodes[i]->id;
        size_t dst_idx = strategies->id;

        Matrix edge_cost(node_lens[src_idx], node_lens[dst_idx]);
        for (size_t k = 0; k < strategies->leaf_vector.size(); ++k) {
          const ShardingStrategy& stra = strategies->leaf_vector[k];
          for (size_t j = 0; j < stra.resharding_costs[i].size(); ++j) {
            edge_cost(j, k) = stra.resharding_costs[i][j];
          }
        }

        AddEdgeCost(src_idx, dst_idx, edge_cost);
      }

      if (strategies->following) {
        to_merge_pairs_.push_back({strategies->id, strategies->following->id});
      }
    }

    // Adjust the edge costs for dot pairs that can be optimized by AllReduceReassociate
    for (const auto& pair : associative_dot_pairs) {
      size_t src_idx = pair.first->id;
      size_t dst_idx = pair.second->id;

      if (node_lens[src_idx] != node_lens[dst_idx]) {
        continue;
      }

      Matrix edge_cost(node_lens[src_idx], node_lens[dst_idx]);
      for (size_t i = 0; i < node_lens[src_idx]; ++i) {
        if (leaf_strategies[src_idx]->leaf_vector[i].communication_cost > 0) {
          CHECK_FLOAT_EQ(leaf_strategies[src_idx]->leaf_vector[i].communication_cost,
                         leaf_strategies[dst_idx]->leaf_vector[i].communication_cost);
          edge_cost(i, i) = -leaf_strategies[src_idx]->leaf_vector[i].communication_cost;
        }
      }
      AddEdgeCost(src_idx, dst_idx, edge_cost);
    }
  }

  Matrix GetEdgeCost(int i, int j) {
    if (i <= j) {
      return edge_costs[{i, j}];
    } else {
      return edge_costs[{j, i}].Transpose();
    }
  }

  void AddEdgeCost(int i, int j, Matrix& cost) {
    if (i > j) {
      std::swap(i, j);
      cost = cost.Transpose();
    }

    if (edge_costs.count({i, j})) {
      CHECK(adjacency[i].count(j));
      CHECK(adjacency[j].count(i));
      edge_costs[{i, j}] = edge_costs[{i, j}] + cost;
    } else {
      adjacency[i].insert(j);
      adjacency[j].insert(i);
      edge_costs[{i, j}] = cost;
    }
  }

  void RemoveEdge(int i, int j) {
    if (i > j) {
      std::swap(i, j);
    }

    CHECK(adjacency[i].count(j));
    CHECK(adjacency[j].count(i));
    CHECK(edge_costs.count({i, j}));

    adjacency[i].erase(j);
    adjacency[j].erase(i);
    edge_costs.erase({i, j});
  }

  void MergeNode(int src, int dst) {
    CHECK(adjacency[src].count(dst));
    CHECK(adjacency[dst].count(src));
    CHECK(!merged_to_.count(src));
    CHECK(!merged_to_.count(dst));
    CHECK_NE(src, dst);

    Matrix edge_cost = GetEdgeCost(dst, src);

    std::vector<int> reindexing(node_lens[dst]);
    if (node_lens[dst] == node_lens[src]) {
      // Assume the orders of strategies in src and dst match
      // (i.e. i-th strategy in src follows i-th strategy in dst).
      // This is true in most cases because of how we create the
      // following strategies.
      std::iota(reindexing.begin(), reindexing.end(), 0);
    } else {
      // Otherwise, find the strategy to follow greedily.
      // For every straetgy in dst, find the strategy in src with
      // the lowest resharding cost.
      std::vector<int> arange(node_lens[src]);
      std::iota(arange.begin(), arange.end(), 0);
      for (int i = 0; i < node_lens[dst]; ++i) {
        std::vector<std::pair<double, int>> keys;

        // If there are multiple strategies with the same lowest costs,
        // prefer to follow "replicated", which has the largest index.
        // Node: We assume the strategy "Repilcated" is always appended
        // as the last strategy in BuildStrategyAndCost.
        for (int j = 0; j < node_lens[src]; ++j) {
          keys.push_back({edge_cost(i, j), -j});
        }

        std::sort(arange.begin(), arange.end(), [&keys](int l, int r) {
          return (keys[l].first < keys[r].first) ||
                 (keys[l].first == keys[r].first &&
                  keys[l].second < keys[r].second);
        });

        reindexing[i] = arange.front();
      }
    }
    merged_to_[src] = dst;
    reindexing_vector[src] = reindexing;

    // Merge edge cost matrix
    std::vector<int> adj_list(adjacency[src].begin(), adjacency[src].end());
    for (int adj : adj_list) {
      if (adj == dst) {
        continue;
      }
      Matrix added_edge_cost(node_lens[dst], node_lens[adj]);

      for (int i = 0; i < node_lens[dst]; ++i) {
        int j = reindexing[i];
        Matrix edge_cost_src_adj = GetEdgeCost(src, adj);
        for (int k = 0; k < node_lens[adj]; ++k) {
          added_edge_cost(i, k) = edge_cost(i, j) + edge_cost_src_adj(j, k);
        }
      }

      AddEdgeCost(dst, adj, added_edge_cost);
    }

    // Remove edges
    for (int adj : adj_list) {
      RemoveEdge(src, adj);
    }
  }

  int QueryDestination(int node) {
    if (merged_to_.count(node)) {
      int old_dst = merged_to_[node];
      int new_dst = QueryDestination(old_dst);
      if (old_dst != new_dst) {
        // Compresss path
        const std::vector<int>& old_reindexing_vector = reindexing_vector[node];
        std::vector<int> new_reindexing_vector;
        for (int i = 0; i < node_lens[new_dst]; ++i) {
          new_reindexing_vector.push_back(
              old_reindexing_vector[reindexing_vector[old_dst][i]]);
        }
        reindexing_vector[node] = new_reindexing_vector;
        merged_to_[node] = new_dst;
      }
      return new_dst;
    } else {
      return node;
    }
  }

  void Simplify() {
    const bool enable =
        pass_context::GetBool("auto_sharding::simplify_graph", true);

    // Merge nodes
    for (const auto& pair : to_merge_pairs_) {
      int src = pair.first;
      int dst = pair.second;
      dst = QueryDestination(dst);
      if (enable) {
        MergeNode(src, dst);
      }
    }

    // Build follow map
    follow_idx.reserve(node_lens.size());
    for (int i = 0; i < node_lens.size(); ++i) {
      if (merged_to_.count(i)) {
        follow_idx.push_back(QueryDestination(i));
      } else {
        follow_idx.push_back(-1);
      }
    }
  }

  int RemapIndex(int node_id, int value) const {
    if (follow_idx[node_id] < 0) {
      return value;
    } else {
      return reindexing_vector.at(node_id)[value];
    }
  }

  std::string ToString() {
    std::ostringstream os;
    os << "Cost Graph:" << std::endl;

    for (int i = 0; i < node_lens.size(); ++i) {
      os << "Node " << i << ": " << node_lens[i] << "\n";
    }
    os << "\n";

    for (const auto& iter : edge_costs) {
      os << "Edge (" << iter.first.first << ", " << iter.first.second << "):\n";
      os << iter.second.ToString() << "\n";
    }

    return os.str();
  }

  // The number of strategies of each node.
  std::vector<int> node_lens;
  // The adjacency list of each node.
  std::vector<absl::flat_hash_set<int>> adjacency;
  // The cost matrix between two nodes.
  absl::flat_hash_map<std::pair<int, int>, Matrix> edge_costs;
  // The reindexing vector of the node.
  // A reindexing vector maps a strategy index from the node being followed
  // to a strategy index of the curret node.
  absl::flat_hash_map<int, std::vector<int>> reindexing_vector;
  // Maps a node id to the node id that is being followed by this node.
  // The value is -1 if the current node does not follow any node.
  std::vector<int> follow_idx;

  // Save the destination of merged nodes.
  absl::flat_hash_map<int, int> merged_to_;
  // Save pairs that need to be merged.
  std::vector<std::pair<int, int>> to_merge_pairs_;
};

// Serialize parameters of the ILP problem as numpy arrays and call the python
// solver.
std::tuple<std::vector<int64>, std::vector<int64>, double> CallSolver(
    const HloInstructionSequence& sequence, const LivenessSet& liveness_set,
    const StrategyMap& strategy_map, const LeafStrategies& leaf_strategies,
    const CostGraph& cost_graph, const AliasSet& alias_set) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Serialize edges and edge costs to 1d numpy arrays
  int64 N = leaf_strategies.size();
  int64 M = pass_context::GetInt("auto_sharding::memory_budget_per_device", -1);
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
        d_np.push_back(strategies->leaf_vector[j].communication_cost);
        m_np.push_back(strategies->leaf_vector[j].memory_cost);
      }
    } else {
      std::vector<int> reindexing = cost_graph.reindexing_vector.at(i);
      s_len_np[i] = reindexing.size();
      for (size_t k = 0; k < reindexing.size(); ++k) {
        size_t j = reindexing[k];
        c_np.push_back(strategies->leaf_vector[j].compute_cost);
        d_np.push_back(strategies->leaf_vector[j].communication_cost);
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
  std::vector<int64> s_val, e_val;
  double objective;

  PyGILState_STATE gstate = PyGILState_Ensure();
  {
    py::object submodule = py::module_::import("parax.auto_sharding");
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

// Get the final sharding strategy according to the ilp solution.
const ShardingStrategy& GetShardingStrategy_(const HloInstruction* inst,
                                             const StrategyMap& strategy_map,
                                             const CostGraph& cost_graph,
                                             const std::vector<int64>& s_val) {
  const StrategyVector* strategies = strategy_map.at(inst).get();
  CHECK(!strategies->is_tuple);
  int node_idx = strategies->id;
  int stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategies->leaf_vector[stra_idx];
}

#define GetShardingStrategy(inst) \
    GetShardingStrategy_((inst), strategy_map, cost_graph, s_val)

// Return the output sharding of the reduce-scatter variant of a given strategy.
HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env) {
  if (ins->opcode() == HloOpcode::kDot) {
    const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
    int64 space_base_dim = dot_dnums.lhs_batch_dimensions_size();

    if (StrStartsWith(strategy.name, "RR = RS x SR")) {
      int mesh_dim;
      if (strategy.name.find("{0}") != std::string::npos) {
        mesh_dim = 0;
      } else {
        mesh_dim = 1;
      }
      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, cluster_env);
    } else {
      int mesh_dim0, mesh_dim1;
      if (strategy.name.find("{0,1}") != std::string::npos) {
        mesh_dim0 = 0; mesh_dim1 = 1;
      } else {
        mesh_dim0 = 1; mesh_dim1 = 0;
      }

      if (ins->shape().dimensions(space_base_dim) < cluster_env.device_mesh.dim(mesh_dim0) ||
          ins->shape().dimensions(space_base_dim + 1) < cluster_env.device_mesh.dim(mesh_dim1)) {
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                 {mesh_dim0, mesh_dim1}, cluster_env);
    }
  } else if (ins->opcode() == HloOpcode::kReduce) {
    // TODO(lmzheng): support more cases.
    CHECK_EQ(ins->shape().rank(), 1);

    int mesh_dim;
    if (strategy.name.find("[0]") != std::string::npos) {
      mesh_dim = 0;
    } else {
      mesh_dim = 1;
    }

    if (strategy.output_sharding.IsReplicated()) {
      return Tile(ins->shape(), {0}, {mesh_dim}, cluster_env);
    } else {
      Array<int64> tile_assignment = strategy.output_sharding.tile_assignment();
      tile_assignment.Reshape({cluster_env.total_devices});
      return HloSharding::Tile(std::move(tile_assignment));
    }
  } else {
    LOG(FATAL) << "Invalid instruction: " << ins->ToString();
  }
}

// Return whether an instruction has the opportunity to generate redcue-scatter.
bool HasReduceScatterOpportunity(const HloInstruction* inst,
                                 const StrategyMap& strategy_map,
                                 const CostGraph& cost_graph,
                                 const std::vector<int64>& s_val) {
  if (inst->opcode() == HloOpcode::kReduce && inst->shape().rank() == 1) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kDot) {
    if (GetShardingStrategy(inst->operand(0)).output_sharding.IsReplicated() &&
        GetShardingStrategy(inst->operand(1)).output_sharding.IsReplicated()) {
      // This dot is replicated on all devices. Do not split it.
      return false;
    }
    return true;
  }

  return false;
}

// Substitute all-reduce strategies with their reduce-scatter variants.
void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           const std::vector<int64>& s_val,
                           const ClusterEnvironment& cluster_env) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  const HloInstruction* output = instructions.back();

  std::vector<HloInstruction*> insert_all_gather;

  for (HloInstruction* inst : instructions) {
    if (HasReduceScatterOpportunity(inst, strategy_map, cost_graph, s_val)) {
      const ShardingStrategy& strategy = GetShardingStrategy(inst);

      if (strategy.name.find("allreduce") == std::string::npos) {
        continue;
      }

      absl::flat_hash_set<HloInstruction*> replicated_set;
      absl::flat_hash_set<HloInstruction*> boundary_set;
      absl::flat_hash_set<HloInstruction*> consumer_set;
      absl::flat_hash_set<const HloInstruction*> visited;

      // Use DFS to find the replicated set.
      std::function<void(HloInstruction*)> find_replicated_set;
      find_replicated_set = [&](HloInstruction* cur) {
        visited.insert(cur);

        // Check whether the node is a boundary node.
        absl::flat_hash_set<HloInstruction*> users = UsersWithAlias(cur, alias_map, output);

        for (HloInstruction* consumer : users) {
          if (consumer != output &&
              (GetShardingStrategy(consumer).output_sharding != strategy.output_sharding ||
               !DimensionsEqual(consumer->shape(), inst->shape()))) {
            boundary_set.insert(cur);
            return;
          }
        }

        // If this node is not a boundary node, propagate from this node.
        replicated_set.insert(cur);
        for (HloInstruction* consumer : users) {
          if (!visited.count(consumer)) {
            consumer_set.insert(consumer);
            find_replicated_set(consumer);
          }
        }
        for (size_t i = 0; i < cur->operand_count(); ++i) {
          HloInstruction* operand = cur->mutable_operand(i);
          if (!visited.count(operand) && !IsAlwaysReplicated(operand) &&
              GetShardingStrategy(operand).output_sharding == strategy.output_sharding &&
              DimensionsEqual(operand->shape(), inst->shape())) {
            find_replicated_set(operand);
          }
        }
      };

      // Find the replicated set starting from the all-reduce instruction.
      replicated_set.insert(inst);
      visited.insert(inst); visited.insert(output);
      for (HloInstruction* consumer : UsersWithAlias(inst, alias_map, output)) {
        find_replicated_set(consumer);
      }

      // Analyze the statistics of replicated set and boundary_set set.
      int num_replicated_parameters = 0;
      for (const HloInstruction* node : replicated_set) {
        if (node->opcode() == HloOpcode::kParameter) {
          num_replicated_parameters++;
        }
      }

      std::vector<HloInstruction*> need_all_gather;
      for (HloInstruction* node : boundary_set) {
        if (consumer_set.count(node)) {
          need_all_gather.push_back(node);
        }
      }

      // Print replicated set and boundary set
      //std::cerr << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
      //std::cerr << "replicated set (#parameter: " << num_replicated_parameters << "):\n";
      //for (auto x : replicated_set) {
      //  std::cerr << "  " << x->ToString(HloPrintOptions::ShortParsable()) << "\n";
      //}
      //std::cerr << "boundary set (#incompatible: " << need_all_gather.size() << "):\n";
      //for (auto x : boundary_set) {
      //  std::cerr << "  " << x->ToString(HloPrintOptions::ShortParsable())
      //            << " " << consumer_set.count(x) << "\n";
      //}

      // If applicable, replace all-reduce with reduce-scatter by
      // setting instructions' sharding.
      if (num_replicated_parameters >= 1 && need_all_gather.size() <= 1 &&
          replicated_set.size() >= 5) {
        HloSharding output_spec = GetReduceScatterOutput(inst, strategy, cluster_env);
        if (IsUndefined(output_spec)) { continue; }

        //std::cerr << "SET:  " << output_spec.ToString() << std::endl;

        if (StrStartsWith(strategy.name, "RR = RS x SR")) {
          // If set the sharding for this dot instruction, the SPMD
          // partitioner will generate bad fallback code.
          replicated_set.erase(inst);
        }

        for (HloInstruction* to_split : replicated_set) {
          to_split->set_sharding(output_spec);
        }

        for (HloInstruction* to_split : need_all_gather) {
          to_split->set_sharding(output_spec);
          if (to_split->users().size() == 1 && to_split->users().front() == output &&
              alias_map.count(to_split)) {
            // Move the all-gather to its alias parameter
            alias_map.at(to_split)->set_sharding(output_spec);
            insert_all_gather.push_back(alias_map.at(to_split));
          } else {
            insert_all_gather.push_back(to_split);
          }
        }
      }

      //std::cerr << "-----------------------done\n" << std::endl;
    }
  }

  // Insert all-gather on the output of boundary nodes by setting
  // their shardings.
  for (HloInstruction* inst : insert_all_gather) {
    HloInstruction* replace_with = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(inst->shape(), inst));
    replace_with->set_sharding(GetShardingStrategy(inst).output_sharding);
    inst->ReplaceAllUsesWith(replace_with);
  }
}

// Set the HloSharding for all instructions according to the ILP solution.
void SetHloSharding(const HloInstructionSequence& sequence,
                    const StrategyMap& strategy_map,
                    const CostGraph& cost_graph,
                    const std::vector<int64>& s_val,
                    const ClusterEnvironment& cluster_env) {
  // Set the HloSharding for every instruction
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  for (HloInstruction* inst : instructions) {
    if (inst->has_sharding()) {  // its sharding has been forcibly set
      continue;
    }
    auto iter = strategy_map.find(inst);
    if (iter == strategy_map.end()) { continue; }

    const StrategyVector* strategies = iter->second.get();
    if (strategies->is_tuple) {
      const Shape& out_shape = inst->shape();
      ShapeTree<HloSharding> tuple_sharding(out_shape, Undefined());
      std::function<void(const StrategyVector*)> get_flattened_shardings;
      std::vector<HloSharding> flattened_shardings;
      get_flattened_shardings = [&](const StrategyVector* cur) {
        if (cur->is_tuple) {
          for (const auto& child : strategies->childs) {
            get_flattened_shardings(child.get());
          }
        } else {
          CHECK(cur->following != nullptr);
          if (!instructions[cur->following->instruction_id]->shape().IsTuple() &&
              instructions[cur->following->instruction_id]->has_sharding()) {
            // its sharding has been forcibly set
            flattened_shardings.push_back(
                instructions[cur->following->instruction_id]->sharding());
          } else {
            int node_idx = cur->id;
            int stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
            flattened_shardings.push_back(
                cur->leaf_vector[stra_idx].output_sharding);
          }
        }
      };
      get_flattened_shardings(strategies);
      int i = 0;
      for (auto& leaf : tuple_sharding.leaves()) {
        leaf.second = flattened_shardings[i++];
      }
      CHECK_EQ(i, flattened_shardings.size());
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
  for (HloInstruction* inst : sequence.instructions()) {
    if (inst->opcode() == HloOpcode::kDot) {
      // For some dot instructions, our formulation think they are valid.
      // But the the spmd partitioner cannot infer the correct dot algorithms
      // from the input/output sharding. It then generates bad fallback code.
      // Here we insert some extra annotated identiy instructions to help the
      // spmd partitioner generate correct code.

      const ShardingStrategy& stra = GetShardingStrategy(inst);

      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const HloSharding& lhs_sharding = lhs->sharding();
      const HloSharding& rhs_sharding = rhs->sharding();
      const DotDimensionNumbers& dot_dnums = inst->dot_dimension_numbers();
      std::vector<int64> lhs_space_dims, rhs_space_dims;
      std::tie(lhs_space_dims, rhs_space_dims) =
          GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);
      const auto& lhs_con_dims = dot_dnums.lhs_contracting_dimensions();
      const auto& rhs_con_dims = dot_dnums.rhs_contracting_dimensions();

      // TODO(lmzheng): cover more cases.
      if (stra.name == "SR = SS x SR @ {0,1} (allreduce @ 1)" &&
          lhs_sharding.IsReplicated()) {
        ForceOperandSharding(inst, 0, Tile(lhs->shape(),
            {lhs_space_dims[0], lhs_con_dims[0]}, {0, 1}, cluster_env));
      } else if (stra.name == "SR = SS x SR @ {1,0} (allreduce @ 0)" &&
          lhs_sharding.IsReplicated()) {
        ForceOperandSharding(inst, 0, Tile(lhs->shape(),
            {lhs_space_dims[0], lhs_con_dims[0]}, {1, 0}, cluster_env));
      }
    } else if (inst->opcode() == HloOpcode::kIota) {
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

// Print sorted instructions.
std::string PrintInstructions(const HloInstructionSequence& sequence) {
  std::ostringstream os;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << "Instruction " << i << ": " << instructions[i]->ToString() << "\n";
  }
  return os.str();
}

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

// Print the choosen auto sharding strategy for debugging.
std::string PrintAutoShardingSolution(const HloInstructionSequence& sequence,
                                      const LivenessSet& liveness_set,
                                      const StrategyMap& strategy_map,
                                      const LeafStrategies& leaf_strategies,
                                      const CostGraph& cost_graph,
                                      const std::vector<int64>& s_val,
                                      double objective) {
  std::ostringstream os;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  size_t N = leaf_strategies.size();

  // Print the choosen strategy
  os << "=== Auto sharding strategy ===\n";
  for (size_t i = 0; i < N; ++i) {
    int stra_idx = cost_graph.RemapIndex(i, s_val[i]);
    const ShardingStrategy& stra = leaf_strategies[i]->leaf_vector[stra_idx];

    os << i << " "
       << instructions[leaf_strategies[i]->instruction_id]->ToString(
              HloPrintOptions::ShortParsable())
       << "  ";
    if (cost_graph.follow_idx[i] < 0) {
      os <<  stra.name << " ";
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
        os << "- edge cost (" << a << ", " << b << "): "
           << std::fixed << std::setprecision(2) << cost << "\n";
      }
    }
  }

  os << "objective : " << std::fixed << std::setprecision(2) << objective << "\n"; 
 
  return os.str();
}

StatusOr<bool> AutoSharding::Run(HloModule* module) {
  if (!pass_context::GetBool("auto_sharding::enable", false)) {
    return false;
  }

  // ----- Set options for this pass -----
  AutoShardingSolverOption solver_option;
  solver_option.override_all_gather_cost = false;
  solver_option.override_all_reduce_cost = false;
  solver_option.override_reduce_scatter_cost = false;

  if (pass_context::GetBool("auto_sharding::force_all_gather_cost", false)) {
    solver_option.override_all_gather_cost = true;
    solver_option.all_gather_cost =
        pass_context::GetDouble("auto_sharding::all_gather_cost");
  }

  solver_option.force_batch_dim_to_mesh_dim =
      pass_context::GetInt("auto_sharding::force_batch_dim_to_mesh_dim", -1);
  solver_option.prefer_reduce_scatter =
      pass_context::GetBool("auto_sharding::prefer_reduce_scatter", false);
  solver_option.allow_recompute_heavy_op =
      pass_context::GetBool("auto_sharding::allow_recompute_heavy_op", false);
  solver_option.load_solution_vector =
      pass_context::GetBool("auto_sharding::load_solution_vector", false);

  // std::cerr << "===== Enter AutoSharding =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

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
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloLiveRange> hlo_live_range,
      HloLiveRange::Run(schedule, *alias_analysis, entry_computation));

  absl::flat_hash_map<const HloValue*, HloLiveRange::TimeBound>&
      buffer_live_ranges = hlo_live_range->buffer_live_ranges();

  LivenessSet liveness_set(hlo_live_range->schedule_end_time() + 1);
  for (const auto& iter : buffer_live_ranges) {
    for (int64 i = iter.second.start; i <= iter.second.end; ++i) {
      liveness_set[i].push_back(iter.first);
    }
  }
  // std::cerr << hlo_live_range->ToString() << std::endl;
  // std::cerr << PrintLivenessSet(liveness_set);

  // ----- Analyze depth -----
  const HloInstructionSequence& sequence =
      hlo_live_range->flattened_instruction_sequence();
  InstructionDepthMap ins_depth_map;
  ins_depth_map = BuildInstructionDepthMap(sequence);

  // ----- Read parameters of device mesh -----
  Array<int64> device_mesh(
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

  // ----- Build strategies and costs -----
  StrategyMap strategy_map;
  LeafStrategies leaf_strategies;
  AssociativeDotPairs associative_dot_pairs;

  AliasMap alias_map = BuildAliasMap(module, alias_analysis->dataflow_analysis());
  std::tie(strategy_map, leaf_strategies, associative_dot_pairs) = BuildStrategyAndCost(
      sequence, ins_depth_map, alias_map, cluster_env, solver_option);
  AliasSet alias_set =
      BuildAliasSet(module, alias_analysis->dataflow_analysis(), strategy_map);
  //std::cerr << PrintStrategyMap(strategy_map, sequence);

  // ----- Build cost graph and merge unimporant nodes -----
  CostGraph cost_graph(leaf_strategies, associative_dot_pairs);
  cost_graph.Simplify();

  // ----- Call the ILP solver -----
  std::vector<int64> s_val, e_val;
  double objective = -1.0;
  if (!solver_option.load_solution_vector) {
    std::tie(s_val, e_val, objective) = CallSolver(
      sequence, liveness_set, strategy_map, leaf_strategies, cost_graph, alias_set);
  } else {
    s_val = pass_context::GetIntVector("auto_sharding::solution_vector");
  }

  if (pass_context::GetBool("auto_sharding::print_strategy", false)) {
    std::cerr << PrintAutoShardingSolution(sequence, liveness_set, strategy_map,
                                           leaf_strategies, cost_graph, s_val, objective);
  }

  // ----- Substitute all-reduce with reduce-scatter -----
  if (solver_option.prefer_reduce_scatter) {
    GenerateReduceScatter(sequence, alias_map, strategy_map,
                          cost_graph, s_val, cluster_env);
  }

  // ----- Set sharding for all instructions -----
  SetHloSharding(sequence, strategy_map, cost_graph, s_val, cluster_env);

  // std::cerr << "===== Exit AutoSharding =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

}  // namespace spmd
}  // namespace xla
