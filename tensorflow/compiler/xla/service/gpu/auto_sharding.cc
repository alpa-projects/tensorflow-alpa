#include "tensorflow/compiler/xla/service/gpu/auto_sharding.h"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/shape_util.h"


namespace xla {
namespace gpu {

namespace py = pybind11;

double GetBytes(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
};

// Options for auto-sharding solver
struct AutoShardingSolverOption {
  bool force_data_parallel;
  int forward_backward_sep_id;

  bool override_all_reduce_cost;
  double all_reduce_cost;

  bool override_reduce_scatter_cost;
  double reduce_scatter_cost;
};

// One sharding strategy 
struct ShardingStrategy {
  std::string name;
  HloSharding output_sharding;
  double compute_cost;
  double communication_cost;
  double memory_cost;
  std::vector<std::vector<double>> resharding_costs;
};

using LivenessSet = std::vector<std::vector<const HloValue*>>;
using StrategyMap = absl::flat_hash_map<const HloInstruction*, std::vector<ShardingStrategy>>;
using InstructionIdMap = absl::flat_hash_map<const HloInstruction*, size_t>;
using FollowMap = absl::flat_hash_map<const HloInstruction*, const HloInstruction*>;
using AliasSet = absl::flat_hash_set<std::pair<size_t, size_t>>;

// Cluster environment to model the communication cost
class ClusterEnvironment {
 public:
  ClusterEnvironment(int num_devices, const AutoShardingSolverOption& solver_option)
    : num_devices(num_devices), solver_option(solver_option) {}

  double AllReduceCost(double num_bytes) const {
    if (solver_option.override_all_reduce_cost) {
      return solver_option.all_reduce_cost;
    }

    return (alpha +
           beta * 2 * (num_devices - 1) / num_devices * num_bytes +
           0.1);
  }

  double AllGatherCost(double num_bytes) const {
    return (alpha +
           beta * (num_devices - 1) / num_devices * num_bytes +
           0.01) + all_gather_penalty;
  }

  double ReduceScatterCost(double num_bytes) const {
    if (solver_option.override_reduce_scatter_cost) {
      return solver_option.reduce_scatter_cost;
    }

    return (alpha +
           beta * (num_devices - 1) / num_devices * num_bytes +
           0.001);
  }

  double ReshardingCost(const Shape& shape, const HloSharding& src_sharding,
                        const HloSharding& dst_sharding) const {
    if (src_sharding == dst_sharding) {
      return 0;
    }

    if (src_sharding.IsReplicated()) {
      if (dst_sharding.IsPartialReduction()) {
        // An elementwise divide will occur here,
        // so we should add a penanlty.
        return partial_reduction_penalty;
      } else {
        return 0;
      }
    }

    if (src_sharding.IsPartialReduction()) {
      return AllReduceCost(GetBytes(shape)) + partial_reduction_penalty;
    }

    return AllGatherCost(GetBytes(shape));
  }

  const double alpha = 1.0;
  const double beta = 1.0;

  // disencourage the apperance of partial reduction
  const double partial_reduction_penalty = 0;

  // prefer all-reduce to all-gather
  const double all_gather_penalty = 0;

  int num_devices;

  // The solver option may override the cost of all-reduce and reduce-scatter
  // to force data-parallel.
  const AutoShardingSolverOption& solver_option;
};

InstructionIdMap BuildInstructionIdMap(const HloInstructionSequence& sequence) {
  InstructionIdMap ret;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    ret[instructions[i]] = i;
  }
  return ret;
}

int EstimateForwardBackwardSep(const HloModule* module,
                               const HloInstructionSequence& sequence,
                               const InstructionIdMap& ins_id_map) {
  HloComputation* entry = module->entry_computation();

  // Count used_by map for every parameter
  absl::flat_hash_map<const HloInstruction*, std::vector<int>> used_by;
  for (const HloInstruction* inst : sequence.instructions()) {
    for (size_t j = 0; j < inst->operand_count(); ++j) {
      const HloInstruction* operand = inst->operand(j);
      if (operand->opcode() == HloOpcode::kParameter) {
        used_by[operand].push_back(ins_id_map.at(inst));
      }
    }
  }

  // Estimate forward/backward separation
  int sep = 0;
  for (const auto& iter : used_by) {
    if (iter.second.size() > 2) {
      sep = std::max(sep, iter.second.front() + 1);
    }
  }

  return sep;
}

// Create a HloSharding that splits a single dim
HloSharding Split(const Shape& shape, int dim, const ClusterEnvironment& cluster_env) {
  std::vector<int64> new_shape;
  for (int i = 0; i < shape.rank(); ++i) {
    new_shape.push_back(i == dim ? cluster_env.num_devices : 1);
  }
  Array<int64> tile_assignment(new_shape);
  tile_assignment.FillIota(0);
  return HloSharding::Tile(tile_assignment);
}

// Get the space dimensions of a dot instruction
std::pair<std::vector<int64>, std::vector<int64>> GetSpaceDims(
  const Shape& lhs_shape,
  const Shape& rhs_shape,
  const DotDimensionNumbers& dnums
) {
  std::vector<int64> lhs_space_dims;
  std::vector<int64> rhs_space_dims;

  for (int64 i = 0; i < lhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    lhs_space_dims.push_back(i);
  }

  for (int64 i = 0; i < rhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    rhs_space_dims.push_back(i);
  }
  return std::make_pair(std::move(lhs_space_dims), std::move(rhs_space_dims));
}

// Compute the resharding cost vector from multiple possible strategies
// to a desired sharding spec
std::vector<double> ReshardingCostVector(const std::vector<ShardingStrategy>& strategies,
                                         const Shape& shape,
                                         const HloSharding& required_sharding,
                                         const ClusterEnvironment& cluster_env) {
  std::vector<double> ret;
  for (const auto& x : strategies) {
    ret.push_back(cluster_env.ReshardingCost(shape, x.output_sharding, required_sharding));
  }
  return ret;
}

// Build possible sharding strategies and their costs for all instructions
std::pair<StrategyMap, FollowMap> BuildStrategyAndCost(
  const HloInstructionSequence& sequence,
  const ClusterEnvironment& cluster_env,
  const AutoShardingSolverOption& solver_option
) {
  StrategyMap strategy_map;
  FollowMap follow_map;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    const HloInstruction* ins = instructions[i];
    std::vector<ShardingStrategy> strategies;
    switch (ins->opcode()) {
      case HloOpcode::kParameter: {
        // Split one dim
        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          HloSharding output_sharding = Split(ins->shape(), i, cluster_env);
          double compute_cost = 0;
          double communication_cost = 0;
          double memory_cost = GetBytes(ins->shape()) / cluster_env.num_devices;
          strategies.push_back(
            ShardingStrategy({name, output_sharding,
                              compute_cost, communication_cost,
                              memory_cost, {}}));
        }

        // Replicate
        strategies.push_back(
          ShardingStrategy({"R", HloSharding::Replicate(),
                            1, 0, GetBytes(ins->shape()),
                            {}}));
        break;
      }

      case HloOpcode::kConstant: {
        strategies.push_back(
          ShardingStrategy({"R", HloSharding::Replicate(),
                            0, 0, GetBytes(ins->shape()),
                            {}}));
        break;
      }

      case HloOpcode::kBroadcast: {
        const auto& dimensions = ins->dimensions();
        const HloInstruction* operand = ins->operand(0);

        // Split one dim
        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          std::vector<double> resharding_costs;

          auto it = absl::c_find(dimensions, i);
          if (it == dimensions.end()) {
            resharding_costs = ReshardingCostVector(
              strategy_map[operand], operand->shape(),
              HloSharding::Replicate(), cluster_env);
          } else {
            int64 original_dim = std::distance(dimensions.begin(), it);
            resharding_costs = ReshardingCostVector(
              strategy_map[operand], operand->shape(),
              Split(operand->shape(), original_dim, cluster_env), cluster_env);
          }

          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {std::move(resharding_costs)}}));
        }

        // Replicate
        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env)}}));
        break;
      }

      case HloOpcode::kTranspose: {
        const HloInstruction* operand = ins->operand(0);
        follow_map[ins] = operand;
        const auto& dimensions = ins->dimensions();

        // Split one dim
        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          int64 original_dim = dimensions[i];
          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {ReshardingCostVector(strategy_map[operand], operand->shape(),
                  Split(operand->shape(), original_dim, cluster_env), cluster_env)}}));
        }

        // Partial Reduction
        strategies.push_back(
          ShardingStrategy({
          "P", HloSharding::PartialReduction(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                HloSharding::PartialReduction(), cluster_env)}}));

        // Replicate
        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env)}}));

        break;
      }

      case HloOpcode::kReshape: {
        const HloInstruction* operand = ins->operand(0);
        follow_map[ins] = operand;
        const Shape& old_shape = operand->shape();
        const Shape& new_shape = ins->shape();

        // Construct a map that maps a new dimension to its corresponding old dimension
        absl::flat_hash_map<int, int> dim_mapping;
        int new_pt = -1;
        int old_pt = -1;
        size_t old_prod = 1;
        size_t new_prod = 1;

        while (true) {
          bool move_new = false;
          bool move_old = false;

          if (new_prod == old_prod) {
            dim_mapping[new_pt + 1] = old_pt + 1;
            move_old = move_new = true;
          } else if (new_prod < old_prod) {
            move_new = true;
          } else {
            move_old = true;
          }

          if (move_new) {
            new_pt += 1;
            if (new_pt < new_shape.rank()) {
              new_prod *= new_shape.dimensions(new_pt);
            } else {
              break;
            }
          }

          if (move_old) {
            old_pt += 1;
            if (old_pt < old_shape.rank()) {
              old_prod *= old_shape.dimensions(old_pt);
            } else {
              break;
            }
          }
        }

        // Split one dim
        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          std::vector<double> resharding_costs;

          auto it = dim_mapping.find(i);
          if (it == dim_mapping.end()) {
            resharding_costs = ReshardingCostVector(
              strategy_map[operand], operand->shape(),
              HloSharding::Replicate(), cluster_env);
          } else {
            int64 original_dim = it->second;
            resharding_costs = ReshardingCostVector(
              strategy_map[operand], operand->shape(),
              Split(operand->shape(), original_dim, cluster_env), cluster_env);
          }

          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {std::move(resharding_costs)}}));
        }

        // Partial Reduction
        strategies.push_back(
          ShardingStrategy({
          "P", HloSharding::PartialReduction(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                HloSharding::PartialReduction(), cluster_env)}}));

        // Replicate
        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env)}}));

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
        follow_map[ins] = ins->operand(0);

        // Split one dim
        for (int64 i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);

          std::vector<std::vector<double>> resharding_costs;
          for (size_t j = 0; j < ins->operand_count(); ++j) {
            const HloInstruction* operand = ins->operand(j);
            resharding_costs.push_back(
              ReshardingCostVector(strategy_map[operand], operand->shape(),
                                   Split(operand->shape(), i, cluster_env),
                                   cluster_env));
          }
          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            resharding_costs}));
        }

        if (ins->opcode() == HloOpcode::kAdd) {
          // Add strategies to Operate on partial reduction.
          // This is a special treatment to simplify 
          // `all_reduce(x) + all_reduce(y) == all_reduce(x + y)`

          // Cannot do follow in this case:
          // The solver has to choose between "R = P + P" and "P = P + P",
          // which cannot be determinded greedily.
          for (size_t i = 0; i < ins->operand_count(); ++i) {
            const HloInstruction* operand = ins->operand(i);
            if (operand->opcode() == HloOpcode::kReshape) {
              follow_map.erase(ins);
            }
          }

          // R = P + P
          std::vector<std::vector<double>> resharding_costs;
          for (size_t i = 0; i < ins->operand_count(); ++i) {
            const HloInstruction* operand = ins->operand(i);
            resharding_costs.push_back(
              ReshardingCostVector(strategy_map[operand], operand->shape(),
                                   HloSharding::PartialReduction(), cluster_env));
          }
          strategies.push_back(
            ShardingStrategy({
            "R = P + P", HloSharding::Replicate(),
            0, cluster_env.AllReduceCost(GetBytes(ins->shape())), GetBytes(ins->shape()),
            resharding_costs}));

          // P = P + P
          resharding_costs.clear();
          for (size_t i = 0; i < ins->operand_count(); ++i) {
            const HloInstruction* operand = ins->operand(i);
            resharding_costs.push_back(
              ReshardingCostVector(strategy_map[operand], operand->shape(),
                                   HloSharding::PartialReduction(), cluster_env));

          }
          strategies.push_back(
            ShardingStrategy({
            "P = P + P", HloSharding::PartialReduction(),
            1, 0, GetBytes(ins->shape()),
            resharding_costs}));
        }

        // Replicate
        std::vector<std::vector<double>> resharding_costs;
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          resharding_costs.push_back(
            ReshardingCostVector(strategy_map[operand], operand->shape(),
                                 HloSharding::Replicate(), cluster_env));
        }
        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          resharding_costs}));

        break;
      }

      case HloOpcode::kReduce: {
        const HloInstruction* operand = ins->operand(0);
        follow_map[ins] = ins->operand(0);
        const HloInstruction* unit = ins->operand(1);
        const auto& dimensions = ins->dimensions();
        absl::flat_hash_map<size_t, size_t> dim_mapping;

        size_t pt = 0;
        for (size_t i = 0; i < operand->shape().rank(); ++i) {
          if (absl::c_find(dimensions, i) != dimensions.end()) {
            continue;
          }
          dim_mapping[pt] = i;
          pt += 1;
        }

        CHECK_EQ(pt, ins->shape().rank());

        // Split one dim
        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          int64 original_dim = dim_mapping.at(i);
          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {ReshardingCostVector(strategy_map[operand], operand->shape(),
                  Split(operand->shape(), original_dim, cluster_env), cluster_env),
             ReshardingCostVector(strategy_map[unit], unit->shape(),
                  HloSharding::Replicate(), cluster_env)}}));
        }

        // Replicate with all-reduce
        for (int dim : dimensions) {
          strategies.push_back(
            ShardingStrategy({
            "R (all-reduce)", HloSharding::Replicate(),
            0, cluster_env.AllReduceCost(GetBytes(ins->shape())), GetBytes(ins->shape()),
            {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                  Split(operand->shape(), dim, cluster_env), cluster_env),
             ReshardingCostVector(strategy_map[unit], unit->shape(),
                                  HloSharding::Replicate(), cluster_env)}}));
        }

        // Replicate
        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(strategy_map[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env),
           ReshardingCostVector(strategy_map[unit], unit->shape(),
                                HloSharding::Replicate(), cluster_env)}}));
        break;
      }

      case HloOpcode::kDot: {
        const HloInstruction* lhs = ins->operand(0);
        const HloInstruction* rhs = ins->operand(1);
        const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
        std::vector<int64> lhs_space_dims, rhs_space_dims;
        std::tie(lhs_space_dims, rhs_space_dims) = GetSpaceDims(
          lhs->shape(), rhs->shape(), dot_dnums);

        CHECK_EQ(lhs_space_dims.size(), 1);
        CHECK_EQ(rhs_space_dims.size(), 1);
        CHECK_EQ(dot_dnums.lhs_contracting_dimensions_size(), 1);
        CHECK_EQ(dot_dnums.rhs_contracting_dimensions_size(), 1);

        int64 space_base_dim = dot_dnums.lhs_batch_dimensions_size();

        // Split the space dim of lhs
        strategies.push_back(
          ShardingStrategy({
          "Sl = Sl x R", Split(ins->shape(), space_base_dim, cluster_env),
          0, 0, GetBytes(ins->shape()) / cluster_env.num_devices, 
          {
            ReshardingCostVector(strategy_map[lhs], lhs->shape(),
                                 Split(lhs->shape(), lhs_space_dims[0], cluster_env), cluster_env),
            ReshardingCostVector(strategy_map[rhs], rhs->shape(), HloSharding::Replicate(), cluster_env),
          }}));

        // Split the space dim of rhs
        if (!solver_option.force_data_parallel) {
          strategies.push_back(
            ShardingStrategy({
            "Sr = R x Sr", Split(ins->shape(), space_base_dim + 1, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices, 
            {
              ReshardingCostVector(strategy_map[lhs], lhs->shape(), HloSharding::Replicate(), cluster_env),
              ReshardingCostVector(strategy_map[rhs], rhs->shape(),
                                   Split(rhs->shape(), rhs_space_dims[0], cluster_env), cluster_env),
            }}));
        }
  
        // Split the contracting dim, do all-reduce immediately
        if (!solver_option.force_data_parallel || i > solver_option.forward_backward_sep_id) {
          strategies.push_back(
            ShardingStrategy({
            "R = Sk x Sk", HloSharding::Replicate(),
            0, cluster_env.AllReduceCost(GetBytes(ins->shape())), GetBytes(ins->shape()),
            {
              ReshardingCostVector(strategy_map[lhs], lhs->shape(),
                Split(lhs->shape(), dot_dnums.lhs_contracting_dimensions(0), cluster_env), cluster_env),
              ReshardingCostVector(strategy_map[rhs], rhs->shape(),
                Split(rhs->shape(), dot_dnums.rhs_contracting_dimensions(0), cluster_env), cluster_env),
            }}));

          // Split the contracting dim, defer the all-reduce
          strategies.push_back(
            ShardingStrategy({
            "P = Sk x Sk", HloSharding::PartialReduction(),
            1, 0, GetBytes(ins->shape()),
            {
              ReshardingCostVector(strategy_map[lhs], lhs->shape(),
                Split(lhs->shape(), dot_dnums.lhs_contracting_dimensions(0), cluster_env), cluster_env),
              ReshardingCostVector(strategy_map[rhs], rhs->shape(),
                Split(rhs->shape(), dot_dnums.rhs_contracting_dimensions(0), cluster_env), cluster_env),
            }}));
        }

        if (dot_dnums.lhs_batch_dimensions_size()) {
          strategies.clear();

          // Split the batch dim
          for (size_t i = 0; i < dot_dnums.lhs_batch_dimensions_size(); ++i) {
            strategies.push_back(
              ShardingStrategy({
              "Sb = Sb x Sb " + std::to_string(i), Split(ins->shape(), i, cluster_env),
              0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
              {
                ReshardingCostVector(strategy_map[lhs], lhs->shape(),
                  Split(lhs->shape(), dot_dnums.lhs_batch_dimensions(i), cluster_env), cluster_env),
                ReshardingCostVector(strategy_map[rhs], rhs->shape(),
                  Split(rhs->shape(), dot_dnums.rhs_batch_dimensions(i), cluster_env), cluster_env),
              }}));
          }
        }
 
        //// Replicate
        //strategies.push_back(
        //  ShardingStrategy({
        //  "R = R x R", HloSharding::Replicate(),
        //  0, 0, GetBytes(ins->shape()),
        //  {
        //    ReshardingCostVector(strategy_map[lhs], lhs->shape(), HloSharding::Replicate(), cluster_env),
        //    ReshardingCostVector(strategy_map[rhs], rhs->shape(), HloSharding::Replicate(), cluster_env),
        //  }}));

        break;
      }
      case HloOpcode::kTuple: {
        std::vector<std::vector<double>> resharding_costs;

        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          resharding_costs.push_back(std::vector<double>(strategy_map[operand].size(), 0));
        }

        strategies.push_back(
          ShardingStrategy({
          "tuple_follow", HloSharding::Replicate(),
          0, 0, 0, resharding_costs}));
        break;
      }
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->name();
    }

    strategy_map[ins] = strategies;
  }

  return std::make_pair(std::move(strategy_map), std::move(follow_map));
}

AliasSet BuildAliasSet(
  const HloModule* module,
  const HloDataflowAnalysis& dataflow_analysis,
  const InstructionIdMap& ins_id_map
) {
  // Adjust the edge cost for alias (donated buffer).
  // Typically, old weights and new weights are aliases, so we should
  // let them have the same sharding spec.
  const HloInputOutputAliasConfig& alias_config = module->input_output_alias_config();

  HloComputation* entry = module->entry_computation();
  const std::vector<HloInstruction*>& parameter_instructions =
    entry->parameter_instructions();
  const HloInstruction* output_tuple = entry->root_instruction();

  // TODO: handle tuple args
  AliasSet alias_set;
  alias_config.ForEachAlias(
    [&](const ShapeIndex& output_index, const HloInputOutputAliasConfig::Alias& alias) {
      const HloInstruction* src_ins = parameter_instructions[alias.parameter_number];
      const HloInstruction* dst_ins = dataflow_analysis.GetUniqueValueAt(
          output_tuple, output_index).instruction();

      alias_set.insert(std::make_pair(ins_id_map.at(src_ins), ins_id_map.at(dst_ins)));
    });

  return alias_set;
}

// A simple matrix class to store and manipulate on cost matrices on edges.
class Matrix {
 public:
  Matrix() : n(0), m(0), transpose(false), data(nullptr) {}

  Matrix(size_t n, size_t m) {
    this->n = n;
    this->m = m;
    transpose = false;
    data = std::make_shared<std::vector<double>>(n * m, 0.0);
  }

  Matrix(size_t n, size_t m, bool transpose, std::shared_ptr<std::vector<double>> data) {
    this->n = n;
    this->m = m;
    this->transpose = transpose;
    this->data = data;
  }

  Matrix Transpose() {
    return Matrix(m, n, !transpose, data);
  }

  double operator()(size_t i, size_t j) const {
    size_t idx;
    if (transpose) {
      idx = j * n + i;
    } else {
      idx = i * m + j;
    }
    CHECK(data != nullptr) << n << " , " << m;
    return (*data)[idx];
  }

  double& operator()(size_t i, size_t j) {
    size_t idx;
    if (transpose) {
      idx = j * n + i;
    } else {
      idx = i * m + j;
    }
    CHECK(data != nullptr) << n << " . " << m;
    return (*data)[idx];
  }

  Matrix operator+(const Matrix& other) {
    CHECK_EQ(n, other.n);
    CHECK_EQ(m, other.m);
    Matrix ret = Matrix(n, m);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        ret(i, j) = operator()(i, j) + other(i, j);
      }
    }
    return ret;
  }

  std::string ToString() const {
    std::ostringstream os;

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        os << operator()(i, j) << " ";
      }
      os << "\n";
    }

    return os.str();
  }

  size_t n;
  size_t m;
  bool transpose;
  std::shared_ptr<std::vector<double>> data;
};

// A graph data structure to simplify the edge cost graph.
// It merges nodes and does path compression.
class CostGraph {
 public:
  CostGraph(const HloInstructionSequence& sequence,
            const StrategyMap& strategy_map,
            const FollowMap& follow_map,
            const InstructionIdMap& ins_id_map) {
    const std::vector<HloInstruction*>& instructions = sequence.instructions();

    node_lens.reserve(instructions.size());
    adjacency.assign(instructions.size(), absl::flat_hash_set<int>());

    for (const auto& ins : instructions) {
      node_lens.push_back(strategy_map.at(ins).size());
  
      for (const auto& strategy : strategy_map.at(ins)) {
        CHECK_EQ(strategy.resharding_costs.size(), ins->operand_count());
      }
  
      for (size_t i = 0; i < ins->operand_count(); ++i) {
        const HloInstruction* src = ins->operand(i);
        const HloInstruction* dst = ins;
  
        size_t src_idx = ins_id_map.at(src);
        size_t dst_idx = ins_id_map.at(dst);

        Matrix edge_cost(node_lens[src_idx], node_lens[dst_idx]);
        for (size_t k = 0; k < strategy_map.at(dst).size(); ++k) {
          const ShardingStrategy& stra = strategy_map.at(dst)[k];
          for (size_t j = 0; j < stra.resharding_costs[i].size(); ++j) {
            edge_cost(j, k) = stra.resharding_costs[i][j];
          }
        }

        AddEdgeCost(src_idx, dst_idx, edge_cost);
      }

      const auto& iter = follow_map.find(ins);
      if (iter != follow_map.end()) {
        to_merge_pairs.push_back({ins_id_map.at(ins), ins_id_map.at(iter->second)});
      }
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
    CHECK(!merged_to.count(dst));
    CHECK_NE(src, dst);

    //std::cerr << "Merge: " << src << " to " << dst << std::endl;

    Matrix edge_cost = edge_costs[{dst, src}];

    // Find the strategy to follow greedily
    std::vector<int> reindexing;

    std::vector<int> arange(node_lens[src]);
    std::iota(arange.begin(), arange.end(), 0);
    for (int i = 0; i < node_lens[dst]; ++i) {
      std::vector<std::pair<double, int>> keys;

      // Pick the strategy with the lowest cost to follow.
      // If there are multiple strategies with the same lowest costs,
      // prefer to follow "replicated", which has the largest index.
      // Node: We assume the strategy "Repilcated" is always appended
      // as the last strategy in BuildStrategyAndCost.

      for (int j = 0; j < node_lens[src]; ++j) {
        keys.push_back({edge_cost(i,j), -j});
      }

      std::sort(arange.begin(), arange.end(), [&keys](int l, int r) {
        return (keys[l].first < keys[r].first) ||
               (keys[l].first == keys[r].first && keys[l].second < keys[r].second);
      });

      reindexing.push_back(arange.front());
    }
    merged_to[src] = dst;
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
    if (merged_to.count(node)) {
      int old_dst = merged_to[node];
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
        merged_to[node] = new_dst;
      }
      return new_dst;
    } else {
      return node;
    }
  }

  void Simplify() {
    // Merge nodes
    for (const auto& pair : to_merge_pairs) {
      int src = pair.first;
      int dst = pair.second;
      CHECK(!merged_to.count(src));
      dst = QueryDestination(dst);
      MergeNode(src, dst);
    }

    // Build follow map
    follow_idx.reserve(node_lens.size());
    for (int i = 0; i < node_lens.size(); ++i) {
      if (merged_to.count(i)) {
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
      os << "Edge (" << iter.first.first <<  ", " << iter.first.second << "):\n";
      os << iter.second.ToString() << "\n";
    }

    return os.str();
  }

  std::vector<int> node_lens;
  std::vector<absl::flat_hash_set<int>> adjacency;
  absl::flat_hash_map<std::pair<int, int>, Matrix> edge_costs;
  absl::flat_hash_map<int, std::vector<int>> reindexing_vector;
  absl::flat_hash_map<int, int> merged_to;
  std::vector<int> follow_idx;

  std::vector<std::pair<int, int>> to_merge_pairs;
};


std::pair<std::vector<int>, std::vector<int>> CallSolver(
  const HloModule* module,
  const HloInstructionSequence& sequence,
  const LivenessSet& liveness_set,
  const StrategyMap& strategy_map,
  const CostGraph& cost_graph,
  const InstructionIdMap& ins_id_map,
  const AliasSet& alias_set
) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Serialize edges and edge costs to 1d numpy arrays
  int64 N = instructions.size();
  int64 M = pass_context::GetInt("auto_sharding::memory_budget_per_device", 1 << 30);
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
  for (size_t i = 0; i < instructions.size(); ++i) {
    const HloInstruction* ins = instructions[i];
    const std::vector<ShardingStrategy>& strategies = strategy_map.at(ins);
    if (s_follow_np[i] < 0) {
      for (size_t j = 0; j < strategies.size(); ++j) {
        c_np.push_back(strategies[j].compute_cost);
        d_np.push_back(strategies[j].communication_cost);
        m_np.push_back(strategies[j].memory_cost);
      }
    } else {
      std::vector<int> reindexing = cost_graph.reindexing_vector.at(i);
      s_len_np[i] = reindexing.size();
      for (size_t k = 0; k < reindexing.size(); ++k) {
        size_t j = reindexing[k];
        c_np.push_back(strategies[j].compute_cost);
        d_np.push_back(strategies[j].communication_cost);
        m_np.push_back(strategies[j].memory_cost);
      }
    }
  }

  // Serialize special edges that forces a alias pair have the same sharding spec
  std::vector<int> A_np;
  std::vector<double> v_np;
  for (const auto& pair : alias_set) {
    const std::vector<ShardingStrategy>& src_strategies =
      strategy_map.at(instructions[pair.first]);
    const std::vector<ShardingStrategy>& dst_strategies =
      strategy_map.at(instructions[pair.second]);

    Matrix raw_cost(src_strategies.size(), dst_strategies.size());
    for (size_t i = 0; i < src_strategies.size(); ++i) {
      for (size_t j = 0; j < dst_strategies.size(); ++j) {
        if (src_strategies[i].output_sharding == dst_strategies[j].output_sharding) {
          raw_cost(i, j) = 0.0;
        } else {
          raw_cost(i, j) = 1.0;
        }
      }
    }

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
  for (size_t i = 0; i < N; ++i) {
    if (filter_func(i)) {
      L_np.push_back(liveness_set[i].size());
    } else {
      L_np.push_back(0);
    }
  }
  for (size_t i = 0; i < N; ++i) {
    if (filter_func(i)) {
      for (const HloValue* value : liveness_set[i]) {
        L_np.push_back(ins_id_map.at(value->instruction()));
      }
    }
  }

  // Call the solver function in python
  size_t num_edges = E_np.size() / 2;
  std::vector<int> s_val, e_val;

  PyGILState_STATE gstate = PyGILState_Ensure();
  {
    py::object submodule = py::module_::import("parax.auto_sharding");
    py::object call_solver_serialized_args =
      submodule.attr("call_solver_serialized_args");
    py::object ret = call_solver_serialized_args(
      N, M,
      py::array(s_len_np.size(), s_len_np.data()), // TODO: avoid this copy
      py::array(s_follow_np.size(), s_follow_np.data()),
      py::array(E_np.size(), E_np.data()),
      py::array(A_np.size(), A_np.data()),
      py::array(L_np.size(), L_np.data()),
      py::array(c_np.size(), c_np.data()),
      py::array(d_np.size(), d_np.data()),
      py::array(m_np.size(), m_np.data()),
      py::array(r_np.size(), r_np.data()),
      py::array(v_np.size(), v_np.data())
    );
    if (ret.is_none()) {
      PyGILState_Release(gstate);
      exit(-1);
    }
    py::tuple tuple_ret = ret;

    py::object s_val_obj = tuple_ret[0], e_val_obj = tuple_ret[1];
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

  return std::make_pair(std::move(s_val), std::move(e_val));
}

// Print liveness set for debugging
std::string PrintLivenessSet(const LivenessSet& liveness_set) {
  std::ostringstream os;
  os << "Liveness Set" << std::endl;
  for (size_t i = 0; i < liveness_set.size(); ++i) {
    std::vector<std::string> names;
    for (const HloValue* value: liveness_set[i]) {
      names.push_back(value->instruction()->name() + value->index().ToString());
    }
    std::sort(names.begin(), names.end());

    std::string line;
    for (const std::string& name: names) {
       absl::StrAppendFormat(&line, "%s, ", name);
    }
    os << "Time " << i << ": " << line << std::endl;
  }
  return os.str();
}

// Print sorted instructions
std::string PrintInstructions(const HloInstructionSequence& sequence) {
  std::ostringstream os;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << "Instruction " << i << ": " << instructions[i]->ToString() << "\n";
  }
  return os.str();
}

// Print strategy map for debugging
std::string PrintStrategyMap(
  const StrategyMap& strategy_map,
  const HloInstructionSequence& sequence
) {
  std::ostringstream os;
  os << "Strategy Map" << std::endl;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << "Instruction " << i << ": " << instructions[i]->ToString() << "\n";
    for (const auto& strategy: strategy_map.at(instructions[i])) {
      os << "Strategy " << strategy.name << ", " << strategy.compute_cost << ", "
         << strategy.communication_cost << ", " << strategy.memory_cost << "\n";

      for (const auto& cost_vector: strategy.resharding_costs) {
        os << "[";
        for (double cost: cost_vector) {
          os << cost << ", ";
        }
        os << "]\n";
      }
    }
  }
  return os.str();
}

// Print auto sharding strategy for debugging
std::string PrintAutoShardingSolution(
  const HloInstructionSequence& sequence,
  const LivenessSet& liveness_set,
  const StrategyMap& strategy_map,
  const CostGraph& cost_graph,
  const InstructionIdMap& ins_id_map,
  const std::vector<int>& s_val,
  const std::vector<int>& e_val
) {
  std::ostringstream os;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  size_t N = instructions.size();

  // Print the choosen strategy
  os << "=== Auto sharding Strategy ===\n";
  for (size_t i = 0; i < N; ++i) {
    os << i << " " << instructions[i]->ToString(HloPrintOptions::ShortParsable())
       << "  ";
    int stra_idx = cost_graph.RemapIndex(i, s_val[i]);
    if (cost_graph.follow_idx[i] < 0) {
      os << strategy_map.at(instructions[i])[stra_idx].name << "\n";
    } else {
      os << strategy_map.at(instructions[i])[stra_idx].name << " follow "
         << cost_graph.follow_idx[i] << "\n";
    }
  }

  // Print the memory usage
  os << "=== Memory ===\n";
  for (size_t i = 0; i < N; ++i) {
    double mem = 0.0;
    for (const auto& val : liveness_set.at(i)) {
      const HloInstruction* ins = val->instruction();
      int ins_idx = ins_id_map.at(ins);
      int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
      const ShardingStrategy& strategy = strategy_map.at(ins)[stra_idx];
      mem += strategy.memory_cost;
    }
    os << "Time " << i << ": " << mem / (1024 * 1024) << " MB\n";
  }

  return os.str();
}

StatusOr<bool> AutoSharding::Run(HloModule* module) {
  if (!pass_context::GetBool("auto_sharding::enable", false)) {
    return false;
  }

  AutoShardingSolverOption solver_option;
  solver_option.force_data_parallel = false;
  solver_option.override_all_reduce_cost = false;
  solver_option.override_reduce_scatter_cost = false;

  std::string strategy_name = pass_context::GetString("auto_sharding::solver_strategy",
                                                      "normal");
  if (strategy_name == "normal") {
    ;
  } else if (strategy_name == "force_data_parallel") {
    solver_option.force_data_parallel = true;
    solver_option.override_all_reduce_cost = true;
    solver_option.all_reduce_cost = 1000;
  } else if (strategy_name == "force_zero_data_parallel") {
    solver_option.force_data_parallel = true;
    solver_option.override_reduce_scatter_cost = true;
    solver_option.reduce_scatter_cost = 1000;
  }

  //std::cerr << "===== Enter AutoSharding =====" << std::endl;
  //std::cerr << module->ToString();
  //std::cerr << "=====================================" << std::endl;

  // ----- Get a sequential schedule and do liveness analysis -----
  auto size_fn = [](const BufferValue& buffer) {
    return static_cast<double>(ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8));
  };
  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      ScheduleModule(module, size_fn,
                      ComputationSchedulerToModuleScheduler(DFSMemoryScheduler)));
  const HloComputation* entry_computation = module->entry_computation();
  std::unique_ptr<HloAliasAnalysis> alias_analysis =
    HloAliasAnalysis::Run(module).ConsumeValueOrDie();
  TF_ASSIGN_OR_RETURN(
    std::unique_ptr<HloLiveRange> hlo_live_range,
    HloLiveRange::Run(schedule, *alias_analysis, entry_computation));

  absl::flat_hash_map<const HloValue*, HloLiveRange::TimeBound>& buffer_live_ranges =
    hlo_live_range->buffer_live_ranges();

  LivenessSet liveness_set(hlo_live_range->schedule_end_time() + 1);
  for (const auto& iter: buffer_live_ranges) {
    for (int64 i = iter.second.start; i <= iter.second.end; ++i) {
      liveness_set[i].push_back(iter.first);
    }
  }
  //std::cerr << hlo_live_range->ToString() << std::endl;
  //std::cerr << PrintLivenessSet(liveness_set);

  // ----- Analyze forward/backward for force_data_parallel
  const HloInstructionSequence& sequence = hlo_live_range->flattened_instruction_sequence();
  InstructionIdMap ins_id_map = BuildInstructionIdMap(sequence);
  if (solver_option.force_data_parallel) {
    int forward_backward_sep_id = EstimateForwardBackwardSep(
      module, sequence, ins_id_map);
    solver_option.forward_backward_sep_id = forward_backward_sep_id;
    //std::cerr << PrintInstructions(sequence) << std::endl;;
    //std::cerr << "Forward/backward sep id: " << forward_backward_sep_id << std::endl;
  }

  // ----- Build strategies and costs -----
  int num_devices = module->config().num_partitions();
  ClusterEnvironment cluster_env(num_devices, solver_option);
  StrategyMap strategy_map;
  FollowMap follow_map;
  std::tie(strategy_map, follow_map) = BuildStrategyAndCost(sequence, cluster_env, solver_option);
  AliasSet alias_set = BuildAliasSet(module, alias_analysis->dataflow_analysis(), ins_id_map);
  //std::cerr << PrintStrategyMap(strategy_map, sequence);

  // ----- Build cost graph and merge unimporant nodes -----
  CostGraph cost_graph(sequence, strategy_map, follow_map, ins_id_map);
  cost_graph.Simplify();

  // ----- Call the ILP Solver -----
  std::vector<int> s_val, e_val;
  std::tie(s_val, e_val) = CallSolver(module, sequence, liveness_set,
                                      strategy_map, cost_graph, ins_id_map, alias_set);
  if (pass_context::GetBool("auto_sharding::print_strategy", false)) {
    std::cerr << PrintAutoShardingSolution(sequence, liveness_set, strategy_map,
                                           cost_graph, ins_id_map, s_val, e_val);
  }

  // ----- Set Sharding -----
  HloComputation* entry = module->entry_computation();
  HloInstruction* root_inst = entry->root_instruction();

  // Set sharding for inputs and intermdiates
  for (HloInstruction* inst : entry->instructions()) {
    if (inst == root_inst) {
      continue;
    }

    auto iter = strategy_map.find(inst);
    if (iter != strategy_map.end()) {
      int ins_idx = ins_id_map[inst];
      int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
      inst->set_sharding(iter->second[stra_idx].output_sharding);
    }
  }

  // set sharding for outputs
  std::vector<HloSharding> shardings;
  const Shape& out_shape = entry->root_instruction()->shape();
  ShapeTree<HloSharding> tuple_sharding(out_shape, HloSharding::Replicate());
  for (int i = 0; i < out_shape.tuple_shapes_size(); ++i) {
    const HloInstruction* operand = root_inst->operand(i);
    int ins_idx = ins_id_map[operand];
    int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
    *tuple_sharding.mutable_element({i}) =
      strategy_map[operand][stra_idx].output_sharding;
  }
  entry->root_instruction()->set_sharding(HloSharding::Tuple(tuple_sharding));

  //std::cerr << "===== Exit AutoSharding =====" << std::endl;
  //std::cerr << module->ToString();
  //std::cerr << "=====================================" << std::endl;

  return true;
}


}  // namespace gpu
}  // namespace xla
