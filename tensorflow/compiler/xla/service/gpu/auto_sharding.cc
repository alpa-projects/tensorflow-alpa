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
#include "tensorflow/compiler/xla/shape_util.h"


namespace xla {
namespace gpu {

namespace py = pybind11;

double GetBytes(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
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
using AliasSet = absl::flat_hash_set<std::pair<size_t, size_t>>;

// Cluster environment to model the communication cost
class ClusterEnvironment {
 public:
  ClusterEnvironment(int num_devices) : num_devices(num_devices) {}

  double AllReduceCost(double num_bytes) const {
    return alpha + \
           beta * 2 * (num_devices - 1) / num_devices * num_bytes + \
           0.01;
  }

  double AllGatherCost(double num_bytes) const {
    return alpha + \
           beta * (num_devices - 1) / num_devices * num_bytes + \
           0.001;
  }

  double ReshardingCost(const Shape& shape, const HloSharding& src_sharding,
                        const HloSharding& dst_sharding) const {
    if (src_sharding == dst_sharding) {
      return 0;
    }
    if (src_sharding.IsReplicated()) {
      return 0;
    }
    return AllGatherCost(GetBytes(shape));
  }

  double alpha = 1.0;
  double beta = 1.0;
  int num_devices;
};

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

// Get the space dimentions of a dot instruction
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
StrategyMap BuildStrategyAndCost(
  const HloInstructionSequence& sequence,
  const ClusterEnvironment& cluster_env
) {
  absl::flat_hash_map<const HloInstruction*, std::vector<ShardingStrategy>> ret;

  for (const auto& ins: sequence.instructions()) {
    std::vector<ShardingStrategy> strategies;
    switch (ins->opcode()) {
      case HloOpcode::kParameter: {
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
        strategies.push_back(
          ShardingStrategy({"R", HloSharding::Replicate(),
                            0, 0, GetBytes(ins->shape()),
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

        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          std::vector<double> resharding_costs;

          auto it = absl::c_find(dimensions, i);
          if (it == dimensions.end()) {
            resharding_costs = ReshardingCostVector(
              ret[operand], operand->shape(),
              HloSharding::Replicate(), cluster_env);
          } else {
            int64 original_dim = std::distance(dimensions.begin(), it);
            resharding_costs = ReshardingCostVector(
              ret[operand], operand->shape(),
              Split(operand->shape(), original_dim, cluster_env), cluster_env);
          }

          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {std::move(resharding_costs)}}));
        }

        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(ret[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env)}}));
        break;
      }

      case HloOpcode::kTranspose: {
        const auto& dimensions = ins->dimensions();
        const HloInstruction* operand = ins->operand(0);

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
            {ReshardingCostVector(ret[operand], operand->shape(),
                  Split(operand->shape(), original_dim, cluster_env), cluster_env)}}));
        }

        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(ret[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env)}}));
        break;
      }

      case HloOpcode::kReshape: {
        const HloInstruction* operand = ins->operand(0);
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

        // Register strategies
        for (size_t i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);
          std::vector<double> resharding_costs;

          auto it = dim_mapping.find(i);
          if (it == dim_mapping.end()) {
            resharding_costs = ReshardingCostVector(
              ret[operand], operand->shape(),
              HloSharding::Replicate(), cluster_env);
          } else {
            int64 original_dim = it->second;
            resharding_costs = ReshardingCostVector(
              ret[operand], operand->shape(),
              Split(operand->shape(), original_dim, cluster_env), cluster_env);
          }

          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {std::move(resharding_costs)}}));
        }

        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(ret[operand], operand->shape(),
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
        for (int64 i = 0; i < ins->shape().rank(); ++i) {
          if (ins->shape().dimensions(i) < cluster_env.num_devices) {
            continue;
          }

          std::string name = "S" + std::to_string(i);

          std::vector<std::vector<double>> resharding_costs;
          for (size_t j = 0; j < ins->operand_count(); ++j) {
            const HloInstruction* operand = ins->operand(j);
            resharding_costs.push_back(
              ReshardingCostVector(ret[operand], operand->shape(),
                                   Split(operand->shape(), i, cluster_env),
                                   cluster_env));
          }
          strategies.push_back(
            ShardingStrategy({
            name, Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            resharding_costs}));
        }

        std::vector<std::vector<double>> resharding_costs;
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          resharding_costs.push_back(
            ReshardingCostVector(ret[operand], operand->shape(),
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
            {ReshardingCostVector(ret[operand], operand->shape(),
                  Split(operand->shape(), original_dim, cluster_env), cluster_env),
             ReshardingCostVector(ret[unit], unit->shape(),
                  HloSharding::Replicate(), cluster_env)}}));
        }

        strategies.push_back(
          ShardingStrategy({
          "R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {ReshardingCostVector(ret[operand], operand->shape(),
                                HloSharding::Replicate(), cluster_env),
           ReshardingCostVector(ret[unit], unit->shape(),
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

        // split the space dim of lhs
        strategies.push_back(
          ShardingStrategy({
          "Sl = Sl x R", Split(ins->shape(), space_base_dim, cluster_env),
          0, 0, GetBytes(ins->shape()) / cluster_env.num_devices, 
          {
            ReshardingCostVector(ret[lhs], lhs->shape(),
                                 Split(lhs->shape(), lhs_space_dims[0], cluster_env), cluster_env),
            ReshardingCostVector(ret[rhs], rhs->shape(), HloSharding::Replicate(), cluster_env),
          }}));

        // split the space dim of rhs
        strategies.push_back(
          ShardingStrategy({
          "Sr = R x Sr", Split(ins->shape(), space_base_dim + 1, cluster_env),
          0, 0, GetBytes(ins->shape()) / cluster_env.num_devices, 
          {
            ReshardingCostVector(ret[lhs], lhs->shape(), HloSharding::Replicate(), cluster_env),
            ReshardingCostVector(ret[rhs], rhs->shape(),
                                 Split(rhs->shape(), rhs_space_dims[0], cluster_env), cluster_env),
          }}));

        // split the contracting dim
        strategies.push_back(
          ShardingStrategy({
          "R = Sk x Sk", HloSharding::Replicate(),
          0, cluster_env.AllReduceCost(GetBytes(ins->shape())), GetBytes(ins->shape()),
          {
            ReshardingCostVector(ret[lhs], lhs->shape(),
              Split(lhs->shape(), dot_dnums.lhs_contracting_dimensions(0), cluster_env), cluster_env),
            ReshardingCostVector(ret[rhs], rhs->shape(),
              Split(rhs->shape(), dot_dnums.rhs_contracting_dimensions(0), cluster_env), cluster_env),
          }}));

        // split the batch dim
        for (size_t i = 0; i < dot_dnums.lhs_batch_dimensions_size(); ++i) {
          strategies.push_back(
            ShardingStrategy({
            "R = Sb x Sb " + std::to_string(i), Split(ins->shape(), i, cluster_env),
            0, 0, GetBytes(ins->shape()) / cluster_env.num_devices,
            {
              ReshardingCostVector(ret[lhs], lhs->shape(),
                Split(lhs->shape(), dot_dnums.lhs_batch_dimensions(i), cluster_env), cluster_env),
              ReshardingCostVector(ret[rhs], rhs->shape(),
                Split(rhs->shape(), dot_dnums.rhs_batch_dimensions(i), cluster_env), cluster_env),
            }}));
        }
 
        // replicate all
        strategies.push_back(
          ShardingStrategy({
          "R = R x R", HloSharding::Replicate(),
          0, 0, GetBytes(ins->shape()),
          {
            ReshardingCostVector(ret[lhs], lhs->shape(), HloSharding::Replicate(), cluster_env),
            ReshardingCostVector(ret[rhs], rhs->shape(), HloSharding::Replicate(), cluster_env),
          }}));

        break;
      }
      case HloOpcode::kTuple: {
        std::vector<std::vector<double>> resharding_costs;

        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          resharding_costs.push_back(std::vector<double>(ret[operand].size(), 0));
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

    ret[ins] = strategies;
  }

  return ret;
}

InstructionIdMap BuildInstructionIdMap(const HloInstructionSequence& sequence) {
  InstructionIdMap ret;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  for (size_t i = 0; i < instructions.size(); ++i) {
    ret[instructions[i]] = i;
  }
  return ret;
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

std::pair<std::vector<int>, std::vector<int>> CallSolver(
  const HloModule* module,
  const HloInstructionSequence& sequence,
  const LivenessSet& liveness_set,
  const StrategyMap& strategy_map,
  const InstructionIdMap& ins_id_map,
  const AliasSet& alias_set
) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Serialize edges and edge costs to 1d numpy arrays
  int64 N = instructions.size();
  int64 M = module->config().memory_budget_per_device();
  std::vector<int> s_len_np;
  std::vector<int> E_np;
  std::vector<double> r_np;
  s_len_np.reserve(instructions.size());
  for (const auto& ins : instructions) {
    s_len_np.push_back(strategy_map.at(ins).size());

    for (const auto& strategy : strategy_map.at(ins)) {
      CHECK_EQ(strategy.resharding_costs.size(), ins->operand_count());
    }

    for (size_t i = 0; i < ins->operand_count(); ++i) {
      const HloInstruction* src = ins->operand(i);
      const HloInstruction* dst = ins;

      size_t src_idx = ins_id_map.at(src);
      size_t dst_idx = ins_id_map.at(dst);

      if (alias_set.count(std::make_pair(src_idx, dst_idx))) {
        // If the edge is an alias, its costs will be set later.
        continue;
      }

      E_np.push_back(src_idx);
      E_np.push_back(dst_idx);

      // Need a transpose, because the resharding cost is stored at dst.
      std::vector<const std::vector<double>*> r_tmp;
      for (const auto& strategy : strategy_map.at(dst)) {
        r_tmp.push_back(&strategy.resharding_costs[i]);
      }
      CHECK_GT(r_tmp.size(), 0);
      for (size_t p = 0; p < r_tmp[0]->size(); ++p) {
        for (size_t q = 0; q < r_tmp.size(); ++q) {
          r_np.push_back((*r_tmp[q])[p]);
        }
      }
    }
  }

  // Serialize special edges that forces a alias pair have the same sharding spec
  for (const auto& pair : alias_set) {
    E_np.push_back(pair.first);
    E_np.push_back(pair.second);
    const std::vector<ShardingStrategy>& src_strategies =
      strategy_map.at(instructions[pair.first]);
    const std::vector<ShardingStrategy>& dst_strategies =
      strategy_map.at(instructions[pair.second]);

    for (const auto& src_strategy : src_strategies) {
      for (const auto& dst_strategy : dst_strategies) {
        if (src_strategy.output_sharding == dst_strategy.output_sharding) {
          r_np.push_back(0.0);
        } else {
          r_np.push_back(1e20);
        }
      }
    }
  }

  // Serialize liveness_set
  size_t num_edges = E_np.size() / 2;

  std::vector<int> L_np;
  for (size_t i = 0; i < N; ++i) {
    L_np.push_back(liveness_set[i].size());
  }
  for (size_t i = 0; i < N; ++i) {
    for (const HloValue* value : liveness_set[i]) {
      L_np.push_back(ins_id_map.at(value->instruction()));
    }
  }

  // Serialize node costs
  std::vector<double> c_np, d_np, m_np;
  for (const auto& ins : instructions) {
    for (const auto& strategy : strategy_map.at(ins)) {
      c_np.push_back(strategy.compute_cost);
      d_np.push_back(strategy.communication_cost);
      m_np.push_back(strategy.memory_cost);
    }
  }

  // Call the solver function in python
  std::vector<int> s_val, e_val;

  PyGILState_STATE gstate = PyGILState_Ensure();
  {
    py::object submodule = py::module_::import("paranum.auto_sharding");
    py::object call_solver_serialized_args =
      submodule.attr("call_solver_serialized_args");
    py::tuple ret = call_solver_serialized_args(
      N, M,
      py::array(s_len_np.size(), s_len_np.data()), // TODO: avoid this copy
      py::array(E_np.size(), E_np.data()),
      py::array(L_np.size(), L_np.data()),
      py::array(c_np.size(), c_np.data()),
      py::array(d_np.size(), d_np.data()),
      py::array(m_np.size(), m_np.data()),
      py::array(r_np.size(), r_np.data()));

    py::object s_val_obj = ret[0], e_val_obj = ret[1];
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
    os << instructions[i]->ToString(HloPrintOptions::ShortParsable()) << "  ";
    os << strategy_map.at(instructions[i])[s_val[i]].name << "\n";
  }

  // Print the memory usage
  os << "=== Memory ===\n";
  for (size_t i = 0; i < N; ++i) {
    double mem = 0.0;
    for (const auto& val : liveness_set.at(i)) {
      const HloInstruction* ins = val->instruction();
      const ShardingStrategy& strategy = strategy_map.at(ins)[s_val[ins_id_map.at(ins)]];

      mem += strategy.memory_cost;
    }
    os << "Time " << i << ": " << mem / (1024 * 1024) << " MB\n";
  }

  return os.str();
}

StatusOr<bool> AutoSharding::Run(HloModule* module) {
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

  // ----- Build strategies and costs -----
  int num_devices = module->config().num_partitions();
  ClusterEnvironment cluster_env(num_devices);

  const HloInstructionSequence& sequence = hlo_live_range->flattened_instruction_sequence();
  StrategyMap strategy_map = BuildStrategyAndCost(sequence, cluster_env);
  InstructionIdMap ins_id_map = BuildInstructionIdMap(sequence);
  AliasSet alias_set = BuildAliasSet(module, alias_analysis->dataflow_analysis(), ins_id_map);
  //std::cerr << PrintStrategyMap(strategy_map, sequence);

  // ----- Call the ILP Solver -----
  std::vector<int> s_val, e_val;
  std::tie(s_val, e_val) = CallSolver(module, sequence, liveness_set,
                                      strategy_map, ins_id_map, alias_set);
  //std::cerr << PrintAutoShardingSolution(sequence, liveness_set, strategy_map,
  //                                       ins_id_map, s_val, e_val);

  // ----- Set Sharding -----
  HloComputation* entry = module->entry_computation();
  HloInstruction* root_inst = entry->root_instruction();


  // Set sharding for inputs and intermdiates
  for (HloInstruction* inst: entry->instructions()) {
    if (inst == root_inst) {
      continue;
    }

    auto iter = strategy_map.find(inst);
    if (iter != strategy_map.end()) {
      inst->set_sharding(iter->second[s_val[ins_id_map[inst]]].output_sharding);
    }
  }

  // set sharding for outputs
  std::vector<HloSharding> shardings;
  const Shape& out_shape = entry->root_instruction()->shape();
  ShapeTree<HloSharding> tuple_sharding(out_shape, HloSharding::Replicate());
  for (int i = 0; i < out_shape.tuple_shapes_size(); ++i) {
    const HloInstruction* operand = root_inst->operand(i);
    *tuple_sharding.mutable_element({i}) =
      strategy_map[operand][s_val[ins_id_map[operand]]].output_sharding;
  }
  entry->root_instruction()->set_sharding(HloSharding::Tuple(tuple_sharding));

  //std::cerr << "===== Exit AutoSharding =====" << std::endl;
  //std::cerr << module->ToString();
  //std::cerr << "=====================================" << std::endl;

  return true;
}


}  // namespace gpu
}  // namespace xla
