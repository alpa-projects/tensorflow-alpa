#include "tensorflow/compiler/xla/service/gpu/auto_sharding.h"
#include "tensorflow/compiler/xla/service/gpu/auto_sharding_util.h"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/pass_context.h"

namespace xla {
namespace gpu {

namespace py = pybind11;

// A constant to represent infinity cost.
constexpr double INFINITY_COST = 1e10;

// Options for the auto-sharding solver.
struct AutoShardingSolverOption {
  bool force_batch_dim_to_mesh_dim;
  int64 forward_backward_sep_id;

  bool override_all_gather_cost;
  double all_gather_cost;

  bool override_all_reduce_cost;
  double all_reduce_cost;

  bool override_reduce_scatter_cost;
  double reduce_scatter_cost;

  bool load_strategy;
};

// One sharding strategy
struct ShardingStrategy {
  std::string name;
  HloSharding output_sharding;
  double compute_cost;
  double communication_cost;
  double memory_cost;
  // shape of resharding_costs: [#operands, #strategies of the operand]
  std::vector<std::vector<double>> resharding_costs;
};

// The strategy for each instructions.
// We use unique_ptr for ownership, and raw pointers for other references.
struct StrategyVector {
  bool is_tuple;
  // the index used in the solver. For non-leaf nodes, this is set to -1.
  int64 id;
  // the index of the HLO instruction that generates this strategy vector.
  size_t instruction_id;
  // the conneced nodes used for resharding costs;
  std::vector<const StrategyVector*> in_nodes;
  // the followed strategy. Used for merging nodes.
  const StrategyVector* following = nullptr;
  // Used when is_tuple == False. Leaf strategy vector.
  std::vector<ShardingStrategy> leaf_vector;
  // Used when is_tuple == True. A list of strategy vectors of child nodes.
  std::vector<std::unique_ptr<StrategyVector>> childs;
};

// Forward declerations and type aliases.
using LivenessSet = std::vector<std::vector<const HloValue*>>;
using StrategyMap =
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<StrategyVector>>;
using LeafStrategies = std::vector<StrategyVector*>;
using InstructionDepthMap = absl::flat_hash_map<const HloInstruction*, int64>;
using AliasSet = absl::flat_hash_set<std::pair<int64, int64>>;


// Store the profiling results of communication and computation.
class ProfilingResult {
 public:
  // Construct the class from the corresponding python object
  // parax/profile_communication.py::ProfilingResult .
  ProfilingResult(py::object prof_result) {
    if (!prof_result.is_none()) {

      PyGILState_STATE gstate = PyGILState_Ensure();
      {
        PyDictToCppDict(py::cast<py::dict>(prof_result.attr("all_reduce_cost_dict")),
                        all_reduce_cost_dict_);
        PyDictToCppDict(py::cast<py::dict>(prof_result.attr("all_gather_cost_dict")),
                        all_gather_cost_dict_);
        PyDictToCppDict(py::cast<py::dict>(prof_result.attr("reduce_scatter_cost_dict")),
                        reduce_scatter_cost_dict_);
      }
      PyGILState_Release(gstate);
    }

    if (all_reduce_cost_dict_.empty()) {
      enabled_ = false;
    } else {
      enabled_ = true;
    }
  }

  bool Enabled() const { return enabled_; }

  double EstimateAllGatherCost(const std::vector<std::vector<int>>& replica_groups,
                               int64 size, std::string dtype) const {
    if (all_gather_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype, all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(const std::vector<std::vector<int>>& replica_groups,
                               int64 size, std::string dtype) const {
    return EstimateInternal(replica_groups, size, dtype, all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateReduceScatterCost(const std::vector<std::vector<int>>& replica_groups,
                                   int64 size, std::string dtype) const {
    if (reduce_scatter_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype, reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, reduce_scatter_cost_dict_);
  }

  std::string ToString() {
    std::ostringstream os;
    for (const auto& item : all_reduce_cost_dict_) {
      os << item.first.first << " " << item.first.second << "\n";
    }
    return os.str();
  }

 private:
  // pair<group, dtype>
  using Key = std::pair<std::string, std::string>;
  // vector<pair<size, time>>
  using Value = std::vector<std::pair<int64, double>>;

  // Estimate the cost by linear interpolation bewteen the two closest points.
  double EstimateInternal(const std::vector<std::vector<int>>& replica_groups,
                          int64 size,
                          const std::string& dtype,
                          const absl::flat_hash_map<Key, Value>& cost_dict) const {
    Key key(Group2Str(replica_groups), dtype);
    Value cost_list = cost_dict.at(key);

    CHECK(!cost_list.empty());

    size_t i;
    if (size > cost_list.back().first) {
      i = cost_list.size() - 2;
    } else if (size < cost_list.front().first) {
      i = 0;
    } else {
      for (i = 0; i < cost_list.size() - 1; ++i) {
        if (cost_list[i].first <= size && size <= cost_list[i+1].first) {
          break;
        }
      }
    }

    int64 left_size = cost_list[i].first;
    double left_cost = cost_list[i].second;
    int64 right_size = cost_list[i+1].first;
    double right_cost = cost_list[i+1].second;

    return 1.0 * (size - left_size) / (right_size - left_size) * (right_cost - left_cost) + left_cost;
  }

  // Convert a python dict to c++ dict.
  void PyDictToCppDict(py::dict py_dict, absl::flat_hash_map<Key, Value>& cpp_dict) {
    // the type of py_dict: Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
    for (auto item : py_dict) {
      py::tuple tuple_key = py::cast<py::tuple>(item.first);
      Key key(Group2Str(py::cast<py::tuple>(tuple_key[0])),
              py::cast<std::string>(tuple_key[1]));

      py::list list_val = py::cast<py::list>(item.second);
      for (const auto x : list_val) {
        py::tuple tuple_val = py::cast<py::tuple>(x);
        cpp_dict[key].push_back(std::make_pair(
          py::cast<int64>(tuple_val[0]), py::cast<double>(tuple_val[1])));
      }
    }
  }

  // Make a string key of a replica_groups.
  std::string Group2Str(const py::tuple& replica_groups) const {
    std::ostringstream os;
    os << "(";
    for (const auto& group : replica_groups) {
      os << "(";
      for (const auto& id : py::cast<py::tuple>(group)) {
        os << py::cast<int64>(id) << ",";
      }
      os << "),";
    }
    os << ")";

    return os.str();
  }

  // Make a string key of a replica_groups.
  std::string Group2Str(const std::vector<std::vector<int>>& replica_groups) const {
    std::ostringstream os;

    os << "(";
    for (const auto& group : replica_groups) {
      os << "(";
      for (const auto& id : group) {
        os << id << ",";
      }
      os << "),";
    }
    os << ")";

    return os.str();
  }

  bool enabled_;
  absl::flat_hash_map<Key, Value> all_reduce_cost_dict_;
  absl::flat_hash_map<Key, Value> all_gather_cost_dict_;
  absl::flat_hash_map<Key, Value> reduce_scatter_cost_dict_;
};

// The cluster has a multi-dimensional device mesh topology.
// Each mesh dimension has its own latency and bandwidth.
// We use alpha-beta model to model the communication cost.
// If profiling result is provided, we always prefer to use
// the real profiling result.
class ClusterEnvironment {
 public:
  ClusterEnvironment(const Array<int64>& device_mesh,
                     const std::vector<double>& mesh_alpha,
                     const std::vector<double>& mesh_beta,
                     const ProfilingResult& prof_result,
                     const AutoShardingSolverOption& solver_option)
      : device_mesh(device_mesh),
        total_devices(device_mesh.num_elements()),
        mesh_alpha(mesh_alpha),
        mesh_beta(mesh_beta),
        prof_result(prof_result),
        solver_option(solver_option) {
    // Build replica group for each dimension.
    CHECK_EQ(device_mesh.num_dimensions(), 2);

    // Replica groups when communicating across dim 0
    std::vector<std::vector<int>> replica_groups;
    for (size_t j = 0; j < device_mesh.dim(1); ++j) {
      std::vector<int> group;
      for (size_t i = 0; i < device_mesh.dim(0); ++i) {
        group.push_back(device_mesh(i, j));
      }
      replica_groups.push_back(std::move(group));
    }
    cached_replica_groups.push_back(replica_groups);

    // Replica groups when communicating across dim 1
    replica_groups.clear();
    for (size_t i = 0; i < device_mesh.dim(0); ++i) {
      std::vector<int> group;
      for (size_t j = 0; j < device_mesh.dim(1); ++j) {
        group.push_back(device_mesh(i, j));
      }
      replica_groups.push_back(std::move(group));
    }
    cached_replica_groups.push_back(replica_groups);
  }

  double AllGatherCost(double num_bytes, int mesh_dim) const {
    if (solver_option.override_all_gather_cost) {
      return solver_option.all_gather_cost;
    }

    if (prof_result.Enabled()) {
      return prof_result.EstimateAllGatherCost(cached_replica_groups[mesh_dim],
        num_bytes / 4, "float32");
    }

    int64 num_devices = device_mesh.dim(mesh_dim);
    return (mesh_alpha[mesh_dim] +
            mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
            0.1);
  }

  // TODO(lmzheng): distinguish dtype and reduce_op.
  double AllReduceCost(double num_bytes, int mesh_dim) const {
    if (solver_option.override_all_reduce_cost) {
      return solver_option.all_reduce_cost;
    }

    if (prof_result.Enabled()) {
      return prof_result.EstimateAllReduceCost(cached_replica_groups[mesh_dim],
        num_bytes / 4, "float32");
    }

    int64 num_devices = device_mesh.dim(mesh_dim);
    return (mesh_alpha[mesh_dim] +
            mesh_beta[mesh_dim] * 2 * (num_devices - 1) / num_devices *
                num_bytes +
            0.01);
  }

  double ReduceScatterCost(double num_bytes, int mesh_dim) const {
    if (solver_option.override_reduce_scatter_cost) {
      return solver_option.reduce_scatter_cost;
    }

    if (prof_result.Enabled()) {
      return prof_result.EstimateReduceScatterCost(cached_replica_groups[mesh_dim],
        num_bytes / 4, "float32");
    }

    int64 num_devices = device_mesh.dim(mesh_dim);
    return (mesh_alpha[mesh_dim] +
            mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
            0.001);
  }

  // Get the corresponding mesh dimension for every tensor dimension
  // -1 means replicated on that dimension
  std::vector<int> GetTensorDimToMeshDim(const Shape& shape,
                                         const HloSharding& spec) const {
    CHECK(shape.IsArray());
    CHECK(!IsUndefined(spec));

    if (spec.IsReplicated()) {
      return std::vector<int>(shape.rank(), -1);
    }

    std::vector<int> tensor_dim_vals(shape.rank(), 0);
    for (int64 i = 0; i < shape.rank(); ++i) {
      tensor_dim_vals[i] = GetDimLastValue(spec.tile_assignment(), i);
    }

    std::vector<int> mesh_dim_vals(device_mesh.num_dimensions(), 0);
    for (int64 j = 0; j < device_mesh.num_dimensions(); ++j) {
      mesh_dim_vals[j] = GetDimLastValue(device_mesh, j);
    }

    std::vector<int> ret(shape.rank(), -1);
    for (int64 i = 0; i < shape.rank(); ++i) {
      if (spec.tile_assignment().dim(i) != 1) {
        for (int64 j = 0; j < device_mesh.num_dimensions(); ++j) {
          if (tensor_dim_vals[i] == mesh_dim_vals[j]) {
            ret[i] = j;
          }
        }
      }
    }

    return ret;
  }

  // The communication cost of resharding a tensor from src to dst
  double ReshardingCost(const Shape& shape, const HloSharding& src_spec,
                        const HloSharding& dst_spec) const {
    if (src_spec == dst_spec || IsUndefined(src_spec)) {
      return 0.0;
    }
    CHECK(!IsUndefined(dst_spec));

    std::vector<int> src_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(shape, src_spec);
    std::vector<int> dst_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(shape, dst_spec);

    double cost = 0.0;
    for (int64 i = 0; i < shape.rank(); ++i) {
      int src_mesh_dim = src_tensor_dim_to_mesh_dim[i];
      if (src_mesh_dim == -1) {
        continue;
      }
      if (src_mesh_dim == dst_tensor_dim_to_mesh_dim[i]) {
        continue;
      }
      // TODO(lmzheng): this can be more accurate
      if (dst_tensor_dim_to_mesh_dim[i] == -1) {
        cost += AllGatherCost(GetBytes(shape), src_mesh_dim);
      }
      // do not allow other re-sharding strategies (e.g., collective-permute)
      return INFINITY_COST;
    }

    return cost;
  }

  std::string ToString() {
    std::ostringstream os;
    os << "device_mesh: " << device_mesh.ToString() << "\n";
    os << "mesh_alpha: ";
    for (auto x : mesh_alpha) {
      os << x << " ";
    }
    os << "\n";
    os << "mesh_beta: ";
    for (auto x : mesh_beta) {
      os << x << " ";
    }
    os << "\n";
    return os.str();
  }

  // Shape and bandwidth of the device mesh
  const Array<int64> device_mesh;
  const int total_devices;
  const std::vector<double> mesh_alpha;
  const std::vector<double> mesh_beta;
  const ProfilingResult& prof_result;

  // Disencourage the apperance of partial reduction
  const double partial_reduction_penalty = 10;

  // The solver option may override the cost of communication primitives
  const AutoShardingSolverOption& solver_option;

  // Cached replica groups. Shape: [mesh_dim, group_id, ids in this group].
  std::vector<std::vector<std::vector<int>>> cached_replica_groups;
};

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
  // std::cerr << "shape: " << shape.ToString() << std::endl;
  // std::cerr << "tensor dims: " << ToString(tensor_dims) << std::endl;
  // std::cerr << "mesh dims: " << ToString(mesh_dims) << std::endl;
  // std::cerr << "tile_assignment: " << ToString(tile_assignment.dimensions())
  // << std::endl;
  tile_assignment.SetValues(tile_assignment_devices);

  return replicate_on_last_tile_dim ? HloSharding::PartialTile(tile_assignment)
                                    : HloSharding::Tile(tile_assignment);
}

// Depth analysis (breadth first search).
// A heavey operator (e.g., dot, convolution) has a much larger distance.
InstructionDepthMap BuildInstructionDepthMap(
    const HloInstructionSequence& sequence) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  InstructionDepthMap depth_map;
  absl::flat_hash_map<const HloInstruction*, std::vector<const HloInstruction*>>
      edge_dict;
  absl::flat_hash_map<const HloInstruction*, size_t> degree_dict;

  for (const HloInstruction* inst : instructions) {
    for (int64 i = 0; i < inst->operand_count(); ++i) {
      degree_dict[inst] += 1;
      edge_dict[inst->operand(i)].push_back(inst);
    }
  }

  // Init frontier
  size_t collected = 0;
  std::vector<const HloInstruction*> current_frontier;
  for (const HloInstruction* inst : instructions) {
    if (degree_dict[inst] == 0) {
      depth_map[inst] = 0;
      current_frontier.push_back(inst);
      collected++;
    }
  }

  // Push forward
  std::vector<const HloInstruction*> next_frontier;
  while (collected < instructions.size()) {
    CHECK(!current_frontier.empty());
    next_frontier.clear();
    for (const HloInstruction* inst : current_frontier) {
      int delta = 0;

      // Heavy operators have more weight (distance).
      switch (inst->opcode()) {
        case HloOpcode::kConstant:
        case HloOpcode::kBroadcast:
          delta = 0;
          break;
        case HloOpcode::kDot:
        case HloOpcode::kConvolution:
          delta = 1000;
          break;
        default:
          delta = 1;
          break;
      }

      for (const HloInstruction* node : edge_dict[inst]) {
        int now_degree = --degree_dict[node];
        if (now_degree == 0) {
          next_frontier.push_back(node);
          depth_map[node] = depth_map[inst] + delta;
          collected += 1;
        }
      }
    }

    std::swap(current_frontier, next_frontier);
  }

  return depth_map;
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

// Filter strategies by name. This is used for debugging.
std::vector<ShardingStrategy> FilterStrategy(
    const std::vector<ShardingStrategy>& strategies,
    const std::vector<std::string>& names) {
  std::vector<ShardingStrategy> ret;
  for (const auto& strategy : strategies) {
    if (absl::c_linear_search(names, strategy.name)) {
      ret.push_back(strategy);
    }
  }
  return ret;
}

// Build possible sharding strategies and their costs for all instructions
std::pair<StrategyMap, LeafStrategies> BuildStrategyAndCost(
    const HloInstructionSequence& sequence,
    const InstructionDepthMap& depth_map, const ClusterEnvironment& cluster_env,
    const AutoShardingSolverOption& solver_option) {
  const Array<int64>& device_mesh = cluster_env.device_mesh;
  StrategyMap strategy_map;
  LeafStrategies leaf_strategies;
  absl::flat_hash_set<const HloInstruction*> undefined_set;

  const std::vector<HloInstruction*>& instructions = sequence.instructions();

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

            std::string name =
                "S" + std::to_string(i) + " @ " + std::to_string(j);
            HloSharding output_spec = Tile(ins->shape(), {i}, {j}, cluster_env);
            double compute_cost = 0, communication_cost = 0;
            double memory_cost = GetBytes(ins->shape()) / output_spec.NumTiles();
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
        if (undefined_set.count(operand)) { break; }

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
        // strategy. So we should register new strategies instead of following it.
        if (operand->shape().rank() == 0) {
          if (!output_set.count(ins) && operand->opcode() == HloOpcode::kConstant) {
            // one execption: always replicate intermidiate broadcasted constants.
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

              std::string name =
                  "S" + std::to_string(i) + " @ " + std::to_string(j);
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
        if (undefined_set.count(operand)) { break; }

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
        if (undefined_set.count(operand)) { break; }

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
        if (undefined_set.count(operand)) { break; }

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

        // Follow the deepest instruction
        int64 follow_idx = -1;
        int64 max_depth = -1;
        for (int64 i = 0; i < ins->operand_count(); ++i) {
          if (!undefined_set.count(ins->operand(i)) &&
              depth_map.at(ins->operand(i)) > max_depth) {
            follow_idx = i;
            max_depth = depth_map.at(ins->operand(i));
          }
        }
        if (follow_idx == -1) { break; }

        // Create follow strategies
        const HloInstruction* operand = ins->operand(follow_idx);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
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
        break;
      }
      case HloOpcode::kReduce: {
        strategies = CreateLeafStrategyVector(instruction_id, leaf_strategies);
        SetInNodesWithInstruction(strategies, ins, strategy_map);

        const HloInstruction* operand = ins->operand(0);
        const HloInstruction* unit = ins->operand(1);
        if (undefined_set.count(operand)) { break; }

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

    if (!strategies->is_tuple && strategies->leaf_vector.empty()) {
      // Set the strategy as "undefined".
      // Its sharding spec will be annotaed by the ShardingPropagation pass later.
      std::vector<std::vector<double>> resharding_costs;
      for (size_t i = 0; i < ins->operand_count(); ++i) {
        const HloInstruction* operand = ins->operand(i);
        const StrategyVector* src_strategies = strategy_map.at(operand).get();
        CHECK(!src_strategies->is_tuple);
        resharding_costs.push_back(std::vector<double>(src_strategies->leaf_vector.size(), 0));
      }
      strategies->leaf_vector.push_back(
        ShardingStrategy({"undefined", Undefined(), 0, 0, 0, resharding_costs}));
      undefined_set.insert(ins);
    }

    CHECK(strategies->is_tuple || !strategies->leaf_vector.empty());
    strategy_map[ins] = std::move(strategies);
  }

  return std::make_pair(std::move(strategy_map), std::move(leaf_strategies));
}

AliasSet BuildAliasSet(const HloModule* module,
                       const HloDataflowAnalysis& dataflow_analysis,
                       const StrategyMap& strategy_map) {
  // Adjust the edge cost for alias (donated buffer).
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
    const HloInstruction* dst_ins =
        dataflow_analysis.GetUniqueValueAt(output_tuple, output_index)
            .instruction();
    traverse_tuple_alias(strategy_map.at(src_ins).get(),
                         strategy_map.at(dst_ins).get());
  });

  return alias_set;
}

// A simple matrix class to store and manipulate on cost matrices on edges.
// It can create a view for transpose without copying the memory.
class Matrix {
 public:
  Matrix() : n(0), m(0), transpose(false), data(nullptr) {}

  Matrix(size_t n, size_t m) {
    this->n = n;
    this->m = m;
    transpose = false;
    data = std::make_shared<std::vector<double>>(n * m, 0.0);
  }

  Matrix(size_t n, size_t m, bool transpose,
         std::shared_ptr<std::vector<double>> data) {
    this->n = n;
    this->m = m;
    this->transpose = transpose;
    this->data = data;
  }

  Matrix Transpose() { return Matrix(m, n, !transpose, data); }

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
  CostGraph(const LeafStrategies& leaf_strategies) {
    node_lens.reserve(leaf_strategies.size());
    adjacency.assign(leaf_strategies.size(), absl::flat_hash_set<int>());

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
        to_merge_pairs.push_back({strategies->id, strategies->following->id});
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

    // std::cerr << "Merge: " << src << " to " << dst << std::endl;

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
        keys.push_back({edge_cost(i, j), -j});
      }

      std::sort(arange.begin(), arange.end(), [&keys](int l, int r) {
        return (keys[l].first < keys[r].first) ||
               (keys[l].first == keys[r].first &&
                keys[l].second < keys[r].second);
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
    const bool enable =
        pass_context::GetBool("auto_sharding::simplify_graph", true);

    // Merge nodes
    for (const auto& pair : to_merge_pairs) {
      int src = pair.first;
      int dst = pair.second;
      CHECK(!merged_to.count(src));
      dst = QueryDestination(dst);
      if (enable) {
        MergeNode(src, dst);
      }
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
      os << "Edge (" << iter.first.first << ", " << iter.first.second << "):\n";
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

// Serialize parameters of the ILP problem as numpy arrays and call the python solver.
std::pair<std::vector<int64>, std::vector<int64>> CallSolver(
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
    for (size_t i = 0; i < src_strategies->leaf_vector.size(); ++i) {
      for (size_t j = 0; j < dst_strategies->leaf_vector.size(); ++j) {
        if (src_strategies->leaf_vector[i].output_sharding ==
            dst_strategies->leaf_vector[j].output_sharding) {
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

// Set the HloSharding for all instructions according to the ILP solution.
void SetHloSharding(HloModule* module,
                    const StrategyMap& strategy_map,
                    const CostGraph& cost_graph,
                    const std::vector<int64> s_val) {
  HloComputation* entry = module->entry_computation();

  for (HloInstruction* inst : entry->instructions()) {
    auto iter = strategy_map.find(inst);
    if (iter != strategy_map.end()) {
      const StrategyVector* strategies = iter->second.get();
      if (strategies->is_tuple) {
        const Shape& out_shape = inst->shape();
        ShapeTree<HloSharding> tuple_sharding(out_shape,
                                              HloSharding::Replicate());
        std::function<void(const StrategyVector*)> get_flattened_shardings;
        std::vector<HloSharding> flattened_shardings;
        get_flattened_shardings = [&](const StrategyVector* strategies_) {
          if (strategies_->is_tuple) {
            for (const auto& child : strategies->childs) {
              get_flattened_shardings(child.get());
            }
          } else {
            int ins_idx = strategies_->id;
            int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
            flattened_shardings.push_back(
                strategies_->leaf_vector[stra_idx].output_sharding);
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
        int ins_idx = strategies->id;
        int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
        CHECK_LT(stra_idx, strategies->leaf_vector.size());
        const HloSharding& sharding_spec =
            strategies->leaf_vector[stra_idx].output_sharding;
        if (IsUndefined(sharding_spec)) {
          continue;
        }
        inst->set_sharding(sharding_spec);
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
                                      const std::vector<int64>& s_val) {
  std::ostringstream os;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();
  size_t N = leaf_strategies.size();

  // Print the choosen strategy
  os << "=== Auto sharding strategy ===\n";
  for (size_t i = 0; i < N; ++i) {
    os << i << " "
       << instructions[leaf_strategies[i]->instruction_id]->ToString(
              HloPrintOptions::ShortParsable())
       << "  ";
    int stra_idx = cost_graph.RemapIndex(i, s_val[i]);
    if (cost_graph.follow_idx[i] < 0) {
      os << leaf_strategies[i]->leaf_vector[stra_idx].name << "\n";
    } else {
      os << leaf_strategies[i]->leaf_vector[stra_idx].name << " follow "
         << cost_graph.follow_idx[i] << "\n";
    }
  }

  // Print the memory usage
  // os << "=== Memory ===\n";
  // for (size_t i = 0; i < N; ++i) {
  //  double mem = 0.0;
  //  for (const auto& val : liveness_set.at(i)) {
  //    const HloInstruction* ins = val->instruction();
  //    int ins_idx = ins_id_map.at(ins);
  //    int stra_idx = cost_graph.RemapIndex(ins_idx, s_val[ins_idx]);
  //    const ShardingStrategy& strategy = strategy_map.at(ins)[stra_idx];
  //    mem += strategy.memory_cost;
  //  }
  //  os << "Time " << i << ": " << mem / (1024 * 1024) << " MB\n";
  //}

  return os.str();
}

StatusOr<bool> AutoSharding::Run(HloModule* module) {
  if (!pass_context::GetBool("auto_sharding::enable", false)) {
    return false;
  }

  // ----- Set options for this pass -----
  AutoShardingSolverOption solver_option;
  solver_option.force_batch_dim_to_mesh_dim = false;
  solver_option.override_all_gather_cost = false;
  solver_option.override_all_reduce_cost = false;
  solver_option.override_reduce_scatter_cost = false;
  if (pass_context::GetBool("auto_sharding::force_all_gather_cost", false)) {
    solver_option.override_all_gather_cost = true;
    solver_option.all_gather_cost =
        pass_context::GetDouble("auto_sharding::all_gather_cost");
  }
  solver_option.load_strategy =
      pass_context::GetBool("auto_sharding::load_strategy", false);

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

  // ----- Analize depth -----
  const HloInstructionSequence& sequence =
      hlo_live_range->flattened_instruction_sequence();
  InstructionDepthMap ins_depth_map = BuildInstructionDepthMap(sequence);

  // ----- Read parameters of device mesh -----
  Array<int64> device_mesh(
      pass_context::GetIntVector("auto_sharding::device_mesh_shape"));
  device_mesh.SetValues(
      pass_context::GetIntVector("auto_sharding::device_mesh_ids"));

  ProfilingResult prof_result(pass_context::GetPyObject(
      "auto_sharding::device_mesh_prof_result"));
  ClusterEnvironment cluster_env(
      device_mesh,
      pass_context::GetDoubleVector("auto_sharding::device_mesh_alpha"),
      pass_context::GetDoubleVector("auto_sharding::device_mesh_beta"),
      prof_result,
      solver_option);

  // ----- Build strategies and costs -----
  StrategyMap strategy_map;
  LeafStrategies leaf_strategies;
  std::tie(strategy_map, leaf_strategies) =
      BuildStrategyAndCost(sequence, ins_depth_map, cluster_env, solver_option);
  AliasSet alias_set =
      BuildAliasSet(module, alias_analysis->dataflow_analysis(), strategy_map);
  // std::cerr << PrintStrategyMap(strategy_map, sequence);

  // ----- Build cost graph and merge unimporant nodes -----
  CostGraph cost_graph(leaf_strategies);
  cost_graph.Simplify();

  // ----- Call the ILP Solver -----
  std::vector<int64> s_val, e_val;
  if (!solver_option.load_strategy) {
    std::tie(s_val, e_val) = CallSolver(sequence, liveness_set, strategy_map,
                                        leaf_strategies, cost_graph, alias_set);
  } else {
    s_val = pass_context::GetIntVector("auto_sharding::strategy_vector");
  }

  if (pass_context::GetBool("auto_sharding::print_strategy", false)) {
    std::cerr << PrintAutoShardingSolution(sequence, liveness_set, strategy_map,
                                           leaf_strategies, cost_graph, s_val);
  }

  // ----- Set Sharding -----
  SetHloSharding(module, strategy_map, cost_graph, s_val);

  // std::cerr << "===== Exit AutoSharding =====" << std::endl;
  // std::cerr << module->ToString();
  // std::cerr << "=====================================" << std::endl;

  return true;
}

}  // namespace gpu
}  // namespace xla
