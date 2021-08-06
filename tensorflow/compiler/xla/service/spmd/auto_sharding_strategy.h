#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

#include <vector>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"

namespace xla {
namespace spmd {

namespace py = pybind11;

// A constant to represent infinity cost.
constexpr double INFINITY_COST = 1e10;

// Options for the auto-sharding solver.
struct AutoShardingSolverOption {
  bool force_batch_dim_to_mesh_dim;

  // If true, override the cost of all-gather with the given value.
  bool override_all_gather_cost;
  double all_gather_cost;

  // If true, override the cost of all-reduce with the given value.
  bool override_all_reduce_cost;
  double all_reduce_cost;

  // If true, override the cost of reduce-scatter with the given value.
  bool override_reduce_scatter_cost;
  double reduce_scatter_cost;

  // If true, prefer reduce-scatter + all-gather over all-reduce
  bool prefer_reduce_scatter;

  // If ture, allow strategies that recompute heavy operators (e.g., dot)
  // to reduce communication.
  bool allow_recompute_heavy_op;

  // If true, load solution vector from PassContext
  bool load_solution_vector;
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

// Type aliases.
using LivenessSet = std::vector<std::vector<const HloValue*>>;
// Map an instruction to its strategy vector.
using StrategyMap =
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<StrategyVector>>;
// The list of all leaf strategies.
using LeafStrategies = std::vector<StrategyVector*>;
// The list of all dot instruction pairs that can be optimized by AllReduceReassociate pass.
using AssociativeDotPairs =
    std::vector<std::pair<const StrategyVector*, const StrategyVector*>>;
// Map an instruction to its depth.
using InstructionDepthMap = absl::flat_hash_map<const HloInstruction*, int64>;
// Map an instruction to its alias source parameter.
using AliasMap = absl::flat_hash_map<const HloInstruction*, const HloInstruction*>;
// The set of all alias pairs
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
        PyDictToCppDict(
            py::cast<py::dict>(prof_result.attr("all_reduce_cost_dict")),
            all_reduce_cost_dict_);
        PyDictToCppDict(
            py::cast<py::dict>(prof_result.attr("all_gather_cost_dict")),
            all_gather_cost_dict_);
        PyDictToCppDict(
            py::cast<py::dict>(prof_result.attr("reduce_scatter_cost_dict")),
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

  double EstimateAllGatherCost(
      const std::vector<std::vector<int>>& replica_groups, int64 size,
      std::string dtype) const {
    if (all_gather_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(
      const std::vector<std::vector<int>>& replica_groups, int64 size,
      std::string dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateReduceScatterCost(
      const std::vector<std::vector<int>>& replica_groups, int64 size,
      std::string dtype) const {
    if (reduce_scatter_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype,
                            reduce_scatter_cost_dict_);
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
  double EstimateInternal(
      const std::vector<std::vector<int>>& replica_groups, int64 size,
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
        if (cost_list[i].first <= size && size <= cost_list[i + 1].first) {
          break;
        }
      }
    }

    int64 left_size = cost_list[i].first;
    double left_cost = cost_list[i].second;
    int64 right_size = cost_list[i + 1].first;
    double right_cost = cost_list[i + 1].second;

    return 1.0 * (size - left_size) / (right_size - left_size) *
               (right_cost - left_cost) +
           left_cost;
  }

  // Convert a python dict to c++ dict.
  void PyDictToCppDict(py::dict py_dict,
                       absl::flat_hash_map<Key, Value>& cpp_dict) {
    // the type of py_dict: Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
    for (auto item : py_dict) {
      py::tuple tuple_key = py::cast<py::tuple>(item.first);
      Key key(Group2Str(py::cast<py::tuple>(tuple_key[0])),
              py::cast<std::string>(tuple_key[1]));

      py::list list_val = py::cast<py::list>(item.second);
      for (const auto x : list_val) {
        py::tuple tuple_val = py::cast<py::tuple>(x);
        cpp_dict[key].push_back(std::make_pair(py::cast<int64>(tuple_val[0]),
                                               py::cast<double>(tuple_val[1])));
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
  std::string Group2Str(
      const std::vector<std::vector<int>>& replica_groups) const {
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
      return prof_result.EstimateReduceScatterCost(
          cached_replica_groups[mesh_dim], num_bytes / 4, "float32");
    }

    int64 num_devices = device_mesh.dim(mesh_dim);
    return (mesh_alpha[mesh_dim] +
            mesh_beta[mesh_dim] * (num_devices - 1) / num_devices * num_bytes +
            0.001);
  }

  double DotCost(const Shape& lhs_shape, const Shape& rhs_shape,
                 const DotDimensionNumbers& dot_dnums) const {
    if (!solver_option.allow_recompute_heavy_op) {
      return INFINITY_COST;
    }

    // TODO(lmzheng): When profiling data is not available, it is not easy to align the
    // scale of compute cost and communication cost. Here we just use some
    // a simple heurstic to compute the compute cost with communication cost.
    double num_bytes = GetBytes(lhs_shape) + GetBytes(rhs_shape);
    return AllReduceCost(num_bytes, 0) + AllReduceCost(num_bytes, 1);
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
    // TODO(lmzheng): This function can be wrong and needs more tests.
    if (src_spec == dst_spec || IsUndefined(src_spec)) {
      return 0.0;
    }
    CHECK(!IsUndefined(dst_spec));

    std::vector<int> src_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(shape, src_spec);
    std::vector<int> dst_tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(shape, dst_spec);

    int n_slice = 0;
    int n_all_gather = 0;

    double bytes = GetBytes(shape) / src_spec.NumTiles();
    double cost = 0.0;
    for (int64 i = 0; i < shape.rank(); ++i) {
      int src_mesh_dim = src_tensor_dim_to_mesh_dim[i];
      if (src_mesh_dim == dst_tensor_dim_to_mesh_dim[i]) {
        continue;
      }
      if (src_mesh_dim == -1) {
        n_slice++;
        continue;
      }
      if (dst_tensor_dim_to_mesh_dim[i] == -1) {
        n_all_gather++;
        bytes *= device_mesh.dim(src_mesh_dim);
        cost += AllGatherCost(bytes, src_mesh_dim);
        continue;
      }
      // Do not allow other re-sharding patterns.
      return INFINITY_COST;
    }

    if (n_slice >= 1 && n_all_gather >= 1) {
      // Do not allow some strange re-sharding patterns.
      return INFINITY_COST;
    }

    return cost;
  }

  // Print the information of this device mesh.
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

// Function declarations
// Their comments can be found in their definitions in *.cc files.
HloSharding Tile(const Shape& shape, const std::vector<int64> tensor_dims,
                 const std::vector<int64> mesh_dims,
                 const ClusterEnvironment& cluster_env);

std::vector<double> ReshardingCostVector(
    const StrategyVector* strategies, const Shape& shape,
    const HloSharding& required_sharding,
    const ClusterEnvironment& cluster_env);

std::vector<double> FollowInsCostVector(int64 source_len, int64 index);

std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, LeafStrategies& leaf_strategies);

void SetInNodesWithInstruction(std::unique_ptr<StrategyVector>& strategies,
                               const HloInstruction* ins,
                               const StrategyMap& strategy_map);

void HandleDot(std::unique_ptr<StrategyVector>& strategies,
               LeafStrategies& leaf_strategies,
               StrategyMap& strategy_map,
               const HloInstruction* ins,
               size_t instruction_id,
               const ClusterEnvironment& cluster_env,
               const AutoShardingSolverOption& solver_option);

HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& straetgy,
                                   const ClusterEnvironment& cluster_env);

}  // namespace spmd
}  // namespace xla
