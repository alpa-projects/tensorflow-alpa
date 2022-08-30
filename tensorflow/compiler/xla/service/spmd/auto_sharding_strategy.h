#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_STRATEGY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_STRATEGY_H_

#include <cmath>
#include <vector>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

namespace xla {
namespace spmd {

namespace py = pybind11;

// A constant to represent infinity cost.
constexpr double INFINITY_COST = 1e13;

// Options for the auto-sharding solver.
struct AutoShardingSolverOption {
  // Forcibly split the batch dimension and map it to a mesh dimension.
  // This can force the auto-sharding pass to generate the data parallel
  // strategy.
  int force_batch_dim_to_mesh_dim;

  // If true, override the cost of all-gather with the given value.
  bool override_all_gather_cost;
  double all_gather_cost;

  // If true, override the cost of all-reduce with the given value.
  bool override_all_reduce_cost;
  double all_reduce_cost;

  // If true, override the cost of reduce-scatter with the given value.
  bool override_reduce_scatter_cost;
  double reduce_scatter_cost;

  // If true, override the cost of all-to-all with the given value.
  bool override_all_to_all_cost;
  double all_to_all_cost;

  // If true, allow replicated parameters.
  bool allow_replicated_parameters;

  // If true, prefer reduce-scatter + all-gather over all-reduce.
  // A post process will be applied to replace all-reduce with reduce-scater +
  // all-gather if no communication overhead is introduced.
  bool prefer_reduce_scatter;

  // If True, generate a gradient-accumulation friendly variant of
  // reduce-scatter
  bool reduce_scatter_grad_acc_friendly;

  // If true, aggressively partition more tensors when generating
  // reduce-scatter, even if it introduces more communication.
  bool reduce_scatter_aggressive_partition;

  // If true, the batch matmul will always be parallelized on the batch dim in
  // 2d mesh case.
  bool batch_matmul_always_split_batch;

  // If ture, allow strategies that recompute heavy operators (e.g., dot)
  // to reduce communication.
  bool allow_recompute_heavy_op;

  // If ture, allow adding 1d strategies in 2d logical mesh.
  bool allow_mixed_mesh_shape;

  // The number of micro batches if gradient accumulation is used.
  // If this is not 1, the cost of all-reduce for gradient synchronization
  // is divided by this number.
  int grad_acc_num_micro_batches;

  // If true, load solution vector from PassContext
  bool load_solution_vector;

  // If it is not empty, forcibly use simple heuristic strategies
  // instead of the ILP solver. This is used for ablation study.
  std::string force_simple_heuristic;
};

// One sharding strategy
struct ShardingStrategy {
  std::string name;
  HloSharding output_sharding;
  double compute_cost;
  double communication_cost;
  double memory_cost;
  // resharding_costs[i][j] is the resharding cost from the output of
  // i-th operand's j-th strategy to this strategy.
  std::vector<std::vector<double>> resharding_costs;
  // Optional: the required shardings of operands.
  // This is used to guide the SPMD partitioner.
  std::vector<HloSharding> input_shardings;
};

// The strategy for each instruction.
// We use unique_ptr for ownership, and raw pointers for other references.
struct StrategyVector {
  bool is_tuple;
  // the index used in the solver. For non-leaf nodes, this is set to -1.
  int64_t id;
  // the index of the HLO instruction that generates this strategy vector.
  size_t instruction_id;
  // the connected nodes used for resharding costs;
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
// The list of all dot instruction pairs that can be optimized by
// AllReduceReassociate pass.
using AssociativeDotPairs =
    std::vector<std::pair<const StrategyVector*, const StrategyVector*>>;
// The set of all alias pairs
using AliasSet = absl::flat_hash_set<std::pair<int64_t, int64_t>>;

// Store the profiling results of communication and computation.
class ProfilingResult {
 public:
  // Construct the class from the corresponding python object
  // alpa/mesh_profiling.py::ProfilingResult.
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
      const std::vector<std::vector<int>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    if (all_gather_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(
      const std::vector<std::vector<int>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateReduceScatterCost(
      const std::vector<std::vector<int>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    if (reduce_scatter_cost_dict_.empty()) {
      // Use all-reduce to approximate reduce-scatter.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype,
                            reduce_scatter_cost_dict_);
  }

  double EstimateAllToAllCost(
      const std::vector<std::vector<int>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    // A penalty factor to make the theoretical cost match the
    // empirical cost on v100 + nvlink.
    int64_t num_devices = replica_groups.front().size();
    double penalty_factor = double(num_devices) / 2.0;
    // Use all-gather to approximate all-to-all.
    return EstimateAllGatherCost(replica_groups, size / num_devices, dtype) *
           penalty_factor;
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
  using Value = std::vector<std::pair<int64_t, double>>;

  // Estimate the cost by linear interpolation between the two closest points.
  double EstimateInternal(
      const std::vector<std::vector<int>>& replica_groups, int64_t size,
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

    int64_t left_size = cost_list[i].first;
    double left_cost = cost_list[i].second;
    int64_t right_size = cost_list[i + 1].first;
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
        cpp_dict[key].push_back(std::make_pair(py::cast<int64_t>(tuple_val[0]),
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
        os << py::cast<int64_t>(id) << ",";
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
  ClusterEnvironment(const Array<int64_t>& device_mesh,
                     const std::vector<double>& mesh_alpha,
                     const std::vector<double>& mesh_beta,
                     const ProfilingResult& prof_result,
                     const AutoShardingSolverOption& solver_option)
      : device_mesh(device_mesh),
        mesh_alpha(mesh_alpha),
        mesh_beta(mesh_beta),
        prof_result(prof_result),
        total_devices(device_mesh.num_elements()),
        device_mesh_1d(device_mesh),
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

    if (device_mesh.dim(0) > 1) {
      non_zero_mesh_dims.push_back(0);
    }
    if (device_mesh.dim(1) > 1) {
      non_zero_mesh_dims.push_back(1);
    }

    device_mesh_1d.Reshape({device_mesh.num_elements(), 1});
  }

  double AllGatherCost(double num_bytes, int mesh_dim) const {
    if (solver_option.override_all_gather_cost) {
      return solver_option.all_gather_cost;
    }

    if (prof_result.Enabled()) {
      return prof_result.EstimateAllGatherCost(cached_replica_groups[mesh_dim],
                                               num_bytes / 4, "float32");
    }

    if (solver_option.force_batch_dim_to_mesh_dim == mesh_dim) {
      // if data-parallel is forced on this dim, we only allow all-reduce
      // in this dimension.
      return INFINITY_COST;
    }

    int64_t num_devices = device_mesh.dim(mesh_dim);
    return (round(mesh_alpha[mesh_dim] + mesh_beta[mesh_dim] *
                                             (num_devices - 1) / num_devices *
                                             num_bytes) +
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

    int64_t num_devices = device_mesh.dim(mesh_dim);
    return (round(mesh_alpha[mesh_dim] + mesh_beta[mesh_dim] * 2 *
                                             (num_devices - 1) / num_devices *
                                             num_bytes) +
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

    int64_t num_devices = device_mesh.dim(mesh_dim);
    return (round(mesh_alpha[mesh_dim] + mesh_beta[mesh_dim] *
                                             (num_devices - 1) / num_devices *
                                             num_bytes) +
            0.001);
  }

  double AllToAllCost(double num_bytes, int mesh_dim) const {
    if (solver_option.override_all_to_all_cost) {
      return solver_option.all_to_all_cost;
    }

    if (prof_result.Enabled()) {
      return prof_result.EstimateAllToAllCost(cached_replica_groups[mesh_dim],
                                              num_bytes / 4, "float32");
    }

    if (solver_option.force_batch_dim_to_mesh_dim == mesh_dim) {
      // if data-parallel is forced on this dim, we only allow all-reduce
      // in this dimension.
      return INFINITY_COST;
    }

    // A penalty factor to make the theoretical cost match the
    // empirical cost on v100 + nvlink.
    int64_t num_devices = device_mesh.dim(mesh_dim);
    double penalty_factor = double(num_devices) / 2.0;
    return (round(mesh_alpha[mesh_dim] +
                  mesh_beta[mesh_dim] * (num_devices - 1) / num_devices /
                      num_devices * num_bytes * penalty_factor) +
            0.001);
  }

  double DotCost(const Shape& lhs_shape, const Shape& rhs_shape,
                 const DotDimensionNumbers& dot_dnums) const {
    if (!solver_option.allow_recompute_heavy_op) {
      return INFINITY_COST;
    }

    // TODO(lmzheng): When profiling data is not available, it is not easy to
    // align the scale of compute cost and communication cost. Here we just use
    // a simple heurstic to compute the compute cost with communication cost.
    double num_bytes = GetBytes(lhs_shape) + GetBytes(rhs_shape);
    return AllReduceCost(num_bytes, 0) + AllReduceCost(num_bytes, 1);
  }

  // Get the corresponding mesh dimension for every tensor dimension.
  // -1 means replicated on that dimension
  std::vector<int> GetTensorDimToMeshDim(const Shape& shape,
                                         const HloSharding& spec) const {
    std::vector<int> ret;
    int n_dim;
    std::tie(ret, n_dim) = GetTensorDimToMeshDimInternal(shape, spec);
    AdjustTensorMeshDimMapping(ret, n_dim);
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

    int src_n_dim, dst_n_dim;
    std::vector<int> src_tensor_dim_to_mesh_dim, dst_tensor_dim_to_mesh_dim;

    std::tie(src_tensor_dim_to_mesh_dim, src_n_dim) =
        GetTensorDimToMeshDimInternal(shape, src_spec);
    std::tie(dst_tensor_dim_to_mesh_dim, dst_n_dim) =
        GetTensorDimToMeshDimInternal(shape, dst_spec);

    if (src_n_dim != dst_n_dim && src_n_dim != -1 && dst_n_dim != -1) {
      return ReshardingCostMixedMeshShape(
          shape, src_spec, dst_spec, src_tensor_dim_to_mesh_dim,
          dst_tensor_dim_to_mesh_dim, src_n_dim, dst_n_dim);
    }

    AdjustTensorMeshDimMapping(src_tensor_dim_to_mesh_dim, src_n_dim);
    AdjustTensorMeshDimMapping(dst_tensor_dim_to_mesh_dim, dst_n_dim);

    // Analyze the dims that need to dynamic-sliced or all-gather.
    std::vector<int> slice_dims;
    std::vector<int> all_gather_dims;
    for (int64_t i = 0; i < shape.rank(); ++i) {
      int src_mesh_dim = src_tensor_dim_to_mesh_dim[i];
      int dst_mesh_dim = dst_tensor_dim_to_mesh_dim[i];
      if (src_mesh_dim == dst_mesh_dim) {
        continue;
      }
      if (src_mesh_dim == -1) {
        slice_dims.push_back(src_mesh_dim);
        continue;
      }
      if (dst_mesh_dim == -1) {
        all_gather_dims.push_back(src_mesh_dim);
        continue;
      }
      // Do not allow other re-sharding patterns. (e.g., collective-permute)
      return INFINITY_COST;
    }

    // Case 1: no communication is required. Only needs dynamic-slice.
    if (all_gather_dims.size() == 0) {
      return 0;
    }

    // Do not allow some strange re-sharding patterns.
    if (slice_dims.size() > 1 && all_gather_dims.size() > 1) {
      return INFINITY_COST;
    }

    // Case 2: all-to-all
    if (slice_dims.size() == 1 && all_gather_dims.size() == 1) {
      if (device_mesh.dim(0) > 1 && device_mesh.dim(1) > 1) {
        return INFINITY_COST;
      }

      double bytes = GetBytes(shape);
      return AllToAllCost(bytes, all_gather_dims.front());
    }

    // Case 3: all-gather
    double bytes = GetBytes(shape) / src_spec.NumTiles();
    double cost = 0.0;
    for (int dim : all_gather_dims) {
      if (dim >= device_mesh.num_dimensions()) {
        return INFINITY_COST;
      }
      bytes *= device_mesh.dim(dim);
      cost += AllGatherCost(bytes, dim);
    }
    return cost;
  }

  double ReshardingCostMixedMeshShape(
      const Shape& shape, const HloSharding& src_spec,
      const HloSharding& dst_spec, std::vector<int> src_tensor_dim_to_mesh_dim,
      std::vector<int> dst_tensor_dim_to_mesh_dim, int src_n_dim,
      int dst_n_dim) const {
    // The type, volume, and mesh dim of the required communications
    std::vector<int> comm_type;  // 0: slice,  1: all-to-all,  2: all-gather
    std::vector<double> comm_bytes;
    std::vector<double> comm_mesh_dim;

    // Generate required communication primitives.
    // lhs is the mesh with 2d shape and rhs is the mesh with 1d shape
    bool compatible = true;
    auto generate_comm =
        [&](const std::vector<int>& lhs_tensor_dim_to_mesh_dim,
            const std::vector<int>& rhs_tensor_dim_to_mesh_dim) {
          double bytes = GetBytes(shape) / total_devices;

          for (int i = 0; i < shape.rank(); ++i) {
            int lhs_mesh_dim = lhs_tensor_dim_to_mesh_dim[i];
            int rhs_mesh_dim = rhs_tensor_dim_to_mesh_dim[i];

            if (lhs_mesh_dim == 1 && rhs_mesh_dim == -1) {
              comm_type.push_back(1);  // all-to-all
              comm_bytes.push_back(
                  bytes);  // FIXME(lmzheng): this bytes is wrong
              comm_mesh_dim.push_back(1);
            } else if (lhs_mesh_dim == -1) {
              if (rhs_mesh_dim == -1) {
                ;  // do nothing
              } else {
                comm_type.push_back(0);  // slice
                comm_bytes.push_back(bytes);
                comm_mesh_dim.push_back(0);
              }
            } else if (lhs_mesh_dim == rhs_mesh_dim) {
              continue;
            } else {
              compatible = false;
              break;
            }
          }

          if (comm_type.empty()) {
            comm_type.push_back(0);  // slice
            comm_bytes.push_back(bytes);
            comm_mesh_dim.push_back(1);
          }
        };

    if (src_n_dim == 2) {
      generate_comm(src_tensor_dim_to_mesh_dim, dst_tensor_dim_to_mesh_dim);
    } else {
      generate_comm(dst_tensor_dim_to_mesh_dim, src_tensor_dim_to_mesh_dim);

      // Reverse communication pattern
      for (int i = 0; i < comm_type.size(); ++i) {
        if (comm_type[i] == 0) {  // if is slice, reverse it to all-gather
          comm_type[i] = 2;
        } else if (comm_type[i] ==
                   2) {  // if is all-gather, reverse it to slice
          comm_type[i] = 0;
        }
      }
    }

    // std::cerr << src_spec.ToString() << " " << dst_spec.ToString() <<
    // std::endl; std::cerr <<
    // ::xla::spmd::ToString<int>(src_tensor_dim_to_mesh_dim) << " "
    //          << ::xla::spmd::ToString<int>(dst_tensor_dim_to_mesh_dim)
    //          << std::endl;

    double ret = 0;
    if (compatible) {
      // Sum up communication cost
      int n_comm = 0;
      for (int i = 0; i < comm_type.size(); ++i) {
        if (comm_type[i] == 0) {  // slice
          ret += 0;
        } else if (comm_type[i] == 1) {  // all-to-all
          ret += AllToAllCost(comm_bytes[i], comm_mesh_dim[i]);
          n_comm += 1;
        } else if (comm_type[i] == 2) {  // all-gather
          ret += AllGatherCost(comm_bytes[i], comm_mesh_dim[i]);
          n_comm += 1;
        } else {
          LOG(FATAL) << "Invalid communication type";
        }
      }

      if (n_comm > 1) {
        // Currently, SPMD partitioner do not support all-to-all + all-gather;
        ret = INFINITY_COST;
      }
    } else {
      ret = INFINITY_COST;
    }

    return ret;
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
  const Array<int64_t> device_mesh;
  const std::vector<double> mesh_alpha;
  const std::vector<double> mesh_beta;
  const ProfilingResult& prof_result;
  std::vector<int> non_zero_mesh_dims;
  const int total_devices;

  // Cache a flatten 1d version of the device mesh.
  // Used for mixed mesh shape strategies.
  Array<int64_t> device_mesh_1d;

  // The solver option may override the cost of communication primitives
  const AutoShardingSolverOption& solver_option;

  // Cached replica groups. Shape: [mesh_dim, group_id, ids in this group].
  std::vector<std::vector<std::vector<int>>> cached_replica_groups;

 private:
  void AdjustTensorMeshDimMapping(std::vector<int>& mapping, int n_dim) const {
    // Shift the non-zero dim for 1d mesh
    if (n_dim == 1 && non_zero_mesh_dims.size() == 1) {
      for (int i = 0; i < mapping.size(); ++i) {
        if (mapping[i] == 0) {
          mapping[i] = non_zero_mesh_dims.front();
        }
      }
    }
  }
};

// A graph data structure to simplify the edge cost graph.
// It merges nodes and does path compression.
class CostGraph {
 public:
  CostGraph(const LeafStrategies& leaf_strategies,
            const AssociativeDotPairs& associative_dot_pairs) {
    node_lens.reserve(leaf_strategies.size());
    extra_node_costs.reserve(leaf_strategies.size());
    adjacency.assign(leaf_strategies.size(), absl::flat_hash_set<int>());

    // Build the cost graph
    for (const auto& strategies : leaf_strategies) {
      node_lens.push_back(strategies->leaf_vector.size());
      extra_node_costs.push_back(
          std::vector<double>(strategies->leaf_vector.size(), 0.0));

      for (size_t i = 0; i < strategies->in_nodes.size(); ++i) {
        size_t src_idx = strategies->in_nodes[i]->id;
        size_t dst_idx = strategies->id;

        Matrix edge_cost(node_lens[src_idx], node_lens[dst_idx]);
        for (size_t k = 0; k < strategies->leaf_vector.size(); ++k) {
          const ShardingStrategy& stra = strategies->leaf_vector[k];

          CHECK_EQ(node_lens[src_idx], stra.resharding_costs[i].size());
          CHECK_EQ(stra.resharding_costs.size(), strategies->in_nodes.size());

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

    // Adjust the edge costs for dot pairs that can be optimized by
    // AllReduceReassociate
    for (const auto& pair : associative_dot_pairs) {
      size_t src_idx = pair.first->id;
      size_t dst_idx = pair.second->id;

      if (node_lens[src_idx] != node_lens[dst_idx]) {
        continue;
      }

      Matrix edge_cost(node_lens[src_idx], node_lens[dst_idx]);
      for (size_t i = 0; i < node_lens[src_idx]; ++i) {
        if (leaf_strategies[src_idx]->leaf_vector[i].communication_cost > 0) {
          CHECK_FLOAT_EQ(
              leaf_strategies[src_idx]->leaf_vector[i].communication_cost,
              leaf_strategies[dst_idx]->leaf_vector[i].communication_cost);
          edge_cost(i, i) =
              -leaf_strategies[src_idx]->leaf_vector[i].communication_cost;
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
        for (int i = 0; i < node_lens[dst]; ++i) {
          extra_node_costs[dst][i] += edge_cost(i, reindexing[i]);
        }
      } else {
        Matrix added_edge_cost(node_lens[dst], node_lens[adj]);
        Matrix edge_cost_src_adj = GetEdgeCost(src, adj);

        for (int i = 0; i < node_lens[dst]; ++i) {
          for (int k = 0; k < node_lens[adj]; ++k) {
            added_edge_cost(i, k) = edge_cost_src_adj(reindexing[i], k);
          }
        }

        AddEdgeCost(dst, adj, added_edge_cost);
      }
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
  // The extra node costs introduced by merging nodes.
  std::vector<std::vector<double>> extra_node_costs;
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

// Get the final sharding strategy according to the ilp solution.
inline const ShardingStrategy& GetShardingStrategy_(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, const std::vector<int64_t>& s_val) {
  const StrategyVector* strategies = strategy_map.at(inst).get();
  if (strategies->is_tuple) {
    if (inst->opcode() == HloOpcode::kReduce) {
      strategies = strategies->childs[0].get();
    } else {
      LOG(FATAL) << "Unhandled instruction: " << inst->ToString();
    }
  }
  int node_idx = strategies->id;
  int stra_idx = cost_graph.RemapIndex(node_idx, s_val[node_idx]);
  return strategies->leaf_vector[stra_idx];
}

// An abbreviation of GetShardingStrategy_
#define GetShardingStrategy(inst) \
  GetShardingStrategy_((inst), strategy_map, cost_graph, s_val)

// Function declarations
// Their comments can be found in their definitions in *.cc files.
HloSharding Tile(const Shape& shape, const std::vector<int64_t> tensor_dims,
                 const std::vector<int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh);

std::vector<double> ReshardingCostVector(const StrategyVector* strategies,
                                         const Shape& shape,
                                         const HloSharding& required_sharding,
                                         const ClusterEnvironment& cluster_env);

std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, LeafStrategies& leaf_strategies);

void RemoveDuplicatedStrategy(std::unique_ptr<StrategyVector>& strategies);

void RemoveIndivisibleStrategies(std::unique_ptr<StrategyVector>& strategies,
                                 const Shape& shape);

Status FilterStrategy(const HloInstruction* ins,
                      std::unique_ptr<StrategyVector>& strategies,
                      const ClusterEnvironment& cluster_env,
                      const InstructionBatchDimMap& batch_map,
                      const AutoShardingSolverOption& solver_option);

Status HandleDot(std::unique_ptr<StrategyVector>& strategies,
                 LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                 const HloInstruction* ins, size_t instruction_id,
                 const ClusterEnvironment& cluster_env,
                 const InstructionBatchDimMap& batch_map,
                 const AutoShardingSolverOption& solver_option);

Status HandleConv(std::unique_ptr<StrategyVector>& strategies,
                  LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                  const HloInstruction* ins, size_t instruction_id,
                  const ClusterEnvironment& cluster_env,
                  const InstructionBatchDimMap& batch_map,
                  const AutoShardingSolverOption& solver_option);

void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const InstructionDepthMap& depth_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           const std::vector<int64_t>& s_val,
                           const ClusterEnvironment& cluster_env,
                           const AutoShardingSolverOption& solver_option);

void AnnotateShardingWithSimpleHeuristic(HloModule* module,
                                         const std::string& heuristic,
                                         const AliasMap& alias_map,
                                         const ClusterEnvironment& cluster_env);

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_STRATEGY_H_
