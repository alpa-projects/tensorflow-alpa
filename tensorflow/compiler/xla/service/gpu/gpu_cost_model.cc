#include "tensorflow/compiler/xla/service/gpu/gpu_cost_model.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pass_context.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

namespace xla {
namespace gpu {

namespace py = pybind11;

std::string ToString(const PrimitiveType& type) {
  return primitive_util::LowercasePrimitiveTypeName(type);
}

// Make a string key of a replica_groups.
std::string Group2Str(py::object tuple) {
  if (tuple.is_none()) {
    return "()";
  }

  py::tuple replica_groups = py::cast<py::tuple>(tuple);
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
std::string Group2Str(const std::vector<std::vector<int>>& replica_groups) {
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

// Make a string key of a replica_groups.
std::string Group2Str(const std::vector<ReplicaGroup>& replica_groups) {
  std::ostringstream os;

  os << "(";
  for (const auto& group : replica_groups) {
    os << "(";
    for (const auto& id : group.replica_ids()) {
      os << id << ",";
    }
    os << "),";
  }
  os << ")";

  return os.str();
}

// Store the profiling results of communication and computation.
class ProfilingResult {
 public:
  // Construct the class from the corresponding python object
  // alpa/mesh_profiling.py::ProfilingResult.
  ProfilingResult(py::object prof_result) {
    if (!prof_result.is_none()) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      {
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("all_gather_cost_dict")),
            all_gather_cost_dict_);
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("all_reduce_cost_dict")),
            all_reduce_cost_dict_);
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("all_to_all_cost_dict")),
            all_to_all_cost_dict_);
        CommDictPyToCpp(
            py::cast<py::dict>(prof_result.attr("reduce_scatter_cost_dict")),
            reduce_scatter_cost_dict_);
        CommDictPyToCpp(py::cast<py::dict>(prof_result.attr("dot_cost_dict")),
                        dot_cost_dict_);
      }
      PyGILState_Release(gstate);
    }
  }

  bool Enabled() const { return enabled_; }

  double EstimateAllGatherCost(const std::vector<ReplicaGroup>& replica_groups,
                               int64_t size, PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(const std::vector<ReplicaGroup>& replica_groups,
                               int64_t size, PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateAllToAllCost(const std::vector<ReplicaGroup>& replica_groups,
                              int64_t size, PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_to_all_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_to_all_cost_dict_);
  }

  double EstimateReduceScatterCost(
      const std::vector<ReplicaGroup>& replica_groups, int64_t size,
      PrimitiveType dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype,
                            reduce_scatter_cost_dict_);
  }

  double EstimateDotCost(int64_t flop_count, PrimitiveType dtype) {
    std::vector<ReplicaGroup> fake_replica_groups;
    return EstimateInternal(fake_replica_groups, flop_count, dtype,
                            dot_cost_dict_) -
           EstimateInternal(fake_replica_groups, 0, dtype, dot_cost_dict_);
  }

  std::string ToString() {
    std::ostringstream os;
    os << "all-reduce cost dict:\n";
    for (const auto& item : all_reduce_cost_dict_) {
      os << "key: (" << item.first.first << ", "
         << gpu::ToString(item.first.second) << ")\n";
    }
    os << "dot cost dict:\n";
    for (const auto& item : dot_cost_dict_) {
      os << "key: (" << item.first.first << ", "
         << gpu::ToString(item.first.second) << ")\n";
    }
    return os.str();
  }

 private:
  // Dictionary type for communicaiton cost.
  // Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
  // pair<group, dtype>
  using CommDictKey = std::pair<std::string, PrimitiveType>;
  // vector<pair<size, time>>
  using CommDictValue = std::vector<std::pair<int64_t, double>>;
  using CommDict = absl::flat_hash_map<CommDictKey, CommDictValue>;

  // Estimate the cost by linear interpolation bewteen the two closest points.
  double EstimateInternal(const std::vector<ReplicaGroup>& replica_groups,
                          int64_t size, PrimitiveType dtype,
                          const CommDict& cost_dict) const {
    if (dtype != PrimitiveType::F16 && dtype != PrimitiveType::F32) {
      // Cast other types to F32.
      dtype = PrimitiveType::F32;
    }

    CommDictKey key(Group2Str(replica_groups), dtype);
    if (!cost_dict.count(key)) {
      LOG(WARNING) << "Warning: cannot find key: (" << key.first << ", "
                   << gpu::ToString(key.second) << ")" << std::endl;
      return size;
    }
    CommDictValue cost_list = cost_dict.at(key);

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

  // Convert a python communication cost dict to c++ dict.
  void CommDictPyToCpp(py::dict py_dict, CommDict& cpp_dict) {
    // the type of py_dict: Dict[Tuple(group, dtype) -> List[Tuple(size, time)]]
    for (auto item : py_dict) {
      py::tuple tuple_key = py::cast<py::tuple>(item.first);
      std::string dtype_str = py::cast<std::string>(tuple_key[1]);
      PrimitiveType dtype;

      if (dtype_str == "f16") {
        dtype = PrimitiveType::F16;
      } else if (dtype_str == "f32") {
        dtype = PrimitiveType::F32;
      } else {
        LOG(FATAL) << "Invalid dtype: " << dtype_str;
      }

      CommDictKey key(Group2Str(tuple_key[0]), dtype);

      py::list list_val = py::cast<py::list>(item.second);
      for (const auto x : list_val) {
        py::tuple tuple_val = py::cast<py::tuple>(x);
        cpp_dict[key].push_back(std::make_pair(py::cast<int64_t>(tuple_val[0]),
                                               py::cast<double>(tuple_val[1])));
      }
    }
  }

  bool enabled_;
  CommDict all_reduce_cost_dict_;
  CommDict all_gather_cost_dict_;
  CommDict reduce_scatter_cost_dict_;
  CommDict all_to_all_cost_dict_;
  CommDict dot_cost_dict_;  // Reuse CommDict data structure for dot.
};

// Expand the special replica_groups {{0}} to {{0,1,2,..,n}}
const std::vector<ReplicaGroup> ExpandSpecialReplicaGroups(
    const std::vector<ReplicaGroup>& replica_groups, int64_t num_devices) {
  if (replica_groups.size() == 1 && replica_groups[0].replica_ids_size() == 1 &&
      num_devices != 1) {
    ReplicaGroup group;
    for (int64_t i = 0; i < num_devices; ++i) {
      group.add_replica_ids(i);
    }
    return {group};
  } else {
    return replica_groups;
  }
}

double EstimateHloModuleCost(const HloModule* hlo_module) {
  // Load profiling results.
  ProfilingResult prof_result(
      pass_context::GetPyObject("gpu_cost_model::profiling_results"));
  const int64_t num_devices = hlo_module->config().num_partitions();
  int verbose = pass_context::GetInt("gpu_cost_model::verbose", 0);
  int num_micro_batches =
      pass_context::GetInt("gpu_cost_model::num_micro_batches", 1);
  std::string grad_sync_channel_ids =
      pass_context::GetString("gpu_cost_model::grad_sync_channel_ids", "");

  // Compute cost of all instruction.
  double sum = 0.0;
  const HloComputation* entry_computation = hlo_module->entry_computation();
  for (const HloInstruction* ins : entry_computation->instructions()) {
    double cost = 0.0;

    if (ins->opcode() == HloOpcode::kAllGather ||
        ins->opcode() == HloOpcode::kAllReduce ||
        ins->opcode() == HloOpcode::kAllToAll ||
        ins->opcode() == HloOpcode::kReduceScatter) {
      auto coll = DynCast<HloCollectiveInstruction>(ins);
      CHECK(coll != nullptr);

      std::vector<ReplicaGroup> replica_groups = coll->replica_groups();
      // Expand the special replica_groups {{0}}
      replica_groups = ExpandSpecialReplicaGroups(replica_groups, num_devices);

      for (const auto operand : ins->operands()) {
        int64_t size = spmd::GetBytes(operand->shape());
        switch (ins->opcode()) {
          case HloOpcode::kAllGather:
            cost += prof_result.EstimateAllGatherCost(
                replica_groups, size, operand->shape().element_type());
            break;
          case HloOpcode::kAllReduce: {
            double normalizer = 1.0;

            // Amortize the cost of grad sync with the number of micro batches.
            std::string key = absl::StrFormat(".%d.", *ins->channel_id());
            if (grad_sync_channel_ids.find(key) != std::string::npos) {
              normalizer = num_micro_batches;
            }

            cost += prof_result.EstimateAllReduceCost(
                        replica_groups, size, operand->shape().element_type()) /
                    normalizer;
            break;
          }
          case HloOpcode::kAllToAll:
            cost += prof_result.EstimateAllToAllCost(
                replica_groups, size, operand->shape().element_type());
            break;
          case HloOpcode::kReduceScatter:
            cost += prof_result.EstimateReduceScatterCost(
                replica_groups, size, operand->shape().element_type());
            break;
          default:
            break;
        }
      }
    }

    if (ins->IsCustomCall(kGemmCallTarget)) {
      const HloInstruction* lhs = ins->operand(0);
      const HloInstruction* rhs = ins->operand(1);
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      auto config = ins->backend_config<GemmBackendConfig>().value();
      auto dnum = config.dot_dimension_numbers();
      std::tie(lhs_space_dims, rhs_space_dims) =
          spmd::GetSpaceDims(lhs->shape(), rhs->shape(), dnum);

      int64_t flop_count =
          lhs->shape().dimensions(lhs_space_dims[0]) *
          lhs->shape().dimensions(dnum.lhs_contracting_dimensions(0)) *
          rhs->shape().dimensions(rhs_space_dims[0]);
      for (int dim : dnum.lhs_batch_dimensions()) {
        flop_count *= lhs->shape().dimensions(dim);
      }
      flop_count *= 2;

      cost +=
          prof_result.EstimateDotCost(flop_count, ins->shape().element_type());
    }

    if (cost > 0) {
      spmd::StdCerr(verbose) << ins->ToString() << " cost: " << std::fixed
                             << std::setprecision(8) << cost << std::endl;
    }

    sum += cost;
  }

  return sum;
}

}  // namespace gpu
}  // namespace xla
