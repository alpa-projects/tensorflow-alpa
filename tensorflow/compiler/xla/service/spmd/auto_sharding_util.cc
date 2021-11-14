#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace spmd {

const HloInstruction* PassThroughCustomCallMarkerGetSource(const HloInstruction* ins);

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}

// Propagate sharding for broadcast.
// The output will be tiled along the broadcasted dimension the same way
// as the input for the broadcast while the other dimensions are kept
// non-tiled.
HloSharding BroadcastSharding(const HloSharding& input_spec,
                              const Shape& new_shape,
                              const std::vector<int64_t>& dimensions) {
  if (input_spec.IsReplicated()) {
    return input_spec;
  }
  CHECK(new_shape.IsArray());

  std::vector<int64_t> target_tile_assignment_dimensions;
  for (int64_t i = 0; i < new_shape.rank(); ++i) {
    auto it = absl::c_find(dimensions, i);
    if (it == dimensions.end()) {
      target_tile_assignment_dimensions.push_back(1);
    } else {
      const int64_t source_dim = std::distance(dimensions.begin(), it);
      target_tile_assignment_dimensions.push_back(
          input_spec.tile_assignment().dim(source_dim));
    }
  }
  if (input_spec.ReplicateOnLastTileDim()) {
    target_tile_assignment_dimensions.push_back(
        input_spec.tile_assignment().dimensions().back());
  }
  Array<int64_t> new_tile_assignment = input_spec.tile_assignment();
  new_tile_assignment.Reshape(target_tile_assignment_dimensions);

  return input_spec.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(new_tile_assignment)
             : HloSharding::Tile(new_tile_assignment);
}

// Propagate sharding for dim-wise operations (e.g., slice, pad) which works
// independently on each dimension.
// The sharding can successfully propagate if the operation only happens
// on tensor dimentions that are not tiled.
absl::optional<HloSharding> PropagateDimwiseSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Shape& new_shape) {
  if (input_spec.IsReplicated()) {
    return input_spec;
  }

  CHECK(old_shape.IsArray());

  const auto& tile_assignment = input_spec.tile_assignment();
  for (int64_t i = 0; i < old_shape.rank(); ++i) {
    if (tile_assignment.dim(i) > 1 &&
        new_shape.dimensions(i) != old_shape.dimensions(i)) {
      return absl::nullopt;
    }
  }

  return input_spec;
}

// Propagate sharding for ReduceWindow-like operations.
// The sharding can successfully propagate if the window operation only happens
// on tensor dimentions that are not tiled.
absl::optional<HloSharding> PropagateReduceWindowSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Window& window) {
  if (input_spec.IsReplicated()) {
    return input_spec;
  }

  CHECK(!input_spec.IsTuple());

  const auto& tile_assignment = input_spec.tile_assignment();
  for (int64_t i = 0; i < old_shape.rank(); ++i) {
    if (tile_assignment.dim(i) > 1 && window.dimensions(i).size() != 1) {
      return absl::nullopt;
    }
  }

  return input_spec;
}

// Pass through the custom call marker and get the source instruction
const HloInstruction* PassThroughCustomCallMarkerGetSource(
    const HloInstruction* ins) {
  while (ins->opcode() == HloOpcode::kGetTupleElement &&
         IsCustomCallMarker(ins->operand(0))) {
    const HloInstruction* custom_call = ins->operand(0);
    const HloInstruction* tuple = custom_call->operand(0);
    while (IsCustomCallMarker(tuple)) {
      tuple = tuple->operand(0);
    }
    ins = tuple->operand(ins->tuple_index());
  }
  return ins;
}

// Depth analysis (breadth first search).
// We also assign a much larger distance to heavey operators (e.g., dot,
// convolution).
InstructionDepthMap BuildInstructionDepthMap(
    const HloInstructionSequence& sequence) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  InstructionDepthMap depth_map;
  absl::flat_hash_map<const HloInstruction*, size_t> degree_dict;

  // Init frontier
  size_t collected = 0;
  std::vector<const HloInstruction*> current_frontier;
  for (const HloInstruction* inst : instructions) {
    degree_dict[inst] = inst->unique_operands().size();
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
      for (const HloInstruction* node : inst->users()) {
        int now_degree = --degree_dict[node];
        if (now_degree == 0) {
          int64_t delta = 0;
          bool reset = false;

          // Heavy operators have more weight (distance).
          switch (node->opcode()) {
            case HloOpcode::kDot:
            case HloOpcode::kConvolution:
              delta = 1000;
              break;
            // A temporary hack here: reduce ops will generate replicated
            // sharding. We do not want the later broadcast and elementwise ops
            // to follow it. So we give reduce ops some penalty and let the
            // elementwise ops to follow other operands.
            // TODO(lmzheng): remove this hack by correctly registering
            // strategies for broadcast.
            case HloOpcode::kReduce:
              delta = 1;
              reset = true;
              break;
            // For similar reasons mentioned above, we give some penalty to
            // broadcast.
            case HloOpcode::kBroadcast:
              delta = -5;
              break;
            case HloOpcode::kConstant:
            case HloOpcode::kParameter:
              delta = 0;
              break;
            case HloOpcode::kReshape:
            case HloOpcode::kTranspose:
              delta = 0;
              break;
            case HloOpcode::kGetTupleElement:
            case HloOpcode::kTuple:
            case HloOpcode::kCustomCall:  // Mainly for pipeline_marker
              // Skip these useless instructions.
              delta = 0;
              break;
            default:
              delta = 1;
              break;
          }

          if (reset) {
            depth_map[node] = delta;
          } else if (node->opcode() == HloOpcode::kGetTupleElement &&
                     IsCustomCallMarker(node->operand(0))) {
            depth_map[node] =
                depth_map.at(PassThroughCustomCallMarkerGetSource(node));
          } else {
            int64_t max_depth = depth_map.at(inst) + delta;
            for (const HloInstruction* operand : node->operands()) {
              max_depth = std::max(max_depth, depth_map.at(operand) + delta);
            }
            depth_map[node] = max_depth;
          }

          next_frontier.push_back(node);
          collected += 1;
        }
      }
    }

    std::swap(current_frontier, next_frontier);
  }

  return depth_map;
}

// Batch dimension analysis that finds the batch dimension of each instruction.
InstructionBatchDimMap BuildInstructionBatchDimMap(
    const HloInstructionSequence& sequence) {
  InstructionBatchDimMap batch_map;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // We use the first dot or convolution as the source to start batch dim
  // propagation. Assume the first dim of the first dot is the batch dim.
  bool first_dot_conv = true;
  int batch_dim_of_source = 0;

  for (const HloInstruction* ins : instructions) {
    switch (ins->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kIota:
      case HloOpcode::kRngGetAndUpdateState:
        break;
      case HloOpcode::kBroadcast: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          int old_dim = -1;
          for (int i = 0; i < ins->shape().rank(); ++i) {
            if (absl::c_linear_search(dimensions, i)) {
              old_dim++;
            }

            if (old_dim == value) {
              batch_map[ins] = i;
              break;
            }
          }
        }
        break;
      }
      case HloOpcode::kReshape: {
        const HloInstruction* operand = ins->operand(0);

        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          bool match = true;
          for (int i = 0; i < value; ++i) {
            if (operand->shape().dimensions(i) != ins->shape().dimensions(i)) {
              match = false;
              break;
            }
          }

          if (match) {
            batch_map[ins] = value;
          }
        }
        break;
      }
      case HloOpcode::kTranspose: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          auto it = absl::c_find(dimensions, value);
          batch_map[ins] = it - dimensions.begin();
        }
        break;
      }
      case HloOpcode::kReverse:
      case HloOpcode::kPad:
      case HloOpcode::kSlice:
      case HloOpcode::kConcatenate:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSelectAndScatter:
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
        for (const HloInstruction* operand : ins->unique_operands()) {
          if (batch_map.count(operand)) {
            batch_map[ins] = batch_map[operand];
            break;
          }
        }
        break;
      }
      case HloOpcode::kReduce: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          if (value == 0 && !absl::c_linear_search(dimensions, value)) {
            batch_map[ins] = value;
          }
        }
        break;
      }
      case HloOpcode::kDot: {
        if (first_dot_conv) {
          first_dot_conv = false;
          batch_map[ins] = batch_dim_of_source;
          break;
        }

        const HloInstruction* lhs = ins->operand(0);
        const HloInstruction* rhs = ins->operand(1);
        const auto& dot_dnums = ins->dot_dimension_numbers();
        int64_t space_base_dim = dot_dnums.lhs_batch_dimensions_size();
        const auto& lhs_batch_dims =
            ins->dot_dimension_numbers().lhs_batch_dimensions();
        const auto& rhs_batch_dims =
            ins->dot_dimension_numbers().rhs_batch_dimensions();
        std::vector<int64_t> lhs_space_dims, rhs_space_dims;
        std::tie(lhs_space_dims, rhs_space_dims) =
            GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);

        if (batch_map.count(lhs)) {
          int value = batch_map.at(lhs);
          for (int i = 0; i < lhs_batch_dims.size(); ++i) {
            if (value == lhs_batch_dims[i]) {
              batch_map[ins] = i;
              break;
            }
          }
          if (value == lhs_space_dims[0]) {
            batch_map[ins] = space_base_dim;
          }
        }

        if (batch_map.count(rhs)) {
          int value = batch_map.at(rhs);
          for (int i = 0; i < rhs_batch_dims.size(); ++i) {
            if (value == rhs_batch_dims[i]) {
              batch_map[ins] = i;
              break;
            }
          }
          if (value == rhs_space_dims[0]) {
            batch_map[ins] = space_base_dim + 1;
          }
        }
        break;
      }
      case HloOpcode::kConvolution: {
        if (first_dot_conv) {
          first_dot_conv = false;
          batch_map[ins] = batch_dim_of_source;
          break;
        }

        const HloInstruction* lhs = ins->operand(0);
        const HloInstruction* rhs = ins->operand(1);
        const auto& conv_dnums = ins->convolution_dimension_numbers();

        if (batch_map.count(lhs)) {
          int value = batch_map.at(lhs);
          if (value == conv_dnums.input_batch_dimension()) {
            batch_map[ins] = conv_dnums.output_batch_dimension();
          }
        }

        if (batch_map.count(rhs)) {
          int value = batch_map.at(rhs);
          if (value == conv_dnums.kernel_output_feature_dimension()) {
            batch_map[ins] = conv_dnums.output_feature_dimension();
          }
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* source = PassThroughCustomCallMarkerGetSource(ins);
        if (batch_map.count(source)) {
          batch_map[ins] = batch_map[source];
        }
        break;
      }
      case HloOpcode::kTuple:
      case HloOpcode::kCustomCall:
        break;
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->name();
    }
  }

  // Print batch map
  std::cerr << "Batch dim map" << std::endl;
  for (const HloInstruction* ins : instructions) {
    std::cerr << ins->ToString(HloPrintOptions::ShortParsable());
    if (batch_map.count(ins)) {
      std::cerr << " BATCH " << batch_map[ins] << std::endl;
    } else {
      std::cerr << " NOBATCH " << std::endl;
    }
  }

  return batch_map;
}

// Return the output sharding of the reduce-scatter variant of a given strategy.
HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;

  if (ins->opcode() == HloOpcode::kDot) {
    const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
    int64_t space_base_dim = dot_dnums.lhs_batch_dimensions_size();

    if (StrStartsWith(strategy.name, "RR = RS x SR")) {
      int mesh_dim;
      if (strategy.name.find("{0}") != std::string::npos) {
        mesh_dim = 0;
      } else {
        mesh_dim = 1;
      }
      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, device_mesh);
    } else {
      int mesh_dim0, mesh_dim1;
      if (strategy.name.find("{0,1}") != std::string::npos) {
        mesh_dim0 = 0;
        mesh_dim1 = 1;
      } else {
        mesh_dim0 = 1;
        mesh_dim1 = 0;
      }

      if (ins->shape().dimensions(space_base_dim) %
                  cluster_env.device_mesh.dim(mesh_dim0) !=
              0 ||
          ins->shape().dimensions(space_base_dim + 1) %
                  cluster_env.device_mesh.dim(mesh_dim1) !=
              0) {
        // XLA supports uneven partitioning by adding padding.
        // However, the ShardingSpec in Jax does not support uneven
        // partitioning.
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                  {mesh_dim0, mesh_dim1}, device_mesh);
    }
  }
  if (ins->opcode() == HloOpcode::kConvolution) {
    const ConvolutionDimensionNumbers& conv_dnums =
        ins->convolution_dimension_numbers();
    int out_batch_dim = conv_dnums.output_batch_dimension();
    int out_out_channel_dim = conv_dnums.output_feature_dimension();

    int mesh_dim0, mesh_dim1;
    if (strategy.name.find("{0,1}") != std::string::npos) {
      mesh_dim0 = 0;
      mesh_dim1 = 1;
    } else {
      mesh_dim0 = 1;
      mesh_dim1 = 0;
    }

    if (ins->shape().dimensions(out_batch_dim) %
                cluster_env.device_mesh.dim(mesh_dim0) !=
            0 ||
        ins->shape().dimensions(out_out_channel_dim) %
                cluster_env.device_mesh.dim(mesh_dim1) !=
            0) {
      // XLA supports uneven partitioning by adding padding.
      // However, the ShardingSpec in Jax does not support uneven partitioning.
      return Undefined();
    }

    return Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
                {mesh_dim0, mesh_dim1}, device_mesh);
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
      return Tile(ins->shape(), {0}, {mesh_dim}, device_mesh);
    } else {
      Array<int64_t> tile_assignment =
          strategy.output_sharding.tile_assignment();
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
                                 const std::vector<int64_t>& s_val) {
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
  if (inst->opcode() == HloOpcode::kConvolution) {
    return true;
  }

  return false;
}

// Return whether all users of an instruction is reduce.
bool AllUsersAreReduce(const HloInstruction* inst) {
  for (const HloInstruction* user : inst->users()) {
    if (user->opcode() != HloOpcode::kReduce) {
      return false;
    }
  }
  return true;
}

// Set sharding, and apply transpose if necessary.
void SetSharding(HloInstruction* to_split, const HloSharding& output_spec,
                 const HloInstruction* ref_inst,
                 const HloInstruction* shape_inst) {
  if (DimensionsEqual(to_split->shape(), ref_inst->shape())) {
    to_split->set_sharding(output_spec);
  } else {
    CHECK(shape_inst != nullptr);
    CHECK_EQ(shape_inst->opcode(), HloOpcode::kTranspose);
    to_split->set_sharding(hlo_sharding_util::TransposeSharding(
        output_spec, shape_inst->dimensions()));
  }
}

// Return whether the instruction is always replicated.
// (e.g., constant, broadcasted constant, scalar)
bool IsAlwaysReplicated(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kConstant) {
    return true;
  }
  if (inst->shape().rank() == 0) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kBroadcast) {
    return IsAlwaysReplicated(inst->operand(0));
  }
  return false;
}

// Return whether this instruction is a convert on a parameter.
bool IsParameterConvert(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kConvert &&
      inst->operand(0)->opcode() == HloOpcode::kParameter) {
    return true;
  }
  return false;
}

// Pass through the custom call marker and get the acutal operand.
inline HloInstruction* PassThroughCustomCallMarkerOperand(
    HloInstruction* raw_operand, const HloInstruction* inst) {
  if (!IsCustomCallMarker(raw_operand)) {
    return raw_operand;
  }

  CHECK_EQ(inst->opcode(), HloOpcode::kGetTupleElement);

  int index = inst->tuple_index();
  return raw_operand->mutable_operand(0)->mutable_operand(index);
}

// Return whether the tuple is only used by a custom call marker.
inline bool IsCustomCallMarkerTuple(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kTuple && inst->users().size() == 1 &&
         IsCustomCallMarker(inst->users().front());
}

// Pass through the custom call marker and get the acutal user
inline HloInstruction* PassThroughCustomCallMarkerUser(
    HloInstruction* raw_user, const HloInstruction* inst) {
  if (!IsCustomCallMarkerTuple(raw_user)) {
    return raw_user;
  }

  const HloInstruction* custom_call = raw_user->users().front();

  int index = -1;
  for (int i = 0; i < raw_user->operand_count(); i++) {
    if (raw_user->operand(i) == inst) {
      index = i;
      break;
    }
  }
  CHECK(index != -1);

  HloInstruction* ret = nullptr;
  for (HloInstruction* user : custom_call->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    if (user->tuple_index() == index) {
      CHECK_EQ(ret, nullptr);
      ret = user;
    }
  }

  return ret == nullptr ? raw_user : ret;
}

// Return the users of an instruction and its alias,
// excluding the final output tuple.
inline absl::flat_hash_set<HloInstruction*> UsersWithAlias(
    const HloInstruction* inst, const AliasMap& alias_map,
    const HloInstruction* output) {
  absl::flat_hash_set<HloInstruction*> users;

  for (HloInstruction* user : inst->users()) {
    users.insert(PassThroughCustomCallMarkerUser(user, inst));
  }

  auto iter = alias_map.find(inst);
  if (iter != alias_map.end()) {
    for (HloInstruction* user : iter->second->users()) {
      users.insert(PassThroughCustomCallMarkerUser(user, iter->second));
    }
  }

  users.erase(output);
  return users;
}

// Substitute all-reduce strategies with their reduce-scatter variants.
void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           const std::vector<int64_t>& s_val,
                           const ClusterEnvironment& cluster_env) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Propagation ends at output
  const HloInstruction* output = instructions.back();
  if (IsCustomCallMarker(output)) {
    output = output->operand(0);
  }

  // A debug option: whether to do all-gather after backward pass.
  // This controls the location of all-gather.
  // If true, all-gather happends after backward pass, which is desired for
  // gradient accumulation. If false, all-gather happends before forward pass,
  // which can partitions more tensors.
  bool do_all_gather_after_backward = true;
  int verbose = 0;

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

      // We allow at most one transpose in the path of replication analysis.
      HloInstruction* transpose_inst = nullptr;

      // Use DFS to find the replicated set.
      std::function<void(HloInstruction*)> find_replicated_set;
      find_replicated_set = [&](HloInstruction* cur) {
        visited.insert(cur);

        // Check whether the node is a boundary node.
        absl::flat_hash_set<HloInstruction*> users =
            UsersWithAlias(cur, alias_map, output);
        for (HloInstruction* consumer : users) {
          const HloInstruction* shape_inst = cur;
          if (consumer->opcode() == HloOpcode::kTranspose &&
              transpose_inst == nullptr) {
            shape_inst = consumer;
            transpose_inst = consumer;
            // TODO(lmzheng): fix output_sharding comparison.
          }

          if (consumer->opcode() == HloOpcode::kTuple ||
              (do_all_gather_after_backward && IsParameterConvert(consumer)) ||
              GetShardingStrategy(consumer).output_sharding !=
                  strategy.output_sharding ||
              !DimensionsEqual(consumer->shape(), shape_inst->shape())) {
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
          operand = PassThroughCustomCallMarkerOperand(operand, cur);

          if (!visited.count(operand) && !IsAlwaysReplicated(operand) &&
              GetShardingStrategy(operand).output_sharding ==
                  strategy.output_sharding &&
              DimensionsEqual(operand->shape(), cur->shape())) {
            find_replicated_set(operand);
          }
        }
      };

      // Find the replicated set starting from the all-reduce instruction.
      visited.insert(output);
      find_replicated_set(inst);

      // Analyze the instructions after which all-gather should be inserted.
      std::vector<HloInstruction*> need_all_gather;
      for (HloInstruction* node : boundary_set) {
        if (consumer_set.count(node)) {
          if (AllUsersAreReduce(node)) {
            // If users are reduce, the all-gather cost after this instructioon
            // should be small, so we ignore all-gather cost of these
            // instructions.
            replicated_set.insert(node);
          } else {
            need_all_gather.push_back(node);
          }
        }
      }

      // If we do all-gather on some parameters, move this all-gather after
      // backward.
      if (do_all_gather_after_backward && need_all_gather.size() == 1) {
        HloInstruction* point = need_all_gather.front();
        std::vector<HloInstruction*> path;

        HloInstruction* root = point;
        while (true) {
          path.push_back(root);
          if (root->opcode() == HloOpcode::kGetTupleElement) {
            root = PassThroughCustomCallMarkerOperand(root->mutable_operand(0),
                                                      root);
          } else {
            break;
          }
        }

        if (root->opcode() == HloOpcode::kParameter) {
          for (auto x : path) {
            replicated_set.erase(x);
            boundary_set.erase(x);
          }
          need_all_gather.clear();
          for (auto x : replicated_set) {
            auto iter = alias_map.find(x);
            if (iter != alias_map.end() && iter->second == root) {
              boundary_set.insert(x);
              need_all_gather.push_back(x);
              break;
            }
          }
        }
      }

      // Analyze how many parameters can be partitioned if we do this
      // transformation.
      int num_replicated_parameters = 0;
      for (const HloInstruction* node : replicated_set) {
        if (node->opcode() == HloOpcode::kParameter) {
          num_replicated_parameters++;
        }
      }
      for (const HloInstruction* to_split : need_all_gather) {
        if (to_split->users().size() == 1 &&
            to_split->users().front() == output && alias_map.count(to_split)) {
          // Move the all-gather to its alias parameter.
          num_replicated_parameters++;
        }
      }

      // Print replicated set and boundary set
      StdCerr(verbose) << inst->ToString(HloPrintOptions::ShortParsable())
                       << "\n";
      StdCerr(verbose) << "replicated set (#parameter: "
                       << num_replicated_parameters << "):\n";
      for (auto x : replicated_set) {
        StdCerr(verbose) << "  "
                         << x->ToString(HloPrintOptions::ShortParsable())
                         << "\n";
      }
      StdCerr(verbose) << "boundary set (#incompatible: "
                       << need_all_gather.size() << "):\n";
      for (auto x : boundary_set) {
        StdCerr(verbose) << "  "
                         << x->ToString(HloPrintOptions::ShortParsable()) << " "
                         << absl::c_linear_search(need_all_gather, x) << "\n";
      }

      // If applicable, replace all-reduce with reduce-scatter by
      // setting instructions' sharding.
      if (num_replicated_parameters >= 1 && need_all_gather.size() <= 1 &&
          replicated_set.size() >= 5) {
        HloSharding output_spec =
            GetReduceScatterOutput(inst, strategy, cluster_env);
        if (IsUndefined(output_spec)) {
          continue;
        }

        StdCerr(verbose) << "SET:  " << output_spec.ToString() << std::endl;

        if (StrStartsWith(strategy.name, "RR = RS x SR")) {
          // If set the sharding for this dot instruction, the SPMD
          // partitioner will generate bad fallback code.
          replicated_set.erase(inst);
        }

        for (HloInstruction* to_split : replicated_set) {
          SetSharding(to_split, output_spec, inst, transpose_inst);
        }

        for (HloInstruction* to_split : need_all_gather) {
          SetSharding(to_split, output_spec, inst, transpose_inst);

          if (!do_all_gather_after_backward && to_split->users().size() == 1 &&
              to_split->users().front() == output &&
              alias_map.count(to_split)) {
            // Move the all-gather to its alias parameter.
            // This partitions more tensors but introduces communication
            // in the forward pass, which is not desired in gradient
            // accumulation.
            SetSharding(alias_map.at(to_split), output_spec, inst,
                        transpose_inst);
            insert_all_gather.push_back(alias_map.at(to_split));
          } else {
            insert_all_gather.push_back(to_split);
          }
        }
      }

      StdCerr(verbose) << "-----------------------done\n" << std::endl;
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

void RemoveCustomCallMarker(HloModule* module) {
  HloComputation* entry_computation = module->entry_computation();

  std::vector<HloInstruction*> get_tuple_ins;
  std::vector<HloInstruction*> marker_ins;

  for (HloInstruction* ins : entry_computation->instructions()) {
    if (ins->opcode() == HloOpcode::kGetTupleElement &&
        IsCustomCallMarker(ins->operand(0))) {
      get_tuple_ins.push_back(ins);
      marker_ins.push_back(ins->mutable_operand(0));
    }
  }

  for (HloInstruction* raw_ins : get_tuple_ins) {
    HloInstruction* ins = raw_ins;
    while (ins->opcode() == HloOpcode::kGetTupleElement) {
      HloInstruction* custom_call = ins->mutable_operand(0);
      CHECK(IsCustomCallMarker(custom_call));
      HloInstruction* tuple = custom_call->mutable_operand(0);
      ins = tuple->mutable_operand(ins->tuple_index());
    }

    raw_ins->ReplaceAllUsesWith(ins);
  }

  for (HloInstruction* ins : get_tuple_ins) {
    entry_computation->RemoveInstruction(ins);
  }

  absl::flat_hash_set<const HloInstruction*> removed;
  for (HloInstruction* ins : marker_ins) {
    if (!removed.count(ins)) {
      HloInstruction* tmp = ins->mutable_operand(0);
      entry_computation->RemoveInstruction(ins);
      entry_computation->RemoveInstruction(tmp);
      removed.insert(ins);
    }
  }
}

// Get the values in an array along a dimesion.
// e.g., if dim == 1, this returns array[0,:,0,0] following the numpy syntax.
std::vector<int64_t> GetValuesAlongOneDim(const Array<int64_t>& array,
                                          int dim) {
  std::vector<int64_t> ret;
  std::vector<int64_t> indices(array.num_dimensions(), 0);

  for (int i = 0; i < array.dim(dim); ++i) {
    indices[dim] = i;
    ret.push_back(array(indices));
  }

  return ret;
}

// Check whether a sequence is an arithmetic sequence.
std::pair<int64_t, bool> CheckArithmeticSequence(
    const std::vector<int64_t>& sequence) {
  if (sequence.size() < 2) {
    return std::make_pair(0, false);
  }
  int64_t delta = sequence[1] - sequence[0];
  for (int i = 2; i < sequence.size(); ++i) {
    if (sequence[i] - sequence[i - 1] != delta) {
      return std::make_pair(delta, false);
    }
  }
  return std::make_pair(delta, true);
}

std::pair<std::vector<int>, int> GetTensorDimToMeshDimInternal(
    const Shape& shape, const HloSharding& spec) {
  CHECK(shape.IsArray());
  CHECK(!IsUndefined(spec));

  if (spec.IsReplicated()) {
    return std::make_pair(std::vector<int>(shape.rank(), -1), -1);
  }

  const Array<int64_t>& tile_assignment = spec.tile_assignment();

  // Extract all tile dims
  std::vector<int> tile_dims;
  for (int i = 0; i < tile_assignment.num_dimensions(); i++) {
    if (tile_assignment.dim(i) != 1) {
      tile_dims.push_back(i);
    }
  }

  // Sort the tile dims according to the device id delta along the tile
  // dimension
  bool success;
  std::vector<int> tile_dims_delta;
  for (int dim : tile_dims) {
    std::vector<int64_t> device_ids =
        GetValuesAlongOneDim(tile_assignment, dim);
    int64_t delta;
    std::tie(delta, success) = CheckArithmeticSequence(device_ids);

    CHECK(success) << "Invalid device id assignment";
    tile_dims_delta.push_back(delta);
  }

  std::vector<int> tile_dims_argsort(tile_dims.size(), 0);
  std::iota(tile_dims_argsort.begin(), tile_dims_argsort.end(), 0);
  std::sort(tile_dims_argsort.begin(), tile_dims_argsort.end(),
            [&](int idx_a, int idx_b) {
              return tile_dims_delta[idx_a] > tile_dims_delta[idx_b];
            });
  std::vector<int> tile_dims_rank(tile_dims.size());
  for (int i = 0; i < tile_dims.size(); ++i) {
    tile_dims_rank[tile_dims_argsort[i]] = i;
  }

  // Map tensor dims to mesh dims
  std::vector<int> ret(shape.rank(), -1);
  int ct = 0;
  for (int i = 0; i < shape.rank(); ++i) {
    if (tile_assignment.dim(i) == 1) {
      ret[i] = -1;
    } else {
      ret[i] = tile_dims_rank[ct++];
    }
  }
  if (spec.ReplicateOnLastTileDim()) {
    ct++;
  }
  if (ct != tile_dims.size()) {
    std::cerr << "shape:" << shape.ToString() << std::endl;
    std::cerr << "spec:" << spec.ToString() << std::endl;
    std::cerr << "tile_dims:" << ToString(tile_dims) << std::endl;
  }
  CHECK_EQ(ct, tile_dims.size());

  return std::make_pair(ret, tile_dims.size());
}

void FixMixedMeshShapeResharding(HloInstruction* inst, int operand_num,
                                 const HloSharding& dst_sharding,
                                 const Array<int64_t>& device_mesh,
                                 ReshardingCache& resharding_cache) {
  HloInstruction* operand = inst->mutable_operand(operand_num);
  if (operand->sharding() == dst_sharding) {
    return;
  }

  const HloSharding& src_sharding = operand->sharding();
  const Shape& shape = operand->shape();

  std::vector<int> src_tensor_dim_to_mesh_dim, dst_tensor_dim_to_mesh_dim;
  int src_n_dim, dst_n_dim;

  std::tie(src_tensor_dim_to_mesh_dim, src_n_dim) =
      GetTensorDimToMeshDimInternal(shape, src_sharding);
  std::tie(dst_tensor_dim_to_mesh_dim, dst_n_dim) =
      GetTensorDimToMeshDimInternal(shape, dst_sharding);

  HloInstruction* replace_with = nullptr;

  // Query cache first
  auto& cache = resharding_cache[operand];
  for (auto& entry : cache) {
    if (entry.first == dst_sharding) {
      replace_with = entry.second;
    }
  }

  if (replace_with != nullptr) {
    ;
  } else if (src_n_dim != dst_n_dim && src_n_dim != -1 && dst_n_dim != -1) {
    const HloSharding* sharding_1d;

    if (src_n_dim == 1) {
      sharding_1d = &src_sharding;
    } else {
      sharding_1d = &dst_sharding;
    }

    // Find an intermidiate shape
    std::vector<int64_t> inter_shape_dims;

    for (size_t i = 0; i < shape.rank(); ++i) {
      if (sharding_1d->tile_assignment().dim(i) == 1) {
        inter_shape_dims.push_back(shape.dimensions(i));
      } else {
        CHECK(shape.dimensions(i) % device_mesh.dim(0) == 0) << "Only support even partition";
        inter_shape_dims.push_back(device_mesh.dim(0));
        inter_shape_dims.push_back(shape.dimensions(i) / device_mesh.dim(0));
      }
    }
    Shape inter_shape = ShapeUtil::MakeShape(shape.element_type(), inter_shape_dims);

    absl::optional<HloSharding> src_inter_sharding =
        hlo_sharding_util::ReshapeSharding(shape, inter_shape, src_sharding);
    CHECK(src_inter_sharding.has_value());
    absl::optional<HloSharding> dst_inter_sharding =
        hlo_sharding_util::ReshapeSharding(shape, inter_shape, dst_sharding);
    CHECK(dst_inter_sharding.has_value());

    HloInstruction* src_inter = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(inter_shape, operand));
    src_inter->set_sharding(*src_inter_sharding);

    HloInstruction* dst_inter = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(inter_shape, src_inter));
    dst_inter->set_sharding(*dst_inter_sharding);

    replace_with = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(shape, dst_inter));
    replace_with->set_sharding(dst_sharding);
    cache.push_back({dst_sharding, replace_with});
  } else {
    replace_with = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(operand->shape(), operand));
    replace_with->set_sharding(dst_sharding);
    cache.push_back({dst_sharding, replace_with});
  }

  inst->ReplaceOperandWith(operand_num, replace_with);
}

}  // namespace spmd
}  // namespace xla
