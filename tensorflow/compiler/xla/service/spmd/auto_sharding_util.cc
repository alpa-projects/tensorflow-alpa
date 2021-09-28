#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace spmd {

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
          } else {
            int64_t max_depth = depth_map.at(inst) + delta;
            for (const HloInstruction* operand : node->operands()) {
              max_depth = std::max(max_depth, depth_map.at(operand) + delta);
            }
            depth_map[node] = max_depth;
          }

          next_frontier.push_back(node);
          collected += 1;

          // std::cerr << node->ToString(HloPrintOptions::ShortParsable()) <<
          // "depth: " << depth_map[node] << std::endl;
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
          if (value == 0 && !absl::c_linear_search(dimensions, value)) {
            batch_map[ins] = value;
          }
        }
        break;
      }
      case HloOpcode::kReshape: {
        const HloInstruction* operand = ins->operand(0);

        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          if (value == 0) {
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
          if (value == 0 && dimensions[0] == 0) {
            batch_map[ins] = value;
          }
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
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kTuple:
      case HloOpcode::kCustomCall:
        break;
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->name();
    }
  }

  // for (auto iter : batch_map) {
  //   std::cerr << iter.first->ToString(HloPrintOptions::ShortParsable()) << "
  //   "
  //             << iter.second << std::endl;
  // }
  // exit(0);

  return batch_map;
}

// Return the output sharding of the reduce-scatter variant of a given strategy.
HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env) {
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
      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, cluster_env);
    } else {
      int mesh_dim0, mesh_dim1;
      if (strategy.name.find("{0,1}") != std::string::npos) {
        mesh_dim0 = 0;
        mesh_dim1 = 1;
      } else {
        mesh_dim0 = 1;
        mesh_dim1 = 0;
      }

      if (ins->shape().dimensions(space_base_dim) <
              cluster_env.device_mesh.dim(mesh_dim0) ||
          ins->shape().dimensions(space_base_dim + 1) <
              cluster_env.device_mesh.dim(mesh_dim1)) {
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                  {mesh_dim0, mesh_dim1}, cluster_env);
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

    if (ins->shape().dimensions(out_batch_dim) <
            cluster_env.device_mesh.dim(mesh_dim0) ||
        ins->shape().dimensions(out_out_channel_dim) <
            cluster_env.device_mesh.dim(mesh_dim1)) {
      return Undefined();
    }

    return Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
                {mesh_dim0, mesh_dim1}, cluster_env);
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
    return false;
  }

  return false;
}

// Substitute all-reduce strategies with their reduce-scatter variants.
void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           const std::vector<int64_t>& s_val,
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
        absl::flat_hash_set<HloInstruction*> users =
            UsersWithAlias(cur, alias_map, output);

        for (HloInstruction* consumer : users) {
          if (consumer->opcode() == HloOpcode::kTuple ||
              GetShardingStrategy(consumer).output_sharding !=
                  strategy.output_sharding ||
              !DimensionsEqual(consumer->shape(), inst->shape())) {
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
              GetShardingStrategy(operand).output_sharding ==
                  strategy.output_sharding &&
              DimensionsEqual(operand->shape(), inst->shape())) {
            find_replicated_set(operand);
          }
        }
      };

      // Find the replicated set starting from the all-reduce instruction.
      replicated_set.insert(inst);
      visited.insert(inst);
      visited.insert(output);
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
      // std::cerr << inst->ToString(HloPrintOptions::ShortParsable()) << "\n";
      // std::cerr << "replicated set (#parameter: " <<
      // num_replicated_parameters << "):\n"; for (auto x : replicated_set) {
      //  std::cerr << "  " << x->ToString(HloPrintOptions::ShortParsable()) <<
      //  "\n";
      //}
      // std::cerr << "boundary set (#incompatible: " << need_all_gather.size()
      // << "):\n"; for (auto x : boundary_set) {
      //  std::cerr << "  " << x->ToString(HloPrintOptions::ShortParsable())
      //            << " " << consumer_set.count(x) << "\n";
      //}

      // If applicable, replace all-reduce with reduce-scatter by
      // setting instructions' sharding.
      if (num_replicated_parameters >= 1 && need_all_gather.size() <= 1 &&
          replicated_set.size() >= 5) {
        HloSharding output_spec =
            GetReduceScatterOutput(inst, strategy, cluster_env);
        if (IsUndefined(output_spec)) {
          continue;
        }

        // std::cerr << "SET:  " << output_spec.ToString() << std::endl;

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
          if (to_split->users().size() == 1 &&
              to_split->users().front() == output &&
              alias_map.count(to_split)) {
            // Move the all-gather to its alias parameter
            alias_map.at(to_split)->set_sharding(output_spec);
            insert_all_gather.push_back(alias_map.at(to_split));
          } else {
            insert_all_gather.push_back(to_split);
          }
        }
      }

      // std::cerr << "-----------------------done\n" << std::endl;
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

}  // namespace spmd
}  // namespace xla
