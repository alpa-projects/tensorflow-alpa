#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/service/spmd/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace spmd {

inline const HloInstruction* PassThroughCustomCallMarkerGetSource(
    const HloInstruction* ins);
inline HloInstruction* PassThroughCustomCallMarkerUser(
    HloInstruction* raw_user, const HloInstruction* inst);

NullStream& NullStream::Global() {
  static NullStream stream;
  return stream;
}

const char* const kPipelineMarker = "pipeline_marker";
const char* const kIdentityMarker = "identity";

// Return whether a reshape instruction is a special reshape that switches
// the batch dim of a dot.
bool IsBatchDimSwitchReshape(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kReshape) {
    return false;
  }
  if (inst->users().size() != 1) {
    return false;
  }
  const HloInstruction* operand = inst->operand(0);
  const HloInstruction* user = inst->users().front();

  if (operand->opcode() != HloOpcode::kDot) {
    return false;
  }

  int batch_dims = operand->dot_dimension_numbers().lhs_batch_dimensions_size();
  if (batch_dims <= 0) {
    return false;
  }

  if (user->opcode() != HloOpcode::kTranspose) {
    return false;
  }

  return true;
}

// Return whether the instruction is followed by a broadcast.
bool IsFollowedByBroadcast(const HloInstruction* ins) {
  const int max_depth = 6;
  for (int i = 0; i < max_depth; ++i) {
    if (ins->users().empty()) {
      return false;
    }
    ins = PassThroughCustomCallMarkerUser(ins->users().front(), ins);
    if (ins->opcode() == HloOpcode::kBroadcast) {
      return true;
    } else if (ins->opcode() == HloOpcode::kReshape) {
      i--;
    }
  }

  return false;
}

// Return whether the instruction is followed by a reduce.
bool IsFollowedByReduce(const HloInstruction* ins) {
  int max_depth = 1;
  bool found = false;

  std::function<void(const HloInstruction*, int)> dfs;

  dfs = [&](const HloInstruction* cur, int depth) {
    if (found) {
      return;
    }

    if (cur->opcode() == HloOpcode::kReduce) {
      found = true;
      return;
    }

    if (cur->opcode() == HloOpcode::kGetTupleElement) {
      depth -= 1;
    }

    if (depth < max_depth) {
      for (auto user : cur->users()) {
        dfs(PassThroughCustomCallMarkerUser(user, cur), depth + 1);
      }
    }
  };

  dfs(ins, 0);

  return found;
}

// Return whether the instruction is an activation from another pipeline stage.
bool IsActivationFromAnotherStage(const HloInstruction* ins,
                                  const InstructionBatchDimMap& batch_dim_map) {
  if (!(ins->opcode() == HloOpcode::kParameter && batch_dim_map.count(ins))) {
    return false;
  }

  for (const HloInstruction* user : ins->users()) {
    if (!(user->opcode() == HloOpcode::kTuple && user->users().size() == 1 &&
          user->users().front()->IsCustomCall(kPipelineMarker) &&
          user->users().front()->metadata().op_type().find("start") !=
              std::string::npos)) {
      return false;
    }
  }

  if (primitive_util::IsIntegralType(ins->shape().element_type())) {
    // TODO(lmzheng): This is a temporary hack. We use this to filter out
    // the input word ids and position ids. These are global input so they are
    // not activations from the previous stage. If we do not filter out them,
    // some follow-up instructions will follow the wrong instructions.
    return false;
  }

  return true;
}

// Propagate sharding for broadcast.
// The output will be tiled along the broadcasted dimension the same way
// as the input for the broadcast while the other dimensions are kept
// non-tiled.
HloSharding BroadcastSharding(const HloSharding& input_spec,
                              const Shape& new_shape,
                              const absl::Span<const int64_t>& dimensions) {
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
// on tensor dimensions that are not tiled.
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
// on tensor dimensions that are not tiled.
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
inline const HloInstruction* PassThroughCustomCallMarkerGetSource(
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
// We also assign a much larger distance to heavy operators (e.g., dot,
// convolution).
InstructionDepthMap BuildInstructionDepthMap(
    const HloInstructionSequence& sequence,
    const InstructionBatchDimMap& batch_dim_map) {
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

      // Add some initial depth for activations from other pipeline stages.
      if (IsActivationFromAnotherStage(inst, batch_dim_map)) {
        depth_map[inst] = 20;
      }

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
              reset = true;
              break;
            // For similar reasons mentioned above, we give some penalty to
            // broadcast.
            case HloOpcode::kBroadcast:
              delta = -5;
              break;
            case HloOpcode::kReshape:
              delta = 0;
              break;
            default:
              delta = 1;
              break;
          }

          if (reset) {
            depth_map[node] = 0;
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
  int batch_dim_of_source = 0;

  // Find the source of batch_dim propagation
  bool set_the_next_dot_conv = true;
  for (const HloInstruction* ins : instructions) {
    if (ins->opcode() == HloOpcode::kDot ||
        ins->opcode() == HloOpcode::kConvolution) {
      if (set_the_next_dot_conv) {
        set_the_next_dot_conv = false;
        batch_map[ins] = batch_dim_of_source;
      }
    }

    if (ins->IsCustomCall(kPipelineMarker) &&
        ins->metadata().op_type().find("start") != std::string::npos) {
      // Reset the status after meet a new pipeline marker.
      set_the_next_dot_conv = true;
    }
  }

  // Forward propagation: propagate from operand
  for (const HloInstruction* ins : instructions) {
    switch (ins->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kIota:
      case HloOpcode::kRngGetAndUpdateState:
      case HloOpcode::kRng:
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
            int value = batch_map[operand];
            if (operand->shape().rank() == ins->shape().rank() &&
                operand->shape().dimensions(value) ==
                    ins->shape().dimensions(value)) {
              batch_map[ins] = batch_map[operand];
              break;
            }
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
          int value = batch_map[lhs];
          for (int i = 0; i < lhs_batch_dims.size(); ++i) {
            if (value == lhs_batch_dims[i]) {
              batch_map[ins] = i;
              break;
            }
          }
          if (!lhs_space_dims.empty() && value == lhs_space_dims[0]) {
            batch_map[ins] = space_base_dim;
          }
        }

        if (batch_map.count(rhs)) {
          int value = batch_map[rhs];
          for (int i = 0; i < rhs_batch_dims.size(); ++i) {
            if (value == rhs_batch_dims[i]) {
              batch_map[ins] = i;
              break;
            }
          }
          if (!rhs_space_dims.empty() && value == rhs_space_dims[0]) {
            batch_map[ins] = space_base_dim + 1;
          }
        }
        break;
      }
      case HloOpcode::kConvolution: {
        const HloInstruction* lhs = ins->operand(0);
        const HloInstruction* rhs = ins->operand(1);
        const auto& conv_dnums = ins->convolution_dimension_numbers();

        if (batch_map.count(lhs)) {
          int value = batch_map[lhs];
          if (value == conv_dnums.input_batch_dimension()) {
            batch_map[ins] = conv_dnums.output_batch_dimension();
          }
        }

        if (batch_map.count(rhs)) {
          int value = batch_map[rhs];
          if (value == conv_dnums.kernel_output_feature_dimension()) {
            batch_map[ins] = conv_dnums.output_feature_dimension();
          }
        }
        break;
      }
      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        // We only handle one case for now:
        // If gather/scatter does not happen on the batch dimension,
        // then we can propagate the batch dim.
        const HloInstruction* operand = ins->operand(0);
        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          if (ins->shape().rank() == operand->shape().rank() &&
              ins->shape().dimensions(value) ==
                  operand->shape().dimensions(value)) {
            batch_map[ins] = value;
          }
        }
        break;
      }
      case HloOpcode::kSort: {
        for (size_t i = 0; i < ins->operand_count(); ++i) {
          const HloInstruction* operand = ins->operand(i);
          if (batch_map.count(operand)) {
            int value = batch_map[operand];
            if (!absl::c_linear_search(ins->dimensions(), value)) {
              batch_map[ins] = value;
              break;
            }
          }
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* source =
            PassThroughCustomCallMarkerGetSource(ins);
        if (batch_map.count(source)) {
          batch_map[ins] = batch_map[source];
        }
        break;
      }
      case HloOpcode::kTuple:
      case HloOpcode::kCustomCall:
        break;
      case HloOpcode::kWhile:
        break;
      default:
        LOG(FATAL) << "Unhandled instruction: " + ins->ToString();
    }
  }

  // Backward propagation: propagate to operands
  for (int64_t i = instructions.size() - 1; i >= 0; i--) {
    const HloInstruction* ins = instructions[i];
    switch (ins->opcode()) {
      case HloOpcode::kBroadcast: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(ins) && !batch_map.count(operand)) {
          int value = batch_map[ins];
          int old_dim = -1;
          for (int i = 0; i < ins->shape().rank(); ++i) {
            if (absl::c_linear_search(dimensions, i)) {
              old_dim++;
            }

            if (i == value && old_dim >= 0) {
              batch_map[operand] = old_dim;
              break;
            }
          }
        }
        break;
      }
      case HloOpcode::kReshape: {
        const HloInstruction* operand = ins->operand(0);

        if (batch_map.count(ins) && !batch_map.count(operand)) {
          int value = batch_map[ins];
          bool match = true;
          for (int i = 0; i < value; ++i) {
            if (operand->shape().dimensions(i) != ins->shape().dimensions(i)) {
              match = false;
              break;
            }
          }

          if (match) {
            batch_map[operand] = value;
          }
        }
        break;
      }
      case HloOpcode::kTranspose: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(ins) && !batch_map.count(operand)) {
          batch_map[operand] = dimensions[batch_map[ins]];
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
        if (batch_map.count(ins)) {
          int value = batch_map[ins];
          for (const HloInstruction* operand : ins->unique_operands()) {
            if (!batch_map.count(operand) &&
                operand->shape().rank() == ins->shape().rank() &&
                operand->shape().dimensions(value) ==
                    ins->shape().dimensions(value)) {
              batch_map[operand] = value;
            }
          }
        }
        break;
      }
      case HloOpcode::kReduce: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(ins) && !batch_map.count(operand)) {
          int value = batch_map[ins];
          if (value == 0 && !absl::c_linear_search(dimensions, value)) {
            batch_map[operand] = value;
          }
        }
        break;
      }
      case HloOpcode::kDot: {
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

        if (batch_map.count(ins)) {
          int value = batch_map[ins];
          if (!batch_map.count(lhs)) {
            for (int i = 0; i < lhs_batch_dims.size(); ++i) {
              if (value == i) {
                batch_map[lhs] = lhs_batch_dims[i];
                break;
              }
            }
            if (!lhs_space_dims.empty() && value == space_base_dim) {
              batch_map[lhs] = lhs_space_dims[0];
            }
          }

          if (!batch_map.count(rhs)) {
            for (int i = 0; i < rhs_batch_dims.size(); ++i) {
              if (value == i) {
                batch_map[rhs] = rhs_batch_dims[i];
                break;
              }
            }
            if (!rhs_space_dims.empty() && value == space_base_dim + 1) {
              batch_map[rhs] = rhs_space_dims[0];
            }
          }
        }

        break;
      }
      case HloOpcode::kConvolution: {
        const HloInstruction* lhs = ins->operand(0);
        const HloInstruction* rhs = ins->operand(1);
        const auto& conv_dnums = ins->convolution_dimension_numbers();

        if (batch_map.count(ins)) {
          int value = batch_map[ins];
          if (value == conv_dnums.output_batch_dimension() &&
              !batch_map.count(lhs)) {
            batch_map[lhs] = conv_dnums.input_batch_dimension();
          }

          if (value == conv_dnums.output_feature_dimension() &&
              !batch_map.count(rhs)) {
            batch_map[rhs] = conv_dnums.kernel_output_feature_dimension();
          }
        }

        break;
      }
      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        // We only handle one case for now:
        // If gather/scatter does not happen on the batch dimension,
        // then we can propagate the batch dim.
        if (batch_map.count(ins)) {
          int value = batch_map[ins];
          const HloInstruction* operand = ins->operand(0);
          if (ins->shape().rank() == operand->shape().rank() &&
              ins->shape().dimensions(value) ==
                  operand->shape().dimensions(value)) {
            batch_map[operand] = value;
          }
        }
        break;
      }
      case HloOpcode::kSort: {
        if (batch_map.count(ins)) {
          int value = batch_map[ins];
          if (!absl::c_linear_search(ins->dimensions(), value)) {
            for (size_t i = 0; i < ins->operand_count(); ++i) {
              const HloInstruction* operand = ins->operand(i);
              batch_map[operand] = value;
            }
          }
        }
        break;
      }
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* source =
            PassThroughCustomCallMarkerGetSource(ins);
        if (batch_map.count(ins) && !batch_map.count(source)) {
          batch_map[source] = batch_map[ins];
        }
        break;
      }
      case HloOpcode::kTuple:
      case HloOpcode::kCustomCall:
        break;
      default:
        break;
    }
  }

  // Print batch map for debugging
  // std::cerr << "Batch dim map begin" << std::endl;
  // for (const HloInstruction* ins : instructions) {
  //   std::cerr << ins->ToString(HloPrintOptions::ShortParsable());
  //   if (batch_map.count(ins)) {
  //     std::cerr << " BATCH " << batch_map[ins] << std::endl;
  //   } else {
  //     std::cerr << " NOBATCH " << std::endl;
  //   }
  // }
  // std::cerr << "Batch dim map end" << std::endl;
  // exit(-1);

  return batch_map;
}


// Remove duplicated strategies with the same output sharding spec.
void RemoveDuplicatedStrategy(std::unique_ptr<StrategyVector>& strategies) {
  std::vector<ShardingStrategy> new_vector;
  absl::flat_hash_set<HloSharding> added;

  CHECK(!strategies->is_tuple);

  for (size_t i = 0; i < strategies->leaf_vector.size(); ++i) {
    if (!added.count(strategies->leaf_vector[i].output_sharding)) {
      added.insert(strategies->leaf_vector[i].output_sharding);
      new_vector.push_back(std::move(strategies->leaf_vector[i]));
    }
  }

  strategies->leaf_vector = std::move(new_vector);
}


// Filter strategies according to the solver_option.force_batch_dim_to_mesh_dim.
// This can be used to forcibly generate data-parallel strategies.
Status FilterStrategy(const HloInstruction* ins,
                      std::unique_ptr<StrategyVector>& strategies,
                      const ClusterEnvironment& cluster_env,
                      const InstructionBatchDimMap& batch_map,
                      const AutoShardingSolverOption& solver_option) {
  int mesh_dim = solver_option.force_batch_dim_to_mesh_dim;
  int batch_dim = batch_map.at(ins);
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;

  if (ins->shape().dimensions(batch_dim) % device_mesh.dim(mesh_dim) != 0) {
    return tensorflow::errors::InvalidArgument(
        "The length of batch dimension is "
        "not divisible by the number of devices");
  }

  std::vector<ShardingStrategy> new_leaf_vector;
  for (auto& stra : strategies->leaf_vector) {
    std::vector<int> tensor_dim_to_mesh_dim =
        cluster_env.GetTensorDimToMeshDim(ins->shape(), stra.output_sharding);

    if (device_mesh.dim(mesh_dim) > 1) {
      // If the mesh dim is not one, the output tensor must be
      // tiled along the mesh dim.
      if (tensor_dim_to_mesh_dim[batch_dim] == mesh_dim) {
        new_leaf_vector.push_back(std::move(stra));
      }
    } else {
      // If the mesh dim is one, the output tensor must be replicated
      // on the mesh dim.
      if (tensor_dim_to_mesh_dim[batch_dim] == -1) {
        new_leaf_vector.push_back(std::move(stra));
      }
    }
  }
  CHECK(!new_leaf_vector.empty())
      << ins->ToString() << " does not have any valid strategies";
  strategies->leaf_vector = std::move(new_leaf_vector);

  return Status::OK();
}

inline std::pair<int, int> ParseMeshDims(const std::string& strategy_name) {
  if (strategy_name.find("{0,1}") != std::string::npos) {
    return {0, 1};
  } else {
    return {1, 0};
  }
}

// Return whether the tensor shape is divisible by
// the number of devices along multiple dimensions.
bool IsDivisible(const HloInstruction* ins, const Array<int64_t>& device_mesh,
                 const std::vector<int64_t>& tensor_dims,
                 const std::vector<int64_t>& mesh_dims) {
  CHECK_EQ(tensor_dims.size(), mesh_dims.size());
  for (int64_t i = 0; i < tensor_dims.size(); ++i) {
    if (ins->shape().dimensions(tensor_dims[i]) %
            device_mesh.dim(mesh_dims[i]) !=
        0) {
      return false;
    }
  }
  return true;
}

// Return the output sharding of the reduce-scatter variant of a given strategy.
HloSharding GetReduceScatterOutput(const HloInstruction* ins,
                                   const ShardingStrategy& strategy,
                                   const ClusterEnvironment& cluster_env) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d;

  if (ins->opcode() == HloOpcode::kDot) {
    const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
    int64_t space_base_dim = dot_dnums.lhs_batch_dimensions_size();

    if (StrStartsWith(strategy.name, "SR = SS x SR") ||
        StrStartsWith(strategy.name, "RS = RS x SS")) {
      int mesh_dim0, mesh_dim1;
      std::tie(mesh_dim0, mesh_dim1) = ParseMeshDims(strategy.name);

      if (!IsDivisible(ins, device_mesh, {space_base_dim, space_base_dim + 1},
                       {mesh_dim0, mesh_dim1})) {
        // XLA supports uneven partitioning by adding padding.
        // However, the ShardingSpec in Jax does not support uneven
        // partitioning.
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim, space_base_dim + 1},
                  {mesh_dim0, mesh_dim1}, device_mesh);
    } else if (StrStartsWith(strategy.name, "SbR = SbSk x SbSk")) {
      int mesh_dim0, mesh_dim1;
      std::tie(mesh_dim0, mesh_dim1) = ParseMeshDims(strategy.name);

      if (!IsDivisible(ins, device_mesh, {0, space_base_dim},
                       {mesh_dim0, mesh_dim1})) {
        // XLA supports uneven partitioning by adding padding.
        // However, the ShardingSpec in Jax does not support uneven
        // partitioning.
        return Undefined();
      }

      return Tile(ins->shape(), {0, space_base_dim}, {mesh_dim0, mesh_dim1},
                  device_mesh);
    } else if (StrStartsWith(strategy.name, "RR = RS x SR")) {
      int mesh_dim = strategy.name.find("{0}") != std::string::npos ? 0 : 1;

      if (!IsDivisible(ins, device_mesh, {space_base_dim}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, device_mesh);
    } else if (StrStartsWith(strategy.name, "R = Sk x Sk")) {
      int mesh_dim = 0;

      if (!IsDivisible(ins, device_mesh_1d, {space_base_dim}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {space_base_dim}, {mesh_dim}, device_mesh_1d);
    }
  } else if (ins->opcode() == HloOpcode::kConvolution) {
    const ConvolutionDimensionNumbers& conv_dnums =
        ins->convolution_dimension_numbers();
    int out_batch_dim = conv_dnums.output_batch_dimension();
    int out_out_channel_dim = conv_dnums.output_feature_dimension();

    if (StrStartsWith(strategy.name, "SR = SS x SR") ||
        StrStartsWith(strategy.name, "RS = RS x SS")) {
      int mesh_dim0, mesh_dim1;
      std::tie(mesh_dim0, mesh_dim1) = ParseMeshDims(strategy.name);

      if (!IsDivisible(ins, device_mesh, {out_batch_dim, out_out_channel_dim},
                       {mesh_dim0, mesh_dim1})) {
        return Undefined();
      }

      return Tile(ins->shape(), {out_batch_dim, out_out_channel_dim},
                  {mesh_dim0, mesh_dim1}, device_mesh);
    } else if (StrStartsWith(strategy.name, "R = Sk x Sk")) {
      int mesh_dim = 0;

      if (!IsDivisible(ins, device_mesh_1d, {out_batch_dim}, {mesh_dim})) {
        return Undefined();
      }

      return Tile(ins->shape(), {out_batch_dim}, {mesh_dim}, device_mesh_1d);
    }
  } else if (ins->opcode() == HloOpcode::kReduce) {
    // TODO(lmzheng): support more cases.
    CHECK_EQ(ins->shape().rank(), 1);

    int mesh_dim;
    if (strategy.name.find("allreduce @ [0]") != std::string::npos) {
      mesh_dim = 0;
    } else {
      mesh_dim = 1;
    }

    if (strategy.output_sharding.IsReplicated()) {
      if (strategy.name.find("1d") != std::string::npos) {
        if (!IsDivisible(ins, device_mesh_1d, {0}, {mesh_dim})) {
          return Undefined();
        }

        return Tile(ins->shape(), {0}, {mesh_dim}, device_mesh_1d);
      } else {
        if (!IsDivisible(ins, device_mesh, {0}, {mesh_dim})) {
          return Undefined();
        }

        return Tile(ins->shape(), {0}, {mesh_dim}, device_mesh);
      }
    } else {
      if (!IsDivisible(ins, device_mesh_1d, {0}, {0})) {
        return Undefined();
      }

      Array<int64_t> tile_assignment =
          strategy.output_sharding.tile_assignment();
      tile_assignment.Reshape({cluster_env.total_devices});
      return HloSharding::Tile(std::move(tile_assignment));
    }
  } else {
    LOG(FATAL) << "Invalid instruction: " << ins->ToString();
  }

  return Undefined();
}

// Return whether an instruction has the opportunity to generate reduce-scatter.
bool HasReduceScatterOpportunity(
    const HloInstruction* inst, const StrategyMap& strategy_map,
    const CostGraph& cost_graph, const std::vector<int64_t>& s_val,
    const absl::flat_hash_set<const HloInstruction*>& modified) {
  // If the operand is already modified by other ops, skip this instruction to
  // avoid conflicts.
  for (const HloInstruction* operand : inst->operands()) {
    if (modified.count(operand)) {
      return false;
    }
  }
  if (modified.count(inst)) {
    return false;
  }

  if (inst->opcode() == HloOpcode::kReduce && inst->shape().rank() == 1) {
    return true;
  }
  if (inst->opcode() == HloOpcode::kDot) {
    if (GetShardingStrategy(inst->operand(0)).output_sharding.IsReplicated() &&
        GetShardingStrategy(inst->operand(1)).output_sharding.IsReplicated()) {
      // This dot is replicated on all devices. Do not split it.
      // TODO(lmzheng): improve this condition.
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
                 const HloInstruction* shape_inst,
                 absl::flat_hash_set<const HloInstruction*>& modified) {
  CHECK(!to_split->shape().IsTuple()) << to_split->ToString();
  modified.insert(to_split);
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

// Pass through the custom call marker and get the actual user.
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

// DFS to find the replicated set starting from cur instruction.
void FindReplicateSet(
    HloInstruction* cur, const AliasMap& alias_map, const CostGraph& cost_graph,
    const std::vector<int64_t>& s_val, const StrategyMap& strategy_map,
    const ShardingStrategy& strategy, const HloInstruction* output,
    bool do_all_gather_after_backward, HloInstruction*& transpose_inst,
    absl::flat_hash_set<HloInstruction*>& replicated_set,
    absl::flat_hash_set<HloInstruction*>& boundary_set,
    absl::flat_hash_set<HloInstruction*>& consumer_set,
    absl::flat_hash_set<const HloInstruction*>& visited) {
  visited.insert(cur);

  // Check whether the node is a boundary node.
  absl::flat_hash_set<HloInstruction*> users =
      UsersWithAlias(cur, alias_map, output);
  for (HloInstruction* consumer : users) {
    const HloInstruction* shape_inst = cur;

    // Allow at most one transpose
    if (consumer->opcode() == HloOpcode::kTranspose &&
        (transpose_inst == nullptr ||
         DimensionsEqual(transpose_inst->shape(), consumer->shape()))) {
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
      FindReplicateSet(consumer, alias_map, cost_graph, s_val, strategy_map,
                       strategy, output, do_all_gather_after_backward,
                       transpose_inst, replicated_set, boundary_set,
                       consumer_set, visited);
    }
  }

  for (size_t i = 0; i < cur->operand_count(); ++i) {
    HloInstruction* operand = cur->mutable_operand(i);
    operand = PassThroughCustomCallMarkerOperand(operand, cur);

    if (!visited.count(operand) && !IsAlwaysReplicated(operand) &&
        GetShardingStrategy(operand).output_sharding ==
            strategy.output_sharding &&
        DimensionsEqual(operand->shape(), cur->shape())) {
      FindReplicateSet(operand, alias_map, cost_graph, s_val, strategy_map,
                       strategy, output, do_all_gather_after_backward,
                       transpose_inst, replicated_set, boundary_set,
                       consumer_set, visited);
    }
  }
}

// Try to reduce the boundary set to its common ancestor
void TryReduceWithCommonAncestor(
    absl::flat_hash_set<HloInstruction*>& replicated_set,
    absl::flat_hash_set<HloInstruction*>& boundary_set,
    absl::flat_hash_set<HloInstruction*>& consumer_set,
    const AliasMap& alias_map) {
  if (boundary_set.size() != 2) {
    return;
  }

  HloInstruction* ancestor = nullptr;
  absl::flat_hash_set<HloInstruction*> path;
  for (HloInstruction* node : boundary_set) {
    HloInstruction* cur = node;
    while (cur->operand_count() == 1) {
      HloInstruction* operand =
          PassThroughCustomCallMarkerOperand(cur->mutable_operand(0), cur);
      if (replicated_set.count(operand)) {
        path.insert(cur);
      }
      cur = operand;
    }

    if (ancestor == nullptr) {
      ancestor = cur;
    } else {
      if (ancestor != cur) {
        // The nodes in boundary set do not have a common ancestor.
        // This reduction fails.
        return;
      }
    }
  }
  if (ancestor == nullptr) {
    return;
  }

  // Find a common ancestor, reduce the boundary set
  boundary_set.clear();
  boundary_set.insert(ancestor);
  for (auto x : path) {
    replicated_set.erase(x);
  }
  consumer_set.insert(ancestor);
}

void UseAllReduceForGradAcc(
    absl::flat_hash_set<HloInstruction*>& replicated_set,
    const HloInstruction* inst) {
  if (inst->users().size() != 1) {
    return;
  }

  // Find the add instruction for grad accumulation, skip the identity marker
  // for remat and other elementwise ops.
  const HloInstruction* add =
      PassThroughCustomCallMarkerUser(inst->users().front(), inst);
  if (add->opcode() == HloOpcode::kGetTupleElement ||
      add->opcode() == HloOpcode::kTranspose) {
    if (add->users().size() != 1) {
      return;
    }
    add = add->users().front();
  }

  if (add->opcode() == HloOpcode::kAdd) {
    // Skip multiple adds introduced by AllReduceReassociate.
    while (add->users().size() == 1 &&
           add->users().front()->opcode() == HloOpcode::kAdd) {
      add = add->users().front();
    }
    CHECK_EQ(add->users().size(), 1);
    // Skip the end marker of backward computation
    add = PassThroughCustomCallMarkerUser(add->users().front(), add);

    // Do not partition the dot, add and parameter, so we can generate
    // all-reduce for grad accumulation.
    std::function<void(const HloInstruction*)> dfs_remove;
    dfs_remove = [&](const HloInstruction* cur) {
      if (!replicated_set.count(cur)) {
        return;
      }

      replicated_set.erase(cur);
      for (auto x : cur->operands()) {
        dfs_remove(PassThroughCustomCallMarkerOperand(x, cur));
      }
    };

    dfs_remove(add);
  }
}

// Substitute all-reduce strategies with their reduce-scatter variants.
void GenerateReduceScatter(const HloInstructionSequence& sequence,
                           const AliasMap& alias_map,
                           const InstructionDepthMap& depth_map,
                           const StrategyMap& strategy_map,
                           const CostGraph& cost_graph,
                           const std::vector<int64_t>& s_val,
                           const ClusterEnvironment& cluster_env,
                           const AutoShardingSolverOption& solver_option) {
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  // Propagation ends at output
  const HloInstruction* output = instructions.back();
  if (IsCustomCallMarker(output)) {
    output = output->operand(0);
  }

  // A debug option: whether to do all-gather after backward pass.
  // This controls the location of all-gather.
  // If true, all-gather happens after backward pass, which is desired for
  // gradient accumulation. If false, all-gather happens before forward pass,
  // which can partitions more tensors.
  bool do_all_gather_after_backward = true;

  // If true, do not actually generate reduce-scatter + all-gather,
  // but generate all-reduce + all-gather instead.
  // This saves less memory but is more friendly to gradient accumulation.
  // This is a temporary workaround due to implementation difficulty.
  // Ideally, we should be able to generate a gradient-accumulation-friendly
  // reduce-scatter + all-gather, but for now it is not easy to implement this
  // in our current system. So we generate a gradient-accumulation-friendly
  // all-reduce + all-gather, which has the same memory consumption but with 50%
  // communication overhead.
  bool use_all_reduce_for_grad_acc =
      solver_option.reduce_scatter_grad_acc_friendly;
  int verbose = 0;

  std::vector<HloInstruction*> insert_all_gather;
  absl::flat_hash_set<const HloInstruction*> modified;

  for (HloInstruction* inst : instructions) {
    if (!HasReduceScatterOpportunity(inst, strategy_map, cost_graph, s_val,
                                     modified)) {
      continue;
    }
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

    // Find the replicated set starting from the all-reduce instruction.
    visited.insert(output);
    FindReplicateSet(inst, alias_map, cost_graph, s_val, strategy_map, strategy,
                     output, do_all_gather_after_backward, transpose_inst,
                     replicated_set, boundary_set, consumer_set, visited);

    // Try to reduce the boundary set to its common ancestor
    TryReduceWithCommonAncestor(replicated_set, boundary_set, consumer_set,
                                alias_map);

    // Analyze the instructions after which all-gather should be inserted.
    std::vector<HloInstruction*> need_all_gather;
    for (HloInstruction* node : boundary_set) {
      if (consumer_set.count(node)) {
        if (AllUsersAreReduce(node)) {
          // If users are reduce, the all-gather cost after this instruction
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

    // Print replicated set and boundary set for debugging.
    StdCerr(verbose) << inst->ToString(HloPrintOptions::ShortParsable())
                     << "\n";
    StdCerr(verbose) << "replicated set (#parameter: "
                     << num_replicated_parameters << "):\n";
    for (auto x : replicated_set) {
      StdCerr(verbose) << "  " << x->ToString(HloPrintOptions::ShortParsable())
                       << "\n";
    }
    StdCerr(verbose) << "boundary set (#incompatible: "
                     << need_all_gather.size() << "):\n";
    for (auto x : boundary_set) {
      StdCerr(verbose) << "  " << x->ToString(HloPrintOptions::ShortParsable())
                       << " " << absl::c_linear_search(need_all_gather, x)
                       << "\n";
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

      if (use_all_reduce_for_grad_acc) {
        UseAllReduceForGradAcc(replicated_set, inst);
      }

      for (HloInstruction* to_split : replicated_set) {
        SetSharding(to_split, output_spec, inst, transpose_inst, modified);
      }

      if (!solver_option.reduce_scatter_aggressive_partition) {
        // The normal case
        for (HloInstruction* to_split : need_all_gather) {
          SetSharding(to_split, output_spec, inst, transpose_inst, modified);

          if (!do_all_gather_after_backward && to_split->users().size() == 1 &&
              to_split->users().front() == output &&
              alias_map.count(to_split)) {
            // Move the all-gather to its alias parameter.
            // This partitions more tensors but introduces communication
            // in the forward pass, which is not desired in gradient
            // accumulation.
            SetSharding(alias_map.at(to_split), output_spec, inst,
                        transpose_inst, modified);
            insert_all_gather.push_back(alias_map.at(to_split));
          } else {
            insert_all_gather.push_back(to_split);

            if (to_split->opcode() == HloOpcode::kGetTupleElement &&
                IsCustomCallMarker(to_split->operand(0)) &&
                to_split->users().size() == 1 &&
                to_split->users().front() == output) {
              insert_all_gather.push_back(PassThroughCustomCallMarkerOperand(
                  to_split->mutable_operand(0), to_split));
            }
          }
        }
      } else {
        // Aggressively partition more parameter tensors.
        // This can result in a strategy similar to ZeRO stage 3.
        // NOTE: The combination of this branch with pipeline parallel is not
        // tested.
        for (HloInstruction* to_split : need_all_gather) {
          SetSharding(to_split, output_spec, inst, transpose_inst, modified);

          if (to_split->users().size() == 1 &&
              to_split->users().front() == output &&
              alias_map.count(to_split)) {
            // Move the all-gather to its alias parameter.
            HloInstruction* param = alias_map.at(to_split);

            // Find the branching point (i.e., skip elementwise ops like
            // convert)
            HloInstruction* cur = param;
            while (cur->users().size() == 1) {
              // TODO(lmzheng): handle tuple.
              CHECK(cur->shape().IsArray());
              SetSharding(cur, output_spec, inst, transpose_inst, modified);
              cur = cur->users().front();
            }
            SetSharding(cur, output_spec, inst, transpose_inst, modified);

            CHECK(!cur->users().empty());

            // Find the first user
            HloInstruction* first_user = nullptr;
            int64_t min_depth = ((int64_t)1) << 50;
            for (const auto& x : cur->users()) {
              auto iter = depth_map.find(x);
              if (iter == depth_map.end()) {
                LOG(FATAL) << "ERROR: " << x->ToString() << std::endl;
              }
              if (x->opcode() != HloOpcode::kConvolution &&
                  x->opcode() != HloOpcode::kDot) {
                // Only apply this aggressive optimization for dot and conv
                continue;
              }
              if (iter->second < min_depth) {
                first_user = x;
                min_depth = iter->second;
              }
            }

            if (first_user != nullptr) {
              // Insert an identity to prevent CSE of all-gather
              HloInstruction* identity = inst->parent()->AddInstruction(
                  HloInstruction::CreateCustomCall(cur->shape(), {cur},
                                                   kIdentityMarker));
              SetSharding(identity, output_spec, inst, transpose_inst,
                          modified);
              ReplaceOperand(first_user, cur, identity);
            }
          }
        }
      }
    }

    StdCerr(verbose) << "-----------------------done\n" << std::endl;
  }

  // Insert all-gather on the output of boundary nodes by setting
  // their shardings. This also works as CSE of all-gather.
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

// Get the values in an array along a dimension.
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

bool IsValidTileAssignment(const HloSharding& spec) {
  if (IsUndefined(spec)) {
    return false;
  }

  if (spec.IsReplicated()) {
    return true;
  }

  // Check all tile dims
  const Array<int64_t>& tile_assignment = spec.tile_assignment();
  for (int i = 0; i < tile_assignment.num_dimensions(); i++) {
    if (tile_assignment.dim(i) != 1) {
      std::vector<int64_t> device_ids =
          GetValuesAlongOneDim(tile_assignment, i);
      int64_t delta;
      bool success;
      std::tie(delta, success) = CheckArithmeticSequence(device_ids);
      if (!success) {
        return false;
      }
    }
  }

  return true;
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
  CHECK_EQ(ct, tile_dims.size());

  return std::make_pair(ret, tile_dims.size());
}

void FixMixedMeshShapeResharding(HloInstruction* inst, int operand_num,
                                 const HloSharding& dst_sharding,
                                 const Array<int64_t>& device_mesh,
                                 ReshardingCache* resharding_cache) {
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
  std::vector<std::pair<HloSharding, HloInstruction*>>* cache_vector = nullptr;
  if (resharding_cache != nullptr) {
    cache_vector = &((*resharding_cache)[operand]);
    for (auto& entry : *cache_vector) {
      if (entry.first == dst_sharding) {
        replace_with = entry.second;
      }
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

    // Find an intermediate shape
    std::vector<int64_t> inter_shape_dims;

    for (size_t i = 0; i < shape.rank(); ++i) {
      if (sharding_1d->tile_assignment().dim(i) == 1) {
        inter_shape_dims.push_back(shape.dimensions(i));
      } else {
        CHECK(shape.dimensions(i) % device_mesh.dim(0) == 0)
            << "Only support even partition";
        inter_shape_dims.push_back(device_mesh.dim(0));
        inter_shape_dims.push_back(shape.dimensions(i) / device_mesh.dim(0));
      }
    }
    Shape inter_shape =
        ShapeUtil::MakeShape(shape.element_type(), inter_shape_dims);

    absl::optional<HloSharding> src_inter_sharding =
        hlo_sharding_util::ReshapeSharding(shape, inter_shape, src_sharding);
    absl::optional<HloSharding> dst_inter_sharding =
        hlo_sharding_util::ReshapeSharding(shape, inter_shape, dst_sharding);
    if (!src_inter_sharding.has_value() || !dst_inter_sharding.has_value()) {
      src_inter_sharding = HloSharding::Replicate();
      dst_inter_sharding = HloSharding::Replicate();
      LOG(WARNING) << "Invalid mixed mesh shape resharding.";
    }

    HloInstruction* src_inter = inst->parent()->AddInstruction(
        HloInstruction::CreateReshape(inter_shape, operand));
    src_inter->set_sharding(*src_inter_sharding);

    HloInstruction* dst_inter = inst->parent()->AddInstruction(
        HloInstruction::CreateReshape(inter_shape, src_inter));
    dst_inter->set_sharding(*dst_inter_sharding);

    replace_with = inst->parent()->AddInstruction(
        HloInstruction::CreateReshape(shape, dst_inter));
    replace_with->set_sharding(dst_sharding);
    if (cache_vector != nullptr) {
      cache_vector->push_back({dst_sharding, replace_with});
    }
  } else {
    replace_with = inst->parent()->AddInstruction(
        HloInstruction::CreateReshape(operand->shape(), operand));
    replace_with->set_sharding(dst_sharding);
    if (cache_vector != nullptr) {
      cache_vector->push_back({dst_sharding, replace_with});
    }
  }

  inst->ReplaceOperandWith(operand_num, replace_with);
}

template <typename T>
inline std::vector<int> Argsort(const std::vector<T>& scores) {
  std::vector<int> index;
  index.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    index.push_back(i);
  }
  auto cmp = [&scores](int l, int r) { return scores[l] > scores[r]; };
  std::sort(index.begin(), index.end(), cmp);
  return index;
}

void AnnotateShardingWithSimpleHeuristic(
    HloModule* module, const std::string& heuristic, const AliasMap& alias_map,
    const ClusterEnvironment& cluster_env) {
  const Array<int64_t>& device_mesh = cluster_env.device_mesh;
  const Array<int64_t>& device_mesh_1d = cluster_env.device_mesh_1d;
  int64_t num_devices = device_mesh.num_elements();

  // Count the non-one mesh dimension.
  size_t mesh_nn_dims = 0;
  for (int dim : device_mesh.dimensions()) {
    if (dim > 1) {
      mesh_nn_dims++;
    }
  }

  // Shard instructions
  HloComputation* entry_computation = module->entry_computation();
  for (HloInstruction* inst : entry_computation->instructions()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      HloSharding output_spec = HloSharding::Replicate();
      inst->set_sharding(output_spec);

      if (heuristic == "shard-largest") {
        std::vector<int64_t> lengths;
        for (int64_t i = 0; i < inst->shape().rank(); ++i) {
          lengths.push_back(inst->shape().dimensions(i));
        }

        std::vector<int> indices = Argsort(lengths);
        int common_dims = std::min(mesh_nn_dims, indices.size());

        if (common_dims < 1) {
          continue;
        }

        if (common_dims == 1) {
          int dim = indices[0];
          int length = lengths[dim];
          if (length % num_devices == 0) {
            output_spec = Tile(inst->shape(), {dim}, {0}, device_mesh_1d);
          }
        } else {
          int dim1 = indices[0];
          int length1 = lengths[dim1];
          int dim0 = indices[1];
          int length0 = lengths[dim0];

          if (length0 % device_mesh.dim(0) == 0 &&
              length1 % device_mesh.dim(1) == 0) {
            output_spec =
                Tile(inst->shape(), {dim0, dim1}, {0, 1}, device_mesh);
          }
        }
      } else if (heuristic == "shard-first") {
        if (inst->shape().rank() > 0 &&
            inst->shape().dimensions(0) % num_devices == 0) {
          output_spec = Tile(inst->shape(), {0}, {0}, device_mesh_1d);
        }
      } else if (heuristic == "shard-last") {
        int64_t last_dim = inst->shape().rank() - 1;
        if (inst->shape().rank() > 0 &&
            inst->shape().dimensions(last_dim) % num_devices == 0) {
          output_spec = Tile(inst->shape(), {last_dim}, {0}, device_mesh_1d);
        }
      } else {
        LOG(FATAL) << "Invalid heuristic: " << heuristic;
      }

      inst->set_sharding(output_spec);
      // std::cerr << "ins: " << inst->ToString() << ", spec: " <<
      // output_spec.ToString() << std::endl;
    } else if (inst->opcode() == HloOpcode::kDot) {
      const HloInstruction* lhs = inst->operand(0);
      const HloInstruction* rhs = inst->operand(1);
      const DotDimensionNumbers& dot_dnums = inst->dot_dimension_numbers();
      std::vector<int64_t> lhs_space_dims, rhs_space_dims;
      std::tie(lhs_space_dims, rhs_space_dims) =
          GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);
    }
  }

  // Meet the alias requirement for the output tuple.
  HloInstruction* output = entry_computation->root_instruction();
  const Shape& out_shape = output->shape();
  ShapeTree<HloSharding> tuple_sharding(out_shape, HloSharding::Replicate());
  std::vector<HloSharding> flattened_shardings;

  std::function<void(HloInstruction*)> get_flattened_shardings;
  get_flattened_shardings = [&](HloInstruction* cur) {
    for (int64_t i = 0; i < cur->operand_count(); ++i) {
      HloInstruction* operand = cur->mutable_operand(i);

      if (operand->shape().IsTuple()) {
        get_flattened_shardings(operand);
      } else {
        if (alias_map.count(operand)) {
          operand = alias_map.at(operand);
        }
        if (!operand->has_sharding()) {
          operand->set_sharding(HloSharding::Replicate());
        }
        CHECK(operand->has_sharding());
        flattened_shardings.push_back(operand->sharding());
      }
    }
  };
  get_flattened_shardings(output);
  int i = 0;
  for (auto& leaf : tuple_sharding.leaves()) {
    leaf.second = flattened_shardings[i++];
  }
  CHECK_EQ(i, flattened_shardings.size());
  output->set_sharding(HloSharding::Tuple(tuple_sharding));
}

StatusOr<bool> NormalizeDotDimension(HloModule* module) {
  bool changed = false;

  for (HloComputation *computation : module->MakeNonfusionComputations()) {
    std::vector<HloInstruction*> to_remove;

    for (HloInstruction *ins : computation->instructions()) {
      if (ins->opcode() == HloOpcode::kDot) {
        HloInstruction* lhs = ins->mutable_operand(0);
        HloInstruction* rhs = ins->mutable_operand(1);
        const DotDimensionNumbers& dot_dnums = ins->dot_dimension_numbers();
        const auto& lhs_con_dims = dot_dnums.lhs_contracting_dimensions();
        const auto& rhs_con_dims = dot_dnums.rhs_contracting_dimensions();
        std::vector<int64_t> lhs_space_dims, rhs_space_dims;
        std::tie(lhs_space_dims, rhs_space_dims) =
            GetSpaceDims(lhs->shape(), rhs->shape(), dot_dnums);

        CHECK(lhs_space_dims.size() <= 1 && rhs_space_dims.size() <= 1 &&
              lhs_con_dims.size() == 1 && rhs_con_dims.size() == 1)
          << "Invalid dot. Call DotDecomposer before this pass.";

        if (lhs_space_dims.size() == 0 || rhs_space_dims.size() == 0) {
          CHECK(lhs_space_dims.size() != 0 || rhs_space_dims.size() != 0) << ins->ToString();

          HloInstruction* new_lhs = lhs, *new_rhs = rhs;
          std::vector<int64_t> new_dot_shape_dims(ins->shape().dimensions().begin(),
                                                  ins->shape().dimensions().end());
		  DotDimensionNumbers new_dnums = dot_dnums;

          if (lhs_space_dims.size() == 0) {
            // Create a new lhs
            std::vector<int64_t> new_lhs_shape_dims(lhs->shape().dimensions().begin(),
                                                    lhs->shape().dimensions().end());
            new_lhs_shape_dims.insert(new_lhs_shape_dims.begin() + dot_dnums.lhs_batch_dimensions_size(),
                                      1);
            Shape new_lhs_shape = ShapeUtil::MakeShape(
                lhs->shape().element_type(), new_lhs_shape_dims);
            new_lhs = ins->parent()->AddInstruction(HloInstruction::CreateReshape(new_lhs_shape, lhs));

            new_dot_shape_dims.insert(new_dot_shape_dims.begin() + dot_dnums.lhs_batch_dimensions_size(),
                                      1);
            (*new_dnums.mutable_lhs_contracting_dimensions())[0] += 1;
          }

          if (rhs_space_dims.size() == 0) {
            // Create a new rhs
            std::vector<int64_t> new_rhs_shape_dims(rhs->shape().dimensions().begin(),
                                                    rhs->shape().dimensions().end());
            new_rhs_shape_dims.insert(new_rhs_shape_dims.begin() + dot_dnums.rhs_batch_dimensions_size(),
                                      1);
            Shape new_rhs_shape = ShapeUtil::MakeShape(
                rhs->shape().element_type(), new_rhs_shape_dims);
            new_rhs = ins->parent()->AddInstruction(HloInstruction::CreateReshape(new_rhs_shape, rhs));

            new_dot_shape_dims.insert(new_dot_shape_dims.begin() + dot_dnums.lhs_batch_dimensions_size() + 1,
                                      1);
            (*new_dnums.mutable_rhs_contracting_dimensions())[0] += 1;
          }

          Shape new_dot_shape = ShapeUtil::MakeShape(
              ins->shape().element_type(), new_dot_shape_dims);

          HloInstruction* new_dot = ins->parent()->AddInstruction(
              HloInstruction::CreateDot(new_dot_shape, new_lhs, new_rhs,
                                        new_dnums, ins->precision_config()));

          HloInstruction* new_ins = ins->parent()->AddInstruction(
              HloInstruction::CreateReshape(ins->shape(), new_dot));
          ins->ReplaceAllUsesWith(new_ins);
          to_remove.push_back(ins);
        }
      }
    }

    for (HloInstruction* ins : to_remove) {
      computation->RemoveInstruction(ins);
    }
  }

  return changed;
}

}  // namespace spmd
}  // namespace xla
