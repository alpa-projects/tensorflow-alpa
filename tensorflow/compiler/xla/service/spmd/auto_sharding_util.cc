#include "tensorflow/compiler/xla/service/spmd/auto_sharding_util.h"

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
                              const std::vector<int64>& dimensions) {
  if (input_spec.IsReplicated()) {
    return input_spec;
  }
  CHECK(new_shape.IsArray());

  std::vector<int64> target_tile_assignment_dimensions;
  for (int64 i = 0; i < new_shape.rank(); ++i) {
    auto it = absl::c_find(dimensions, i);
    if (it == dimensions.end()) {
      target_tile_assignment_dimensions.push_back(1);
    } else {
      const int64 source_dim = std::distance(dimensions.begin(), it);
      target_tile_assignment_dimensions.push_back(
          input_spec.tile_assignment().dim(source_dim));
    }
  }
  if (input_spec.ReplicateOnLastTileDim()) {
    target_tile_assignment_dimensions.push_back(
        input_spec.tile_assignment().dimensions().back());
  }
  Array<int64> new_tile_assignment = input_spec.tile_assignment();
  new_tile_assignment.Reshape(target_tile_assignment_dimensions);

  return input_spec.ReplicateOnLastTileDim() ?
    HloSharding::PartialTile(new_tile_assignment):
    HloSharding::Tile(new_tile_assignment);
}

// Propagate sharding for dim-wise operations (e.g., slice, pad) which works
// independently on each dimension.
// The sharding can successfully propagate if the operation only happends on
// tensor dimentions that are not tiled.
absl::optional<HloSharding> PropagateDimwiseSharding(const HloSharding& input_spec,
                                                     const Shape& old_shape,
                                                     const Shape& new_shape) {
  if (input_spec.IsReplicated()) {
    return input_spec;
  }

  CHECK(old_shape.IsArray());

  const auto& tile_assignment = input_spec.tile_assignment();
  for (int64 i = 0; i < old_shape.rank(); ++i) {
    if (tile_assignment.dim(i) > 1 &&
        new_shape.dimensions(i) != old_shape.dimensions(i)) {
      return absl::nullopt;
    }
  }

  return input_spec;
}

// Depth analysis (breadth first search).
// We also assign a much larger distance to heavey operators (e.g., dot, convolution).
InstructionDepthMap BuildInstructionDepthMap(const HloInstructionSequence& sequence) {
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
          int delta = 0;

          // Heavy operators have more weight (distance).
          switch (node->opcode()) {
            case HloOpcode::kDot:
            case HloOpcode::kConvolution:
              delta = 1000;
              break;
            // A temporary hack here: reduce ops will generate replicated sharding.
            // We do not want the later broadcast and elementwise ops to follow it.
            // So we give reduce ops some penalty and let the elementwise ops to
            // follow other operands.
            // TODO(lmzheng): remove this hack by correctly registering strategies
            // for broadcast.
            case HloOpcode::kReduce:
              delta = -10;
              break;
            // For similar reasons mentioned above, we give some penalty to broadcast.
            case HloOpcode::kBroadcast:
              delta = -3;
              break;
            case HloOpcode::kConstant:
              delta = 0;
              break;
            default:
              delta = 1;
              break;
          }

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


// Batch dimension analysis that finds the batch dimension of each instruction.
InstructionBatchDimMap BuildInstructionBatchDimMap(const HloInstructionSequence& sequence) {
  InstructionBatchDimMap batch_map;
  const std::vector<HloInstruction*>& instructions = sequence.instructions();

  bool first_dot = true;

  for (const HloInstruction* ins : instructions) {
    switch (ins->opcode()) {
      case HloOpcode::kDot: {
        if (first_dot) {
          // We use the first dot as source to start batch dim propagation.
          // Assume the first dim of the first dot is batch dim.
          first_dot = false;
          batch_map[ins] = 0;
          break;
        }

        const HloInstruction* lhs = ins->operand(0);
        const HloInstruction* rhs = ins->operand(1);
        const auto& dot_dnums =  ins->dot_dimension_numbers();
        int64 space_base_dim = dot_dnums.lhs_batch_dimensions_size();
        const auto& lhs_batch_dims = ins->dot_dimension_numbers().lhs_batch_dimensions();
        const auto& rhs_batch_dims = ins->dot_dimension_numbers().rhs_batch_dimensions();
        std::vector<int64> lhs_space_dims, rhs_space_dims;
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
      case HloOpcode::kReduce: {
        const HloInstruction* operand = ins->operand(0);
        const auto& dimensions = ins->dimensions();

        if (batch_map.count(operand)) {
          int value = batch_map[operand];
          if (value == 0 && !absl::c_linear_search(dimensions, value)) {
            batch_map[operand] = value;
          }
        }
        break;
      }
      case HloOpcode::kSlice:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
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

      default: break;
    }
  }

  //for (auto iter : batch_map) {
  //  std::cerr << iter.first->ToString(HloPrintOptions::ShortParsable()) << " "
  //            << iter.second << std::endl;
  //}
  //exit(0);

  return batch_map;
}

}  // namespace spmd
}  // namespace xla
