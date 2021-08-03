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

}  // namespace spmd
}  // namespace xla
