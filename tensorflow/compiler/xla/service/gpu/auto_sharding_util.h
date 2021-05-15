#include <vector>
#include <algorithm>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

/*
 * Array/Vector Utility
 */
// Append elements of `array` to `result`. The `indices` is a generalized
// multi-dimensional index that can index a whole row (use -1 to indicate this)
template <typename T>
void AppendFlattenElements(std::vector<T>* result,
                           const Array<T>& array,
                           const std::vector<int64> indices,
                           int cur_depth,
                           std::vector<int64> cur_indices) {
  if (cur_depth == array.num_dimensions() - 1) {
    result->push_back(array(cur_indices));
  } else {
    int next_depth = cur_depth + 1;
    int64 index = indices[next_depth];

    if (index == -1) {
      for (int64 i = 0; i < array.dim(next_depth); ++i) {
        cur_indices[next_depth] = i;
        AppendFlattenElements(result, array, indices, next_depth, cur_indices);
      }
    } else {
      cur_indices[next_depth] = index;
      AppendFlattenElements(result, array, indices, next_depth, cur_indices);
    }
  }
}

// Return the index of key in a vector. -1 means not found.
template <typename T>
int64 GetIndex(const std::vector<T>& v, const T& key) {
  auto iter = std::find(v.cbegin(), v.cend(), key);
  
  if (iter != v.cend()) {
    return std::distance(v.cbegin(), iter);
  } else {
    return -1;
  }
}

// Get the value of the last elemet in a dimension
template <typename T>
T GetDimLastValue(const Array<T>& array, int dim) {
  std::vector<int64> indices(array.num_dimensions(), 0);
  indices[dim] = array.dim(dim) - 1;
  return array(indices);
}

// Print a vector as string
template <typename T>
std::string ToString(const std::vector<T>& vector) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < vector.size(); ++i) {
    os << vector[i];
    if (i != vector.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os.str();
}

/*
 * Shape Utility
 */
// Get the number of bytes of a shape
double GetBytes(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
};

/*
 * HloInstruction Utility
 */
// Get the space dimensions of a dot instruction
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

/*
 * HloSharding Utility
 */
// Pretty print a HloSharding in a simplified form
std::string SimpleToString(const HloSharding& spec) {
  if (spec.IsReplicated()) {
    return "R";
  }
  return ToString(spec.tile_assignment().dimensions());
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

}  // namespace gpu
}  // namespace xla
