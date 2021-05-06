#include <vector>
#include <algorithm>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace gpu {

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
    int index = indices[next_depth];

    if (index == -1) {
      for (int i = 0; i < array.dim(next_depth); ++i) {
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
int GetIndex(const std::vector<T>& v, const T& key) {
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

bool HloShardingEqual(const HloSharding& lhs, const HloSharding& rhs) {
  return lhs == rhs;
}


}  // namespace gpu
}  // namespace xla
