#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_

#include <vector>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace spmd {

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

// A simple matrix class to store and manipulate the cost matrices on edges.
// It can create a view for matrix transpose without copying the memory.
class Matrix {
 public:
  Matrix() : n(0), m(0), transpose(false), data(nullptr) {}

  Matrix(size_t n, size_t m) {
    this->n = n;
    this->m = m;
    transpose = false;
    data = std::make_shared<std::vector<double>>(n * m, 0.0);
  }

  Matrix(size_t n, size_t m, bool transpose,
         std::shared_ptr<std::vector<double>> data) {
    this->n = n;
    this->m = m;
    this->transpose = transpose;
    this->data = data;
  }

  Matrix Transpose() { return Matrix(m, n, !transpose, data); }

  double operator()(size_t i, size_t j) const {
    size_t idx;
    if (transpose) {
      idx = j * n + i;
    } else {
      idx = i * m + j;
    }
    CHECK(data != nullptr) << n << " , " << m;
    return (*data)[idx];
  }

  double& operator()(size_t i, size_t j) {
    size_t idx;
    if (transpose) {
      idx = j * n + i;
    } else {
      idx = i * m + j;
    }
    CHECK(data != nullptr) << n << " . " << m;
    return (*data)[idx];
  }

  Matrix operator+(const Matrix& other) {
    CHECK_EQ(n, other.n);
    CHECK_EQ(m, other.m);
    Matrix ret = Matrix(n, m);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        ret(i, j) = operator()(i, j) + other(i, j);
      }
    }
    return ret;
  }

  std::string ToString() const {
    std::ostringstream os;

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        os << operator()(i, j) << " ";
      }
      os << "\n";
    }

    return os.str();
  }

  size_t n;
  size_t m;
  bool transpose;
  std::shared_ptr<std::vector<double>> data;
};

/*
 * Shape Utility
 */
// Get the bytes of an array shape without checking its layout.
// This is modified from ShapeUtil::ByteSizeOfElements (shape_util.cc)
int64 ByteSizeOfElementsNoCheck(const Shape& shape);

// Get the number of bytes of a shape
double GetBytes(const Shape& shape);

/*
 * HloInstruction Utility
 */
// Get the space dimensions of a dot instruction
std::pair<std::vector<int64>, std::vector<int64>> GetSpaceDims(
  const Shape& lhs_shape,
  const Shape& rhs_shape,
  const DotDimensionNumbers& dnums
);

/*
 * HloSharding Utility
 */
// Pretty print a HloSharding in a simplified form
std::string SimpleToString(const HloSharding& spec);

// We reuse "Manual" to represent "Undefined" sharding strategy.
// If an op has an"Undefined" strategy, it means auto-sharding pass does not
// decide the sharding strategy for this op. 
// We rely on the later sharding propagation pass to assign strategies to them.
HloSharding Undefined();

bool IsUndefined(const HloSharding& hlo_sharding);

// Propagate sharding for broadcast.
// The output will be tiled along the broadcasted dimension the same way
// as the input for the broadcast while the other dimensions are kept
// non-tiled.
HloSharding BroadcastSharding(const HloSharding& input_spec,
                              const Shape& new_shape,
                              const std::vector<int64>& dimensions);

// Propagate sharding for dim-wise operations (e.g., slice, pad) which works
// independently on each dimension.
// The sharding can successfully propagate if the operation only happends on
// tensor dimentions that are not tiled.
absl::optional<HloSharding> PropagateDimwiseSharding(const HloSharding& input_spec,
                                                     const Shape& old_shape,
                                                     const Shape& new_shape);

}  // namespace spmd
}  // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_
