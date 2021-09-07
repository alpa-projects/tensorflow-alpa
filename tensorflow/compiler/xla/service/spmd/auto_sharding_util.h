#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_

#include <vector>
#include <algorithm>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace spmd {

#define CHECK_FLOAT_EQ(a, b) CHECK_LE(std::abs((a) - (b)), 1e-6)

/* Type alias */
// Map an instruction to its depth.
using InstructionDepthMap = absl::flat_hash_map<const HloInstruction*, int64>;
// Map an instruction to its batch dimension.
using InstructionBatchDimMap = absl::flat_hash_map<const HloInstruction*, int>;
// Map an instruction to its alias source parameter.
using AliasMap = absl::flat_hash_map<const HloInstruction*, HloInstruction*>;

/*
 * Array/Vector/Matrix Utility
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

// Return whether a string starts with another substring.
inline bool StrStartsWith(const std::string& a, const std::string& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.c_str(), a.c_str() + b.size(), b.c_str());
}

/*
 * Shape Utility
 */
// Get the bytes of an array shape without checking its layout.
// This is modified from ShapeUtil::ByteSizeOfElements (shape_util.cc)
inline int64 ByteSizeOfElementsNoCheck(const Shape& shape) {
  TF_DCHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(shape.IsArray());
  int64 allocated_element_count;

  // Disable this check. Otherwise, it raises a fatal error on HloOpcode::kIota
  // generated by jax dropout.
  //CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ShortDebugString();
  allocated_element_count = ShapeUtil::ElementsIn(shape);
  return allocated_element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
}

// Get the number of bytes of a shape
inline double GetBytes(const Shape& shape) {
  if (shape.IsArray()) {
    return ByteSizeOfElementsNoCheck(shape);
  }
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
};

// Return whether two shapes are equal in dimension.
// The element type and layout are ignored.
inline bool DimensionsEqual(const Shape& a, const Shape& b) {
  return Shape::Equal().IgnoreLayout().IgnoreElementType()(a, b);
}

/*
 * HloInstruction Utility
 */
// Get the space dimensions of a dot instruction
inline std::pair<std::vector<int64>, std::vector<int64>> GetSpaceDims(
  const Shape& lhs_shape,
  const Shape& rhs_shape,
  const DotDimensionNumbers& dnums) {
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

// Return the users of an instruction and its alias, excluding the final output tuple.
inline absl::flat_hash_set<HloInstruction*> UsersWithAlias(
    const HloInstruction* inst,
    const AliasMap& alias_map,
    const HloInstruction* output) {
  absl::flat_hash_set<HloInstruction*> users(inst->users().begin(), inst->users().end());
  auto iter = alias_map.find(inst);
  if (iter != alias_map.end()) {
    users.insert(iter->second->users().begin(), iter->second->users().end());
  }

  users.erase(output);

  return users;
}

// Return whether the instruction is always replicated.
// (e.g., constant, broadcasted constant, scalar)
bool IsAlwaysReplicated(const HloInstruction* inst);

// Return the number of users and exclude the output instruction.
inline int NumNonOutputUsers(const HloInstruction* inst, const HloInstruction* output) {
  int ret = 0;
  for (const auto x : inst->users()) {
    if (x != output) {
      ret++;
    }
  }
  return ret;
}

// Depth analysis (breadth first search) that compute the depth of each instruction.
// We also assign a much larger distance to heavey operators (e.g., dot, convolution).
InstructionDepthMap BuildInstructionDepthMap(const HloInstructionSequence& sequence);

// Batch dimension analysis that finds the batch dimension of each instruction.
InstructionBatchDimMap BuildInstructionBatchDimMap(const HloInstructionSequence& sequence);

/*
 * HloSharding Utility
 */
// We reuse "Manual" to represent "Undefined" sharding strategy.
// If an op has an"Undefined" strategy, it means auto-sharding pass does not
// decide the sharding strategy for this op. 
// We rely on the later sharding propagation pass to assign strategies to them.
inline HloSharding Undefined() {
  return HloSharding::Manual();
}

inline bool IsUndefined(const HloSharding& hlo_sharding) {
  return hlo_sharding.IsManual();
}

// Pretty print a HloSharding in a simplified form
inline std::string SimpleToString(const HloSharding& spec) {
  if (spec.IsReplicated()) {
    return "R";
  }
  return ToString(spec.tile_assignment().dimensions());
}

// Insert a copy of the operand to force the sharding of the operand
inline void ForceOperandSharding(HloInstruction* inst,
                                 int operand_num,
                                 const HloSharding& sharding) {
  HloInstruction* operand = inst->mutable_operand(operand_num);
  HloInstruction* replace_with = inst->parent()->AddInstruction(
    HloInstruction::CreateReshape(operand->shape(), operand));
  replace_with->set_sharding(sharding);
  inst->ReplaceOperandWith(operand_num, replace_with);
}

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

/*
 * Gradient accumulation
 */
// Find all instrctions that compute gradients in gradient accumulation.
// This is done by using the hint from pipeline_marker (gradient marker).
inline std::vector<const HloInstruction*> GetGradientComputationInstructions(
    const std::vector<HloInstruction*>& instructions) {
  std::vector<const HloInstruction*> ret;

  // Find the second pipeline_marker, which marks all gradient values.
  int ct = 0;
  for (size_t i = 0; i < instructions.size(); ++i) {
    const HloInstruction* ins = instructions[i];

    if (ins->IsCustomCall("xla_pipeline_marker")) {
      if(++ct == 2) {
        const HloInstruction* tuple = ins->operand(0);

        for (size_t j = 0; j < tuple->operand_count(); ++j) {
          const HloInstruction* add = tuple->operand(j);
          if (add->opcode() == HloOpcode::kAdd) {
            ret.push_back(add->operand(1));
          }
        }
      }
    }
  }

  return ret;
}

}  // namespace spmd
}  // namespace xla

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_
