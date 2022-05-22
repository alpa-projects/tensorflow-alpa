#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_

#include <algorithm>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace spmd {

#define CHECK_FLOAT_EQ(a, b) CHECK_LE(std::abs((a) - (b)), 1e-6)

/* Type alias */
// Map an instruction to its depth.
using InstructionDepthMap = absl::flat_hash_map<const HloInstruction*, int64_t>;
// Map an instruction to its batch dimension.
using InstructionBatchDimMap = absl::flat_hash_map<const HloInstruction*, int>;
// Map an instruction to its alias source parameter.
using AliasMap = absl::flat_hash_map<const HloInstruction*, HloInstruction*>;
// Map an instruction to its resharding cache.
using ReshardingCache =
    absl::flat_hash_map<const HloInstruction*,
                        std::vector<std::pair<HloSharding, HloInstruction*>>>;

extern const char* const kPipelineMarker;
extern const char* const kIdentityMarker;
constexpr absl::string_view kPipelineMarkerStartType = "start";
constexpr absl::string_view kPipelineMarkerEndType = "end";

/*
 * Array/Vector/Matrix Utility
 */
// Append elements of `array` to `result`. The `indices` is a generalized
// multi-dimensional index that can index a whole row (use -1 to indicate this).
template <typename T>
void AppendFlattenElements(std::vector<T>* result, const Array<T>& array,
                           const std::vector<int64_t> indices, int cur_depth,
                           std::vector<int64_t> cur_indices) {
  if (cur_depth == array.num_dimensions() - 1) {
    result->push_back(array(cur_indices));
  } else {
    int next_depth = cur_depth + 1;
    int64_t index = indices[next_depth];

    if (index == -1) {
      for (int64_t i = 0; i < array.dim(next_depth); ++i) {
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
int64_t GetIndex(const std::vector<T>& v, const T& key) {
  auto iter = std::find(v.cbegin(), v.cend(), key);

  if (iter != v.cend()) {
    return std::distance(v.cbegin(), iter);
  } else {
    return -1;
  }
}

// Print a vector as string.
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
    CHECK(idx < n * m) << idx << " , " << n << " , " << m;
    return (*data)[idx];
  }

  double& operator()(size_t i, size_t j) {
    size_t idx;
    if (transpose) {
      idx = j * n + i;
    } else {
      idx = i * m + j;
    }
    CHECK(data != nullptr) << n << " , " << m;
    CHECK(idx < n * m) << idx << " , " << n << " , " << m;
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
// This is modified from ShapeUtil::ByteSizeOfElements (shape_util.cc).
inline int64_t ByteSizeOfElementsNoCheck(const Shape& shape) {
  TF_DCHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(shape.IsArray());
  int64_t allocated_element_count;

  // Disable this check. Otherwise, it raises a fatal error on HloOpcode::kIota
  // generated by jax dropout.
  // CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ShortDebugString();
  allocated_element_count = ShapeUtil::ElementsIn(shape);
  return allocated_element_count *
         ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
}

// Get the number of bytes of a shape.
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
// Get the space dimensions of a dot instruction.
inline std::pair<std::vector<int64_t>, std::vector<int64_t>> GetSpaceDims(
    const Shape& lhs_shape, const Shape& rhs_shape,
    const DotDimensionNumbers& dnums) {
  std::vector<int64_t> lhs_space_dims;
  std::vector<int64_t> rhs_space_dims;

  for (int64_t i = 0; i < lhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.lhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.lhs_contracting_dimensions(), i)) {
      continue;
    }
    lhs_space_dims.push_back(i);
  }

  for (int64_t i = 0; i < rhs_shape.rank(); ++i) {
    if (absl::c_linear_search(dnums.rhs_batch_dimensions(), i) ||
        absl::c_linear_search(dnums.rhs_contracting_dimensions(), i)) {
      continue;
    }
    rhs_space_dims.push_back(i);
  }
  return std::make_pair(std::move(lhs_space_dims), std::move(rhs_space_dims));
}

// Replace old operand with the new one.
inline void ReplaceOperand(HloInstruction* inst,
                           const HloInstruction* old_operand,
                           HloInstruction* new_operand) {
  for (int i = 0; i < inst->operand_count(); ++i) {
    if (inst->operand(i) == old_operand) {
      inst->ReplaceOperandWith(i, new_operand);
    }
  }
}

// Return whether this instruction is a custom call marker introduced by us.
inline bool IsCustomCallMarker(const HloInstruction* inst) {
  return inst->IsCustomCall(kPipelineMarker) ||
         inst->IsCustomCall(kIdentityMarker);
}

// Return whether the reshape is a special reshape that switches the batch dim
// of a dot.
bool IsBatchDimSwitchReshape(const HloInstruction* inst);

// Return whether the instruction is followed by a broadcast.
bool IsFollowedByBroadcast(const HloInstruction* inst);

// Return whether the instruction is followed by a reduce.
bool IsFollowedByReduce(const HloInstruction* inst);

// Return whether the instruction is an activation from another pipeline stage.
bool IsActivationFromAnotherStage(const HloInstruction* inst,
                                  const InstructionBatchDimMap& batch_dim_map);

// Depth analysis (breadth first search) that compute the depth of each
// instruction. We also assign a much larger distance to heavy operators (e.g.,
// dot, convolution).
InstructionDepthMap BuildInstructionDepthMap(
    const HloInstructionSequence& sequence,
    const InstructionBatchDimMap& batch_dim_map);

// Batch dimension analysis that finds the batch dimension of each instruction.
InstructionBatchDimMap BuildInstructionBatchDimMap(
    const HloInstructionSequence& sequence);

// Remove all custom call makers in an HloModule.
void RemoveCustomCallMarker(HloModule* module);

/*
 * HloSharding Utility
 */
// We reuse "Manual" to represent "Undefined" sharding strategy.
// If an op has an"Undefined" strategy, it means auto-sharding pass does not
// decide the sharding strategy for this op.
// We rely on the later sharding propagation pass to assign strategies to them.
inline HloSharding Undefined() { return HloSharding::Manual(); }

inline bool IsUndefined(const HloSharding& hlo_sharding) {
  return hlo_sharding.IsManual();
}

// Pretty print a HloSharding in a simplified form
inline std::string ToStringSimple(const HloSharding& spec) {
  if (spec.IsReplicated()) {
    return "R";
  }
  return ToString(spec.tile_assignment().dimensions());
}

// Insert a copy of the operand to force the sharding of the operand.
inline void ForceOperandSharding(HloInstruction* inst, int operand_num,
                                 const HloSharding& sharding) {
  HloInstruction* operand = inst->mutable_operand(operand_num);
  if (operand->sharding() == sharding) {
    return;
  }
  HloInstruction* replace_with = inst->parent()->AddInstruction(
      HloInstruction::CreateReshape(operand->shape(), operand));
  replace_with->set_sharding(sharding);
  inst->ReplaceOperandWith(operand_num, replace_with);
}

// Return whether the sharding is fully tiled.
inline bool IsFullyTiled(const HloSharding& sharding) {
  return sharding.NumTiles() == sharding.tile_assignment().num_elements();
}

// Propagate sharding for broadcast.
// The output will be tiled along the broadcasted dimension the same way
// as the input for the broadcast while the other dimensions are kept
// non-tiled.
HloSharding BroadcastSharding(const HloSharding& input_spec,
                              const Shape& new_shape,
                              const absl::Span<const int64_t>& dimensions);

// Propagate sharding for dim-wise operations (e.g., slice, pad) which works
// independently on each dimension.
// The sharding can successfully propagate if the operation only happens on
// tensor dimensions that are not tiled.
absl::optional<HloSharding> PropagateDimwiseSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Shape& new_shape);

// Propagate sharding for ReduceWindow-like operations.
// The sharding can successfully propagate if the window operation only happens
// on tensor dimensions that are not tiled.
absl::optional<HloSharding> PropagateReduceWindowSharding(
    const HloSharding& input_spec, const Shape& old_shape,
    const Window& window);

// Check whether the tile assignment of a HloSharding is valid for our system.
// Definition of validity:
// For every tile dimension, the device id sequence along that dimension has to
// be an arithmetic sequence.
// e.g., we don't allow specs like sharding={devices=[8,1] 0,4,1,5,2,7,3,8}
bool IsValidTileAssignment(const HloSharding& spec);

// Get the corresponding mesh dimension for every tensor dimension.
// The first return value maps ith tensor dim to ith mesh dim.
// A -1 means the tensor is replicated on that dimension.
// The second value is the number of mesh dimensions.
// -1 means the tensor is replicated on the whole the mesh
// (i.e., we cannot decide the number of mesh dims in this function).
std::pair<std::vector<int>, int> GetTensorDimToMeshDimInternal(
    const Shape& shape, const HloSharding& spec);

// Forcibly set the sharding of the operand of inst.
// Also fix the resharding between 1d and 2d logical mesh.
void FixMixedMeshShapeResharding(HloInstruction* inst, int operand_num,
                                 const HloSharding& dst_sharding,
                                 const Array<int64_t>& device_mesh,
                                 ReshardingCache* resharding_cache);

// Normalize the dimension number of dot.
// After normalization, each operand always have one contracting dim and
// one space dim. The number of batch dims is unrestricted.
StatusOr<bool> NormalizeDotDimension(HloModule* module);

/*
 * Gradient accumulation
 */
// Find all instructions that compute gradients in gradient accumulation.
// This is done by using the hint from pipeline_marker (gradient marker).
inline std::vector<const HloInstruction*> GetGradientComputationInstructions(
    const std::vector<HloInstruction*>& instructions) {
  std::vector<const HloInstruction*> ret;

  for (size_t i = 0; i < instructions.size(); ++i) {
    const HloInstruction* ins = instructions[i];

    if (ins->IsCustomCall(kPipelineMarker) &&
        (ins->metadata().op_name().find("compute_grad") != std::string::npos ||
         ins->metadata().op_name().find("backward") != std::string::npos) &&
        ins->metadata().op_type() == kPipelineMarkerEndType) {
      const HloInstruction* tuple = ins->operand(0);
      for (size_t j = 0; j < tuple->operand_count(); ++j) {
        const HloInstruction* add = tuple->operand(j);
        while (add->opcode() == HloOpcode::kAdd) {
          ret.push_back(add->operand(0));
          ret.push_back(add->operand(1));

          if (add->operand(0)->opcode() == HloOpcode::kAdd) {
            add = add->operand(0);
          } else {
            add = add->operand(1);
          }
        }
      }
    }
  }

  return ret;
}

/*
 * I/O Utility
 */

/*! \brief An empty output stream */
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream&) : std::ostream(nullptr) {}
  static NullStream& Global();
};

template <class T>
NullStream& operator<<(NullStream& os, const T& value) {
  return os;
}

/*! \brief Get std cout with verbose control */
inline std::ostream& StdCerr(int verbose, int setting = 1) {
  return verbose >= setting ? std::cerr : NullStream::Global();
}

}  // namespace spmd
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_AUTO_SHARDING_UTIL_H_
