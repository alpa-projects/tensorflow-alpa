/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"

#include <functional>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

GpuGemmConfig GetGpuGemmConfig(const HloInstruction *gemm) {
  GpuGemmConfig config;
  config.output_shape = gemm->shape();
  config.lhs_shape = gemm->operand(0)->shape();
  config.rhs_shape = gemm->operand(1)->shape();
  auto backend_config_or = gemm->backend_config<GemmBackendConfig>();
  config.backend_config = std::move(backend_config_or.ValueOrDie());
  return config;
}

GemmThunk::GemmThunk(ThunkInfo thunk_info, GpuGemmConfig config,
                     const BufferAllocation::Slice &lhs_buffer,
                     const BufferAllocation::Slice &rhs_buffer,
                     const BufferAllocation::Slice &output_buffer,
                     bool implements_whole_instruction)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer),
      implements_whole_instruction_(implements_whole_instruction) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams &params) {
  auto get_device_address = [&](const BufferAllocation::Slice &slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  VLOG(3) << "Running GEMM thunk";
  se::DeviceMemoryBase lhs_data = get_device_address(lhs_buffer_);
  se::DeviceMemoryBase rhs_data = get_device_address(rhs_buffer_);
  se::DeviceMemoryBase output_data = get_device_address(output_buffer_);
  return RunGemm(config_, lhs_data, rhs_data, output_data, params.stream,
                 implements_whole_instruction_, profile_index(),
                 params.profiler);
}

// This struct contains the metadata of a matrix, e.g., its base address and
// dimensions.
struct MatrixDescriptor {
  se::DeviceMemoryBase data;
  se::blas::Transpose transpose;
  int64 num_rows;
  int64 num_cols;

  int64 stride() const { return num_rows * num_cols; }

  int64 reduced_dim() const {
    return transpose == se::blas::Transpose::kTranspose ? num_rows : num_cols;
  }

  template <typename T>
  se::DeviceMemory<T> cast() const {
    return se::DeviceMemory<T>(data);
  }
};

// Converts from an XLA PrimitiveType to a blas::ComputationType, which is
// used to specify the precision with which matmul computations should be
// performed, separately from the precision of the inputs and result.
static absl::optional<se::blas::ComputationType> ComputationTypeFromPrimitive(
    PrimitiveType type) {
  switch (type) {
    case F16:
    case BF16:
      return se::blas::ComputationType::kF32;
    case F32:
      return se::blas::ComputationType::kF32;
    case F64:
      return se::blas::ComputationType::kF64;
    case C64:
      return se::blas::ComputationType::kComplexF32;
    case C128:
      return se::blas::ComputationType::kComplexF64;
    case S32:
      return se::blas::ComputationType::kI32;
    default:
      return absl::nullopt;
  }
}

template <typename Input, typename Output>
static Status DoGemmWithAlgorithm(
    int64_t batch_size, MatrixDescriptor lhs, MatrixDescriptor rhs,
    MatrixDescriptor output_matrix, Output alpha, Output beta,
    se::Stream *stream, se::blas::AlgorithmType algorithm,
    se::blas::ProfileResult *output_profile_result) {
  CHECK(output_matrix.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  se::blas::ComputationType computation_type =
      *ComputationTypeFromPrimitive(output_type);
  se::DeviceMemory<Output> output_data(output_matrix.data);

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatchedWithAlgorithm(
        lhs.transpose, rhs.transpose, output_matrix.num_rows,
        output_matrix.num_cols,
        /*size of reduce dim=*/lhs.reduced_dim(),
        /*alpha=*/alpha, lhs.cast<Input>(), lhs.stride(),
        /*leading dim of LHS=*/lhs.num_rows, rhs.cast<Input>(),
        /*leading dim of RHS=*/rhs.num_rows, rhs.stride(),
        /*beta=*/beta, &output_data,
        /*leading dim of output=*/output_matrix.num_rows,
        output_matrix.stride(), batch_size, computation_type, algorithm,
        output_profile_result);
  } else {
    return stream->ThenBlasGemmWithAlgorithm(
        lhs.transpose, rhs.transpose, output_matrix.num_rows,
        output_matrix.num_cols,
        /*size of reduce dim=*/lhs.reduced_dim(),
        /*alpha=*/alpha, lhs.cast<Input>(),
        /*lda=*/lhs.num_rows, rhs.cast<Input>(),
        /*ldb=*/rhs.num_rows,
        /*beta=*/beta, &output_data,
        /*ldc=*/output_matrix.num_rows, computation_type, algorithm,
        output_profile_result);
  }
}

template <typename Input>
static Status DoGemm(int64_t batch_size, const MatrixDescriptor &lhs,
                     const MatrixDescriptor &rhs,
                     const MatrixDescriptor &output_matrix, Input alpha,
                     Input beta, se::Stream *stream,
                     absl::optional<se::blas::AlgorithmType> algorithm,
                     se::blas::ProfileResult *output_profile_result) {
  CHECK(output_matrix.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Input> output_data(output_matrix.data);

  if (algorithm) {
    return DoGemmWithAlgorithm<Input, Input>(batch_size, lhs, rhs,
                                             output_matrix, alpha, beta, stream,
                                             *algorithm, output_profile_result);
  }

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatched(
        lhs.transpose, rhs.transpose, output_matrix.num_rows,
        output_matrix.num_cols, /*size of reduce dim=*/lhs.reduced_dim(),
        /*alpha=*/alpha, lhs.cast<Input>(),
        /*leading dim of LHS=*/lhs.num_rows, lhs.stride(), rhs.cast<Input>(),
        /*leading dim of RHS=*/rhs.num_rows, rhs.stride(),
        /*beta=*/beta, &output_data,
        /*leading dim of output=*/output_matrix.num_rows,
        output_matrix.stride(), batch_size);
  }
  return stream->ThenBlasGemm(
      lhs.transpose, rhs.transpose, output_matrix.num_rows,
      output_matrix.num_cols, /*size of reduce dim=*/lhs.reduced_dim(),
      /*alpha=*/alpha, lhs.cast<Input>(),
      /*leading dim of LHS=*/lhs.num_rows, rhs.cast<Input>(),
      /*leading dim of RHS=*/rhs.num_rows,
      /*beta=*/beta, &output_data,
      /*leading dim of output=*/output_matrix.num_rows);
}

Status RunGemm(const GpuGemmConfig &gemm_config,
               se::DeviceMemoryBase lhs_buffer, se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream *stream,
               bool implements_whole_instruction,
               absl::optional<int64> profile_index,
               HloExecutionProfiler *profiler,
               se::blas::ProfileResult *profile_result,
               absl::optional<se::blas::AlgorithmType> algorithm) {
  VLOG(2) << "Executing a GemmThunk";

  const Shape &output_shape = gemm_config.output_shape;
  const Shape &lhs_shape = gemm_config.lhs_shape;
  const Shape &rhs_shape = gemm_config.rhs_shape;
  const GemmBackendConfig &backend_config = gemm_config.backend_config;

  const DotDimensionNumbers &dim_nums = backend_config.dot_dimension_numbers();
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size(),
           dim_nums.rhs_batch_dimensions_size());
  CHECK_EQ(dim_nums.lhs_batch_dimensions_size() + 2, output_shape.rank());

  int64_t row_dim = dim_nums.lhs_batch_dimensions_size();
  int64_t col_dim = dim_nums.lhs_batch_dimensions_size() + 1;

  int64_t batch_size = backend_config.batch_size();

  // Check that the batch dims don't cover the last two dims.
  for (int64_t batch_dim : dim_nums.lhs_batch_dimensions()) {
    CHECK_NE(row_dim, batch_dim);
    CHECK_NE(col_dim, batch_dim);
  }

  // Verify that the non-batch dimensions are minor-most. This is required for
  // efficient access.
  for (const auto *shape : {&lhs_shape, &rhs_shape, &output_shape}) {
    CHECK_LT(shape->layout().minor_to_major(row_dim), 2);
    CHECK_LT(shape->layout().minor_to_major(col_dim), 2);
  }

  int64_t output_num_rows = output_shape.dimensions(row_dim);
  int64_t output_num_cols = output_shape.dimensions(col_dim);

  // BLAS gemm expects the inputs and the output are in column-major order.
  // Therefore, we need to convert dot between row-major matrices to that
  // between column-major matrices. The key insight for the conversion is that,
  // in linear storage, matrix M in column-major order is identical to the
  // transpose of M in row-major order. In other words,
  //
  //   column-major(M) = row-major(M^T).
  //
  // Leveraging this insight, we can perform dot between row-major matrices as
  // follows.
  //
  // row-major(C)
  //   = row-major(A x B) = column-major((A x B)^T) = column-major(B^T x A^T)
  //   = gemm(column-major(B^T), column-major(A^T))
  //   = gemm(row-major(B), row-major(A))
  //
  // Although we do not modify the content of A and B in linear memory, we
  // should use the dimensions of B^T and A^T when calling gemm. For example,
  // the leading dimension of the LHS matrix of gemm is the number of rows in
  // B^T and thus the number of columns in B.
  auto make_descriptor = [&](se::DeviceMemoryBase data, const Shape &shape,
                             bool transpose) -> MatrixDescriptor {
    bool is_row_major = LayoutUtil::Minor(shape.layout(), row_dim) != 0;
    bool layout_mismatch = LayoutUtil::Minor(shape.layout(), row_dim) !=
                           LayoutUtil::Minor(output_shape.layout(), row_dim);
    return MatrixDescriptor{
        data,
        transpose ^ layout_mismatch ? se::blas::Transpose::kTranspose
                                    : se::blas::Transpose::kNoTranspose,
        shape.dimensions(row_dim + static_cast<int64>(is_row_major)),
        shape.dimensions(row_dim + static_cast<int64>(!is_row_major))};
  };

  MatrixDescriptor lhs_matrix = make_descriptor(
      lhs_buffer, lhs_shape, dim_nums.lhs_contracting_dimensions(0) == row_dim);
  MatrixDescriptor rhs_matrix = make_descriptor(
      rhs_buffer, rhs_shape, dim_nums.rhs_contracting_dimensions(0) == col_dim);
  std::unique_ptr<ScopedInstructionProfiler> op_profiler =
      profiler ? profiler->MakeScopedInstructionProfiler(
                     implements_whole_instruction ? profile_index : -1)
               : nullptr;

  if (LayoutUtil::Minor(output_shape.layout(), row_dim) != 0) {
    std::swap(lhs_matrix, rhs_matrix);
    std::swap(output_num_cols, output_num_rows);
  }

  const MatrixDescriptor output_matrix{output_buffer,
                                       se::blas::Transpose::kNoTranspose,
                                       output_num_rows, output_num_cols};
  auto best_algorithm = [&]() -> absl::optional<se::blas::AlgorithmType> {
    if (algorithm) {
      return *algorithm;
    }
    if (backend_config.algorithm_case() ==
        GemmBackendConfig::ALGORITHM_NOT_SET) {
      return absl::nullopt;
    }
    return backend_config.selected_algorithm();
  }();

  complex128 alpha = {backend_config.alpha_real(), backend_config.alpha_imag()};
  double beta = backend_config.beta();

  switch (output_shape.element_type()) {
    case S32: {
      if (!best_algorithm) {
        return InternalError("Only extended GEMM is supported for int32");
      }
      CHECK_EQ(alpha.imag(), 0);
      if (lhs_shape.element_type() == PrimitiveType::S8 &&
          rhs_shape.element_type() == lhs_shape.element_type()) {
        return DoGemmWithAlgorithm<int8, int32>(
            batch_size, lhs_matrix, rhs_matrix, output_matrix,
            static_cast<int32>(alpha.real()), static_cast<int32>(beta), stream,
            *best_algorithm,
            /*output_profile_result=*/profile_result);
      }
      return InternalError(
          "For int32 gemm output only int8 input is supported, got input: %s",
          primitive_util::LowercasePrimitiveTypeName(lhs_shape.element_type()));
    }
    case F16:
      CHECK_EQ(alpha.imag(), 0);
      return DoGemm<Eigen::half>(
          batch_size, lhs_matrix, rhs_matrix, output_matrix,
          static_cast<Eigen::half>(alpha.real()),
          static_cast<Eigen::half>(beta), stream, best_algorithm,
          /*output_profile_result=*/profile_result);
    case BF16:
      CHECK_EQ(alpha.imag(), 0);
      return DoGemm<Eigen::bfloat16>(
          batch_size, lhs_matrix, rhs_matrix, output_matrix,
          static_cast<Eigen::bfloat16>(alpha.real()),
          static_cast<Eigen::bfloat16>(beta), stream, best_algorithm,
          /*output_profile_result=*/profile_result);
    case F32:
      CHECK_EQ(alpha.imag(), 0);
      return DoGemm<float>(batch_size, lhs_matrix, rhs_matrix, output_matrix,
                           alpha.real(), beta, stream, best_algorithm,
                           /*output_profile_result=*/profile_result);
    case F64:
      CHECK_EQ(alpha.imag(), 0);
      return DoGemm<double>(batch_size, lhs_matrix, rhs_matrix, output_matrix,
                            alpha.real(), beta, stream, best_algorithm,
                            /*output_profile_result=*/profile_result);
    case C64:
      return DoGemm<complex64>(batch_size, lhs_matrix, rhs_matrix,
                               output_matrix, static_cast<complex64>(alpha),
                               static_cast<complex64>(beta), stream,
                               best_algorithm,
                               /*output_profile_result=*/profile_result);
    case C128:
      return DoGemm<complex128>(
          batch_size, lhs_matrix, rhs_matrix, output_matrix, alpha,
          static_cast<complex128>(beta), stream, best_algorithm,
          /*output_profile_result=*/profile_result);
    default:
      return InternalError("Unexpected GEMM datatype: %s",
                           output_shape.ToString());
  }
}

}  // namespace gpu
}  // namespace xla
