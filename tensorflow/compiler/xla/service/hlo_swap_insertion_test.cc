#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::_;

class HloSwapInsertionTest : public HloTestBase {
 protected:
  StatusOr<bool> RunHloSwapInsertion(int64 memory_limit_bytes,
                                     HloModule* module) {
    TF_EXPECT_OK(verifier().Run(module).status());
    HloMemoryScheduler scheduler(
        [](const BufferValue& buffer) { return ByteSizeOf(buffer.shape()); },
        ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
    TF_EXPECT_OK(scheduler.Run(module).status());
    HloSwapInsertion swap(ByteSizeOf, memory_limit_bytes);
    auto result = swap.Run(module);
    TF_EXPECT_OK(scheduler.Run(module).status());
    return result;
  }

  /*
  This computation is more close to a good test environment, as MLP has too many
  parameters and too few variables without swap:
  // a, b, c alive
  d = a + b // a, b, c, d alive
  e = c + d // a, c, d, e alive
  f = a + e // a, e, f alive
  Then test what if only three can live simultaneously
  */
  std::unique_ptr<HloComputation> MakeSimpleComputation(
      const string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);

    auto input = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "init"));
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, input));
    auto a = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec10_shape_, reshape, {}));
    auto b = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec10_shape_, reshape, {}));
    auto c = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec10_shape_, reshape, {}));

    auto d = builder.AddInstruction(
        HloInstruction::CreateBinary(vec10_shape_, HloOpcode::kAdd, a, b));
    auto e = builder.AddInstruction(
        HloInstruction::CreateBinary(vec10_shape_, HloOpcode::kAdd, d, c));
    builder.AddInstruction(
        HloInstruction::CreateBinary(vec10_shape_, HloOpcode::kAdd, e, a));
    return builder.Build();
  }
  /*
  creates and returns a forward MLP computation as:
  F32[20, 10] w_1 = {...}
  F32[20] b_1 = {...}
  F32[10, 20] w_2 = {...}
  // F32[10] b_2 = {...}

  F32[1] init = {...}
  F32[] reshape = reshape()

  F32[20] matmul_1 = dot(w_1, x)
  F32[20] lp_1 = binary(add, b_1, matmul_1)
  F32[10] matmul_2 = dot(w_2, lp_1)
  // ignore lp_2
  F32[10] resnet = binary(add, matmul_2, x)

  bias in the second layer(b_2) is ignored to simplify the test

  In this test, if the space is sizeof(F32) * (421(parameter)+20), x should be
  offloaded to make space for `lp_1 = binary(add, b_1, matmul_1)'

  TODO: in place operations
  */
  std::unique_ptr<HloComputation> MakeMLPComputation(
      const string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param_1 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, mat20_10_shape_, "w1"));
    auto param_2 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, mat10_20_shape_, "w2"));
    auto param_3 = builder.AddInstruction(
        HloInstruction::CreateParameter(2, vec20_shape_, "b1"));
    // auto param_4 = builder.AddInstruction(
    //   HloInstruction::CreateParameter(0, vec10_shape_, "b2")
    // );

    auto input = builder.AddInstruction(
        HloInstruction::CreateParameter(3, vec1_shape_, "init"));
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, input));
    auto x = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec10_shape_, reshape, {}));

    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    auto matmul_1 = builder.AddInstruction(HloInstruction::CreateDot(
        vec20_shape_, param_1, x, dot_dnums, DefaultPrecisionConfig(2)));
    auto lp_1 = builder.AddInstruction(HloInstruction::CreateBinary(
        vec20_shape_, HloOpcode::kAdd, matmul_1, param_3));
    auto matmul_2 = builder.AddInstruction(HloInstruction::CreateDot(
        vec10_shape_, param_2, lp_1, dot_dnums, DefaultPrecisionConfig(2)));
    // auto lp_2 = builder.AddInstruction(
    //   HloInstruction::CreateBinary(vec10_shape_, HloOpCode::kAdd, matmul_2,
    //   param_4)
    // );
    builder.AddInstruction(HloInstruction::CreateBinary(
        vec10_shape_, HloOpcode::kAdd, matmul_2, x));
    return builder.Build();
  }

  // Return the byte size of the top-level buffer of the given shape.
  static int64 ByteSizeOf(const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

 protected:
  const Shape scalar_shape_ = ShapeUtil::MakeShape(xla::F32, {});
  const Shape vec1_shape_ = ShapeUtil::MakeShape(xla::F32, {1});
  const Shape vec10_shape_ = ShapeUtil::MakeShape(xla::F32, {10});
  const Shape vec20_shape_ = ShapeUtil::MakeShape(xla::F32, {20});
  const Shape mat20_10_shape_ = ShapeUtil::MakeShape(xla::F32, {20, 10});
  const Shape mat10_20_shape_ = ShapeUtil::MakeShape(xla::F32, {10, 20});
};

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation.
TEST_F(HloSwapInsertionTest, SingleComputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeSimpleComputation());
  // memory constraint: (1+30) * 4
  RunHloSwapInsertion(35 * 4, module.get());
  // TODO(yonghao): add a checker
}

};  // namespace

};  // namespace xla
