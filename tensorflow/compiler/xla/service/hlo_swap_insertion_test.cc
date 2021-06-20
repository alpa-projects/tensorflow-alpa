#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
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

class HloSwapInsertionTest : public gpu::GpuCodegenTest {
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

  template <typename T>
  static absl::Span<float> ToF32Span(std::vector<T>* v) {
    return absl::Span<float>(reinterpret_cast<float*>(v->data()),
                             v->size() * sizeof(T));
  }

 protected:
  StatusOr<std::unique_ptr<Executable>> CompileModule(
      std::unique_ptr<HloModule> module) {
    return backend().compiler()->RunBackend(
        std::move(module), backend().default_stream_executor(),
        Compiler::CompileOptions{
            backend().default_stream_executor()->GetAllocator()});
  }

  StatusOr<ExecutionOutput> RunModule(
      gpu::GpuExecutable* executable, se::Stream* stream,
      absl::Span<const se::DeviceMemoryBase> arguments) {
    ExecutableRunOptions executable_run_options;
    auto stream_d2h = std::make_unique<se::Stream>(stream->parent());
    auto stream_h2d = std::make_unique<se::Stream>(stream->parent());
    stream_d2h->Init();
    stream_h2d->Init();

    DeviceAssignment device_assignment(1, 1);
    device_assignment(0, 0) = backend().default_device_ordinal();
    executable_run_options.set_stream(stream);
    executable_run_options.set_allocator(backend().memory_allocator());
    executable_run_options.set_host_to_device_stream(stream_d2h.get());
    executable_run_options.set_device_to_host_stream(stream_h2d.get());
    executable_run_options.set_device_assignment(&device_assignment);
    ServiceExecutableRunOptions run_options(executable_run_options);
    std::vector<ExecutionInput> execution_inputs;

    for (auto arg : arguments) {
      Shape shape =
          ShapeUtil::MakeShape(xla::F32, {static_cast<int64>(arg.size())});
      execution_inputs.emplace_back(shape);
      execution_inputs.back().SetBuffer({}, MaybeOwningDeviceMemory(arg));
    }

    TF_ASSIGN_OR_RETURN(auto output,
                        executable->ExecuteAsyncOnStream(
                            &run_options, std::move(execution_inputs),
                            /*hlo_execution_profile=*/nullptr));

    TF_CHECK_OK(stream->BlockHostUntilDone());

    return std::move(output);
  }

  StatusOr<std::vector<std::vector<float>>> RunModuleWithHostBuffers(
      gpu::GpuExecutable* executable,
      std::vector<absl::Span<float>> arguments) {
    auto* allocator = backend().memory_allocator();
    std::vector<se::OwningDeviceMemory> owning_memory;
    owning_memory.reserve(arguments.size());
    for (auto host_buffer : arguments) {
      owning_memory.push_back(
          allocator
              ->Allocate(backend().default_device_ordinal(), host_buffer.size())
              .ConsumeValueOrDie());
    }
    auto stream = backend()
                      .BorrowStream(backend().default_device_ordinal())
                      .ConsumeValueOrDie();
    std::vector<se::DeviceMemoryBase> args;
    for (int i = 0; i < owning_memory.size(); i++) {
      se::DeviceMemoryBase memory(*owning_memory[i]);
      stream->ThenMemcpy(&memory, static_cast<void*>(arguments[i].data()),
                         memory.size());
      args.push_back(memory);
    }

    TF_ASSIGN_OR_RETURN(ExecutionOutput output,
                        RunModule(executable, stream.get(), args));
    std::vector<std::vector<float>> host_outputs;
    for (const auto& result : output.Result().buffers().leaves()) {
      host_outputs.emplace_back();
      host_outputs.back().resize(result.second.size());
      stream->ThenMemcpy(static_cast<void*>(host_outputs.back().data()),
                         result.second, result.second.size());
    }
    TF_CHECK_OK(stream->BlockHostUntilDone());
    return host_outputs;
  }

  const Shape scalar_shape_ = ShapeUtil::MakeShape(xla::F32, {});
  const Shape vec1_shape_ = ShapeUtil::MakeShape(xla::F32, {1});
  const Shape vec10_shape_ = ShapeUtil::MakeShape(xla::F32, {10});
  const Shape vec20_shape_ = ShapeUtil::MakeShape(xla::F32, {20});
  const Shape mat20_10_shape_ = ShapeUtil::MakeShape(xla::F32, {20, 10});
  const Shape mat10_20_shape_ = ShapeUtil::MakeShape(xla::F32, {10, 20});
};

// Test swap insertion of a single computation
TEST_F(HloSwapInsertionTest, SingleComputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeSimpleComputation());

  // Find and save the original computation
  const HloInstruction* f = computation->root_instruction();
  ASSERT_THAT(f, op::Add(op::Add(op::Add(op::Broadcast(_), op::Broadcast(_)),
                                 op::Broadcast(_)),
                         op::Broadcast(op::Reshape(_))));
  const HloInstruction* e = f->operand(0);
  const HloInstruction* d = e->operand(0);
  const HloInstruction* c = e->operand(1);
  const HloInstruction* a = f->operand(1);
  // memory constraint: (1+30) * 4
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloSwapInsertion(35 * 4, module.get()));

  EXPECT_TRUE(changed);

  // check swap inserted for a
  EXPECT_EQ(a->user_count(), 2);
  HloInstruction* out_a = a->users().at(1);
  EXPECT_TRUE(out_a->IsCustomCall("__builtin$SwapOut"));

  EXPECT_EQ(out_a->user_count(), 1);
  HloInstruction* in_a = out_a->users().at(0);
  EXPECT_TRUE(in_a->IsCustomCall("__builtin$SwapIn"));
  // check there is only one operand
  EXPECT_EQ(out_a->operand_count(), 1);
  EXPECT_EQ(in_a->operand_count(), 1);
  // check swap inserted for c
  EXPECT_EQ(c->user_count(), 1);
  HloInstruction* out_c = c->users().at(0);
  EXPECT_TRUE(out_c->IsCustomCall("__builtin$SwapOut"));

  EXPECT_EQ(out_c->user_count(), 1);
  HloInstruction* in_c = out_c->users().at(0);
  EXPECT_TRUE(in_c->IsCustomCall("__builtin$SwapIn"));
  // check there is only one operand
  EXPECT_EQ(out_c->operand_count(), 1);
  EXPECT_EQ(in_c->operand_count(), 1);
  // todo(yonghao): check control dependency
  // todo(yonghao): check insertion of SwapDone

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          CompileModule(std::move(module)));
  gpu::GpuExecutable* gExec =
      dynamic_cast<gpu::GpuExecutable*>(executable.get());

  std::vector<float> a_value(10), b_value(10), c_value(10);
  for (int i = 0; i < 10; ++i) {
    a_value.push_back(i);
    b_value.push_back(i);
    c_value.push_back(i);
  }
  RunModuleWithHostBuffers(
      gExec, {ToF32Span(&a_value), ToF32Span(&b_value), ToF32Span(&c_value)});
}

};  // namespace

};  // namespace xla
