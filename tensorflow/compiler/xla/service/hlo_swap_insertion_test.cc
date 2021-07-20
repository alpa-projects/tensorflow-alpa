#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/resource_loader.h"

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
    TF_EXPECT_OK(verifier().Run(module).status());
    return result;
  }

  /*
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
          ShapeUtil::MakeShape(xla::U8, {static_cast<int64>(arg.size())});
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

  std::pair<const HloInstruction*, const HloInstruction*> HasOneSwapOutAndDone(
      const HloInstruction* x) {
    EXPECT_GE(x->user_count(), 2);
    const HloInstruction* swap_out;
    const HloInstruction* swap_done;
    swap_out = swap_done = nullptr;
    for (auto inst : x->users()) {
      if (inst->IsCustomCall("__builtin$SwapOut")) {
        EXPECT_EQ(swap_out, nullptr);
        swap_out = inst;
        continue;
      }
      if (inst->IsCustomCall("__builtin$SwapDone")) {
        EXPECT_EQ(swap_done, nullptr);
        swap_done = inst;
        continue;
      }
    }
    EXPECT_NE(swap_out, nullptr);
    EXPECT_NE(swap_done, nullptr);
    EXPECT_TRUE(
        absl::c_linear_search(swap_out->control_successors(), swap_done));
    return std::make_pair(swap_out, swap_done);
  }

  std::vector<const HloInstruction*> HasSwapIn(const HloInstruction* x) {
    std::vector<const HloInstruction*> results;
    for (auto inst : x->control_successors()) {
      if (inst->IsCustomCall("__builtin$SwapIn")) {
        results.push_back(inst);
      }
    }
    EXPECT_GE(results.size(), 1);
    return results;
  }

  const HloInstruction* HasOneSwapDone(const HloInstruction* x) {
    const HloInstruction* swap_done = nullptr;
    for (auto inst : x->users()) {
      if (inst->IsCustomCall("__builtin$SwapDone")) {
        EXPECT_EQ(swap_done, nullptr);
        swap_done = inst;
      }
    }
    EXPECT_NE(swap_done, nullptr);
    return swap_done;
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
  EXPECT_EQ(a->user_count(), 3);
  const HloInstruction* out_a;
  const HloInstruction* done_out_a;
  {
    auto result = HasOneSwapOutAndDone(a);
    out_a = result.first;
    done_out_a = result.second;
  }

  EXPECT_EQ(out_a->control_successors().size(), 2); // in_a, out_done_a
  EXPECT_TRUE(absl::c_linear_search(out_a->control_successors(),
  done_out_a)); auto all_in_a = HasSwapIn(out_a); EXPECT_EQ(all_in_a.size(),
  1); const HloInstruction* in_a = all_in_a.at(0); const HloInstruction*
  in_a_done = HasOneSwapDone(in_a);
  // check there is only one operand
  EXPECT_EQ(out_a->operand_count(), 1);
  EXPECT_EQ(done_out_a->operand_count(), 1);
  EXPECT_EQ(in_a->operand_count(), 0);
  // check the return value
  EXPECT_EQ(in_a->shape(), a->shape());
  EXPECT_TRUE(absl::c_linear_search(in_a_done->control_successors(), f));

  // check swap inserted for c
  EXPECT_EQ(c->user_count(), 2);
  const HloInstruction* out_c;
  const HloInstruction* done_out_c;
  {
    auto result = HasOneSwapOutAndDone(c);
    out_c = result.first;
    done_out_c = result.second;
  }
  EXPECT_EQ(out_c->control_successors().size(), 2); // out_done_c, in_c
  EXPECT_TRUE(absl::c_linear_search(out_c->control_successors(),
  done_out_c)); auto all_in_c = HasSwapIn(out_c); EXPECT_EQ(all_in_c.size(),
  1); const HloInstruction* in_c = all_in_c.at(0); const HloInstruction*
  in_c_done = HasOneSwapDone(in_c);
  // check there is only one operand
  EXPECT_EQ(out_c->operand_count(), 1);
  EXPECT_EQ(done_out_c->operand_count(), 1);
  EXPECT_EQ(in_c->operand_count(), 0);
  // check the return value
  EXPECT_EQ(in_c->shape(), c->shape());
  EXPECT_TRUE(absl::c_linear_search(in_c_done->control_successors(), e));

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
      gExec, {ToF32Span(&a_value), ToF32Span(&b_value),
      ToF32Span(&c_value)});
}

TEST_F(HloSwapInsertionTest, ReshardingTest) {
  const char* const hlo_string = R"(
HloModule ShardingComputation
ENTRY %SingleComputation (init: f32[1]) -> f32[8,8] {
  %init = f32[1]{0} parameter(0)
  %reshape = f32[] reshape(f32[1]{0} %init)
  %broadcast = f32[8,8]{1,0} broadcast(f32[] %reshape), dimensions={}, sharding={devices=[2, 2]0,1,2,3}
  %broadcast.2 = f32[8,8]{1,0} broadcast(f32[] %reshape), dimensions={}, sharding={devices=[2, 2]0,1,2,3}
  %broadcast.1 = f32[8,8]{1,0} broadcast(f32[] %reshape), dimensions={}, sharding={devices=[2, 2]0,1,2,3}
  %add = f32[8,8]{1,0} add(f32[8,8]{1,0} %broadcast, f32[8,8]{1,0} %broadcast.1)
  %add.1 = f32[8,8]{1,0} add(f32[8,8]{1,0} %add, f32[8,8]{1,0} %broadcast.2), sharding={devices=[4,1]0,1,2,3}
  ROOT %add.2 = f32[8,8]{1,0} add(f32[8,8]{1,0} %add.1, f32[8,8]{1,0} %broadcast), sharding={devices=[4,1]0,1,2,3}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  {
    HloPassPipeline pass("gpu-partitioning");
    pass.AddPass<ShardingPropagation>(/*is_spmd=*/true);
    pass.AddPass<gpu::GpuSpmdPartitioner>(4, /*num_replicas=*/1);
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    TF_ASSERT_OK(pass.Run(module.get()).status());
  }

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloSwapInsertion(580, module.get()));

  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                          CompileModule(std::move(module)));
}

};  // namespace

};  // namespace xla
