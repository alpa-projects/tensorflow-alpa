#include "tensorflow/compiler/xla/pjrt/swap.h"

#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/platform/random.h"

namespace xla{
namespace{

TEST(GpuSwap, Thunks) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtClient> client,
      GetGpuClient(/*asynchronous=*/true, GpuAllocatorConfig(),
                   /*distributed_client=*/nullptr, /*node_id=*/0));

  PjRtDevice* device = client->addressable_devices().at(0);

  int n = 1024;
  Shape shape = ShapeUtil::MakeShape(S32, {n});
  Shape keyShape = ShapeUtil::MakeShape(S64, {});

  XlaBuilder builder("acomputation");
  auto p0 = Parameter(&builder, 0, shape, "param");
  int64 key = 10;
  auto swap_out = CustomCall(&builder, "__builtin$SwapOut", {p0}, keyShape, /*opaque=*/std::to_string(key));
  auto swap_in = CustomCall(&builder, "__builtin$SwapIn", {swap_out}, shape, std::to_string(key));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_disable_multi_streaming(true);
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = device->id();
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtExecutable> executable,
      client->Compile(computation, std::move(compile_options)));
  // start execution
    // prepare
  std::vector<int32> inputs(n);
  std::vector<int32> expected_outputs(n);
  for (int i = 0; i < n; ++i) {
    inputs[i] = tensorflow::random::New64();
    expected_outputs[i] = inputs[i];
  }
  TF_ASSERT_OK_AND_ASSIGN(
    auto in_buffer0,
    client->BufferFromHostBuffer(
        inputs.data(), shape,
        PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
        /*on_done_with_host_buffer=*/nullptr, device));
  
  ExecuteOptions options;
  TF_ASSERT_OK_AND_ASSIGN(
        auto out_buffers,
        executable->Execute({{in_buffer0.get()}}, options));
  TF_ASSERT_OK_AND_ASSIGN(auto out_literal, out_buffers[0][0]->ToLiteral());
  LiteralTestUtil::ExpectR1Equal<int32>(expected_outputs, *out_literal);  
    // check successfully swapped back
}

}
} // namespace xla