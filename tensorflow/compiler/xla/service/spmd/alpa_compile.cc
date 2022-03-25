#include "tensorflow/compiler/xla/service/spmd/alpa_compile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_util.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"

namespace xla {

namespace spmd {
// TODO(yonghao): Check correctness of compile options and modules
void PreCompileCheck(const XlaComputation& computation,
                     CompileOptions options) {
}

StatusOr<std::shared_ptr<xla::HloModule>> AlpaCompile(
    const XlaComputation& computation, CompileOptions options) {
  PreCompileCheck(computation, options);

  const HloModuleProto& module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, DebugOptions()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      CreateModuleFromProto(
                          module_proto, module_config,
                          options.executable_build_options.run_backend_only()));
}
};  // namespace spmd

};  // namespace xla