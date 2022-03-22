namespace xla {

namespace spmd {

// Compile an Xla Computation into HloModule, then apply Alpa's passes.
// The result hlo is later compiled again to apply spmd and other optimizations.
StatusOr<std::shared_ptr<xla::HloModule>> Alpa_Compile(const XlaComputation& computation, CompileOptions options);

};

};