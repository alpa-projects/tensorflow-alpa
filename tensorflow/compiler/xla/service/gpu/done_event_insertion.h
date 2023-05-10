#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DONE_EVENT_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DONE_EVENT_INSERTION_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
class HloDoneInsertion : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;
  HloDoneInsertion() = default;

  ~HloDoneInsertion() override = default;

  absl::string_view name() const override { return "done-Insertion"; }

  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

};  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DONE_EVENT_INSERTION_H_
