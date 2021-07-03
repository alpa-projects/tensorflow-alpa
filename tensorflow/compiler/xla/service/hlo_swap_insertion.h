#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SWAP_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SWAP_INSERTION_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
class HloSwapInsertion : public HloModulePass {
 public:
  using ShapeSizeFunction = std::function<int64(const Shape&)>;
  explicit HloSwapInsertion(const ShapeSizeFunction& size_function,
                            int64 memory_limit_bytes)
      : size_function_(size_function),
        memory_limit_bytes_(memory_limit_bytes) {}

  ~HloSwapInsertion() override = default;

  absl::string_view name() const override { return "swap-Insertion"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  virtual StatusOr<bool> SwapInsertionComputation(HloComputation* computation,
                                                  HloSchedule* schedule,
                                                  int64 memory_limit_bytes);

  // Computes and returns the peak memory used by the given computation. The
  // peak memory is the maximum total size of all live HLO instruction values at
  // any program point. 'order' is the order in which the HLO instructions will
  // be emitted which is used to determine lifespans of HLO values.
  StatusOr<int64> ComputePeakMemory(HloComputation* computation,
                                    const HloInstructionSequence& order) const;

  // Returns the peak memory usage of the called computations for the given
  // instruction. Zero is returned if the instruction calls no computations.
  StatusOr<int64> CalledComputationsMemoryUsage(
      const HloInstruction* instruction) const;

  // Call graph of the hlo_module.
  std::unique_ptr<CallGraph> call_graph_;

  const ShapeSizeFunction size_function_;

  int64 memory_limit_bytes_;

  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;

  absl::flat_hash_map<const HloComputation*, int64> computation_peak_memory_;
};

};  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SWAP_INSERTION_H_