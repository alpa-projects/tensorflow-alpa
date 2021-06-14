#ifndef TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HostMemoryTable {
 private:
 public:
  using AddressList = std::vector<void*>;
  explicit HostMemoryTable();
  ~HostMemoryTable();

  AddressList* GetOrCreate(int64 executable_key, int64 key);

  const AddressList* Get(int64 executable_key, int64 key);

  const AddressList* GetOrNull(int64 executable_key, int64 key);

  void remove(int64 executable_key, int64 key);

  // given the logical executable
 private:
  absl::flat_hash_map<std::pair<int64, int64>, std::unique_ptr<AddressList>>
      lists_;
};

HostMemoryTable& local_host_memory_table();
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_