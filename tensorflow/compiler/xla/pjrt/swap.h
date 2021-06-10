#ifndef TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "absl/container/flat_hash_map.h"

namespace xla{
// TODO(yonghao): A shape acquires its host copy by a key and a int64 RunId

class HostMemoryTable {
 private: 
 public: 
  using AddressList = std::vector<void *>;
  explicit HostMemoryTable();
  ~HostMemoryTable();

  AddressList* GetOrCreate(int64 executable_key, int64 key);

  const AddressList* Get(int64 executable_key, int64 key);

  const AddressList* GetOrNull(int64 executable_key, int64 key);

  void remove(int64 executable_key, int64 key);

  int64 getOrCreateExecutableKey(const Executable &e);

  // given the logical executable
 private: 

  absl::flat_hash_map<std::pair<int64, int64>, 
                      std::unique_ptr<AddressList>> lists_;
};

HostMemoryTable& local_host_memory_table();
} // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_PJRT_SWAP_H_