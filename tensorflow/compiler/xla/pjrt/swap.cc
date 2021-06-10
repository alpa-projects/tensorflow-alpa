#include "tensorflow/compiler/xla/pjrt/swap.h"

#ifdef GOOGLE_CUDA
namespace xla{

HostMemoryTable& local_host_memory_table(){
  static HostMemoryTable local_host_memory_table_;
  return local_host_memory_table_;
}

HostMemoryTable::HostMemoryTable() {}

HostMemoryTable::AddressList* HostMemoryTable::GetOrCreate(int64 executable_key, int64 key) {
  CHECK(executable_key != -1) << "The executable is unregistered"; 
  auto iter = lists_.find(std::make_pair(executable_key, key));
  if (iter != lists_.end()) {
    return iter->second.get();
  }
  std::unique_ptr<AddressList> new_list = 
    std::make_unique<AddressList>(std::move(AddressList()));
  AddressList* ptr = new_list.get();
  lists_.emplace(
    std::make_pair(executable_key, key), std::move(new_list));
  return ptr;
}

const HostMemoryTable::AddressList* HostMemoryTable::Get(int64 executable_key, int64 key) {
  CHECK(executable_key != -1) << "The executable is unregistered"; 
  auto iter = lists_.find(std::make_pair(executable_key, key));
  CHECK(iter != lists_.end()) << "Swap In try to get a tensor not swapped out";
  return iter->second.get();
}

const HostMemoryTable::AddressList* HostMemoryTable::GetOrNull(int64 executable_key, int64 key) {
  auto iter = lists_.find(std::make_pair(executable_key, key));
  if(iter == lists_.end()) return nullptr;
  return iter->second.get();
}

void HostMemoryTable::remove(int64 executable_key, int64 key) {
  lists_.erase(std::make_pair(executable_key, key));
}

int64 HostMemoryTable::getOrCreateExecutableKey(const Executable &e) {
  // TODO
  return 0;
}

HostMemoryTable::~HostMemoryTable() {}  
// deallocate memories in the destructor of SwapOutThunk instead of here

};  // namespace xla
#endif