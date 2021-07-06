#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

// todo(yonghao): store the space for parameter specifically
namespace xla {

namespace {

const char* const kBuiltinSwapOutTarget = "__builtin$SwapOut";
const char* const kBuiltinSwapInTarget = "__builtin$SwapIn";
const char* const kBuiltinSwapDoneTarget = "__builtin$SwapDone";
int64 SwapKey = 0;
int64 SwapDoneEventKey = 0;
const Shape FormalShape = ShapeUtil::MakeNil();

using ::tensorflow::strings::HumanReadableNumBytes;

using BufferId = int64;
const BufferId kInvalidBufferId = -1;
const int64 InfiniteMemory = -1;
const int64 kInvalidPosition = INT64_MAX;

struct Operand {
  BufferId bid;
  HloInstruction* instruction;

  bool operator==(const Operand& other) const {
    return bid == other.bid && instruction == other.instruction;
  }
};

using BufferIdList = absl::InlinedVector<BufferId, 3>;
using OperandList = absl::InlinedVector<Operand, 3>;
using UsesList = absl::InlinedVector<int64, 3>;

class Item {
 public:
  HloInstruction* instruction;
  int64 position = kInvalidPosition;
  OperandList buffers_used;
  BufferIdList buffers_defined;
  bool is_swap = false;

  void ShouldBefore(Item* o) {
    should_before = std::min(should_before, o->position);
  }
  void ShouldAfter(Item* o) {
    should_after = std::max(should_after, o->position);
  }

  std::string ToString() const {
    if (is_swap) return instruction->ToString();
    return instruction->ToShortString();
  }

 private:
  friend class InstructionList;
  int64 should_before = kInvalidPosition, should_after = -1;
  Item* next;
};

bool IsSwapIn(const Item* item) {
  return item->instruction->IsCustomCall(kBuiltinSwapInTarget);
}
bool IsSwapOut(const Item* item) {
  return item->instruction->IsCustomCall(kBuiltinSwapOutTarget);
}
bool IsSwapDone(const Item* item) {
  return item->instruction->IsCustomCall(kBuiltinSwapDoneTarget);
}
std::string OpaqueByIndex(const std::string& full, int index) {
  std::vector<std::string> result =
      absl::StrSplit(full, absl::MaxSplits(';', index + 1));
  return result.at(index);
}

class InstructionList {
 public:
  explicit InstructionList(const HloInstructionSequence& order) {
    int64 position = 0;
    Item* last = nullptr;
    for (HloInstruction* inst : order.instructions()) {
      Item* item = new Item;
      item->next = nullptr;
      if (last == nullptr) {
        first_ = item;
      } else {
        last->next = item;
      }
      last = item;

      item->instruction = inst;
      item->position = position;
      position++;

      item_map_[inst] = item;
    }
  }

  ~InstructionList() {
    for (Item* item = first_; item != nullptr;) {
      Item* next = item->next;
      delete item;
      item = next;
    }
    for (int i = 0; i < swap_ins_.size(); ++i) {
      delete swap_ins_.at(i);
    }
    for (int i = 0; i < swap_outs_.size(); ++i) {
      delete swap_outs_.at(i);
    }
    for (int i = 0; i < swap_dones_.size(); ++i) {
      delete swap_dones_.at(i);
    }
  }

  Item* first() const { return first_; }

  Item* next(Item* it) const { return it->next; }

  Item* GetItem(const HloInstruction* inst) const {
    auto iter = item_map_.find(inst);
    CHECK(iter != item_map_.end()) << "Did not find " << inst->name();
    return iter->second;
  }

  void SwapToSwapEdge(HloInstruction* from, HloInstruction* to) {
    if (from->IsCustomCall(kBuiltinSwapOutTarget) && to->IsCustomCall(kBuiltinSwapInTarget)) {
      auto* from_inst = Cast<HloCustomCallInstruction>(from);
      auto* to_inst = Cast<HloCustomCallInstruction>(to);
      if (OpaqueByIndex(from_inst->opaque(), 0) !=
          OpaqueByIndex(to_inst->opaque(), 0)) {
        // The to instruction should wait until the from inst ends.
        std::string new_opaque = to_inst->opaque();
        new_opaque.append(";").append(OpaqueByIndex(from_inst->opaque(), 1));
        to_inst->set_raw_backend_config_string(to_inst->opaque());
      }
    }
  }

  void AddEdge(Item* from, Item* to) {
    if (from->is_swap && !to->is_swap) {
      from->instruction->AddControlDependencyTo(to->instruction);
      from->ShouldBefore(to);  // swap done should be as late as possible
      return;
    }
    if (!from->is_swap && to->is_swap) {
      from->instruction->AddControlDependencyTo(to->instruction);
      to->ShouldAfter(from);  // swap out should be as early as possible
      return;
    }
    if (from->is_swap && to->is_swap) {
      if (IsSwapDone(from)) {
        // edge is from a swap done to a swap in/out
        CHECK(IsSwapIn(to) || IsSwapOut(to))
            << "add edge from swap done to swap done";
        auto* from_swap = from->buffers_used.at(0).instruction;
        SwapToSwapEdge(from_swap, to->instruction);
        from_swap->AddControlDependencyTo(to->instruction);
        return;
      }
      if (!IsSwapDone(to)) {
        SwapToSwapEdge(from->instruction, to->instruction);
      }
      from->instruction->AddControlDependencyTo(to->instruction);
    }
  }

  // Insert swaps to instruction sequence.
  // First, set the position of swap out: as early a.p.
  // If is after a swap in: swap out must be after a later last_use
  // Then, set the position of swap in: as early a.p.
  // If it is after a swap out: wait until done.
  // Finally, swap done is as late a.p.(exactly before the first use).
  HloInstructionSequence toSequence() {
    HloInstructionSequence seq;

    absl::c_sort(swap_outs_, [](const Item* x, const Item* y) {
      if (x->should_after != y->should_after) {
        return x->should_after < y->should_after;
      }
      return x->should_before < y->should_before;
    });
    // solve swap in's dependency on swap out
    int64 cnt = 0;
    absl::c_for_each(swap_outs_, [&cnt](Item*& item) {
      item->position = cnt++;  // relative position of swap outs
    });
    absl::c_for_each(swap_ins_, [&](Item*& item) {
      for (HloInstruction* pre : item->instruction->control_predecessors()) {
        if (pre->IsCustomCall(kBuiltinSwapOutTarget)) {
          Item* from = GetItem(pre);
          item->should_after = std::max(item->should_after, from->should_after);
          item->position = std::max(item->position, from->position);
          // get relative position of swap in based on swap outs:
          // avoid so.1, so.2, si.3, si.4, si.3 waits so.2 but si.4 waits so.1
        }
      }
    });

    absl::c_sort(swap_ins_, [](const Item* x, const Item* y) {
      if (x->should_after != y->should_after) {
        return x->should_after < y->should_after;
      }
      return x->position < y->position;
    });
    // sort swap done, at least before next use
    absl::c_sort(swap_dones_, [](const Item* x, const Item* y) {
      return x->should_before < y->should_before;
    });

    // set as a sequence
    auto done_iter = swap_dones_.begin();
    auto out_iter = swap_outs_.begin();
    auto in_iter = swap_ins_.begin();

    for (Item* item = first_; item != nullptr; item = item->next) {
      while (done_iter != swap_dones_.end()) {
        if ((*done_iter)->should_before == item->position) {
          seq.push_back((*done_iter)->instruction);
          ++done_iter;
        } else {
          CHECK((*done_iter)->should_before > item->position);
          break;
        }
      }
      seq.push_back(item->instruction);
      while (out_iter != swap_outs_.end()) {
        if ((*out_iter)->should_after == item->position) {
          seq.push_back((*out_iter)->instruction);
          ++out_iter;
        } else {
          CHECK((*out_iter)->should_after > item->position);
          break;
        }
      }
      while (in_iter != swap_ins_.end()) {
        if ((*in_iter)->should_after == item->position) {
          seq.push_back((*in_iter)->instruction);
          ++in_iter;
        } else {
          CHECK((*in_iter)->should_after > item->position);
          break;
        }
      }
    }
    return seq;
  }

  void AddSwapIn(Item* item) {
    swap_ins_.push_back(item);
    item_map_.insert({item->instruction, item});
  }
  void AddSwapOut(Item* item) {
    swap_outs_.push_back(item);
    item_map_.insert({item->instruction, item});
  }

  void AddSwapDone(Item* swap, Item* done) {
    swap_done_map_.insert({swap, done});
    swap_dones_.push_back(done);
    item_map_.insert({done->instruction, done});
  }

  Item* GetSwapDone(Item* swap) { return swap_done_map_.at(swap); }

 private:
  Item* first_;
  absl::flat_hash_map<const HloInstruction*, Item*> item_map_;
  absl::flat_hash_map<Item*, Item*> swap_done_map_;
  std::vector<Item*> swap_ins_, swap_outs_, swap_dones_;
};

UsesList GetUsers(const InstructionList& instruction_list,
                  const LogicalBuffer* logical_buffer, const int64 bid,
                  const TuplePointsToAnalysis& points_to_analysis) {
  UsesList users;
  for (const BufferAlias& buffer_alias :
       points_to_analysis.GetBufferAliases(*logical_buffer)) {
    for (const HloInstruction* user : buffer_alias.instruction()->users()) {
      if (points_to_analysis.DoesNotUseOperandBuffer(
              buffer_alias.instruction(), buffer_alias.index(), user)) {
        // The alias may be an operand of 'user', but the LogicalBuffer cannot
        // possibly be used by the instruction so ignore 'user'. This is the
        // case, for example, for the tuple element buffers in a GetTupleElement
        // instruction (the GTE instruction only uses the pointer vector).
        continue;
      }
      Item* user_item = instruction_list.GetItem(user);
      users.push_back(user_item->position);
      if (!absl::c_linear_search(user_item->buffers_used,
                                 Operand{bid, buffer_alias.instruction()})) {
        user_item->buffers_used.push_back(
            Operand{bid, buffer_alias.instruction()});
      }
    }
  }
  absl::c_sort(users);

  return users;
}

class MemoryRecorder {
 public:
  MemoryRecorder(HloComputation* computation, InstructionList& inst_list,
                 const TuplePointsToAnalysis& points_to_analysis,
                 int64 memory_bound,
                 const HloSwapInsertion::ShapeSizeFunction& size_function);

  void PrepareForInstruction(Item* item);

  void RecycleAfterInstruction(Item* item);

  int64 memory_usage() const { return memory_usage_; }

 private:
  struct Buffer {
    BufferId id;

    int64 size;

    bool live_out;

    Item* defining_instruction;

    Item* latest_alloc;

    int64 occupying_size;

    UsesList use_positions;

    int64 next_use_index;

    int64 next_use() const {
      if (next_use_index == use_positions.size()) {
        return kInvalidPosition;
      }
      CHECK(next_use_index < use_positions.size());
      return use_positions[next_use_index];
    }

    bool InGPU() const { return latest_alloc != nullptr; }

    bool IsSwappedIn() const { return latest_alloc != defining_instruction; }

    void SetInGPU(Item* alloc, int64 occupy) {
      if (alloc == nullptr) {
        CHECK(InGPU());
        CHECK(occupy == 0);
      } else {
        CHECK(!InGPU());
        CHECK(alloc == defining_instruction || IsSwapIn(alloc));
      }
      latest_alloc = alloc;
      occupying_size = occupy;
    }
  };

  struct AllocatedInterval {
    AllocatedInterval(const Buffer* buffer) : buffer_(buffer) {}

    const Buffer* buffer_;

    int64 next_use() const { return buffer_->next_use(); }

    const Buffer* buffer() const { return buffer_; }
  };

  class Heap {
   public:
    explicit Heap() = default;

    void push(AllocatedInterval interval) {
      intervals_.push_back(interval);
      std::push_heap(intervals_.begin(), intervals_.end(), compare);
    }

    void pop() {
      std::pop_heap(intervals_.begin(), intervals_.end(), compare);
      intervals_.pop_back();
    }

    const AllocatedInterval& top() { return intervals_.front(); }

    const std::vector<AllocatedInterval>& intervals() { return intervals_; }

    std::string IntervalsInfo() const {
      std::string info;
      for (const AllocatedInterval& interval : intervals_) {
        absl::StrAppend(&info, interval.buffer()->id, ", ", interval.next_use(),
                        "; ");
      }
      return info;
    }

    int64 size() { return intervals_.size(); }

    bool empty() { return intervals_.empty(); }

    void adjust() {
      std::make_heap(intervals_.begin(), intervals_.end(), compare);
    }

   private:
    std::vector<AllocatedInterval> intervals_;
    static bool compare(const AllocatedInterval& x,
                        const AllocatedInterval& y) {
      return x.next_use() < y.next_use();
    }
  };

  int64 AllocatedSize(const Buffer& buffer) const {
    HloInstruction* inst = buffer.defining_instruction->instruction;
    HloOpcode def_opcode = inst->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
      return buffer.size;
    }
  }

  Buffer& CreateBufferFromLogicalBuffer(Item* defining_instruction,
                                        const Shape& shape, bool live_out,
                                        const LogicalBuffer* logical_buffer) {
    int64 buffer_id = buffers_.size();
    UsesList users = GetUsers(instruction_list_, logical_buffer, buffer_id,
                              points_to_analysis_);
    buffers_.push_back(Buffer{buffer_id, size_function_(shape), live_out,
                              defining_instruction, nullptr, 0,
                              std::move(users), 0});
    swap_out_inst_.push_back(nullptr);
    last_use_inst_.push_back(nullptr);
    return buffers_.back();
  }

  void AddReleasedInternal(int64 size, Item* item);

  // alloc a size of memory
  void RegisterAlloc(BufferId bid, int64 size);

  // release all possible to release memory
  void ReleaseAll(const absl::flat_hash_set<int64>& sizes, Item* release_after);

  // adjust heaps for all used this round: the next use is changed;
  void AdjustAll(const absl::flat_hash_set<int64>& worklist);

  // try to alloc an interval with given size from free memory.
  bool AllocFreeMemory(int64 size);

  // try to alloc an already released interval.
  // If no such a interval, return nullptr.
  std::pair<Item*, int64> AllocReleasedMemory(int64 size);

  // try to alloc by releasing a buffer already in the set.
  std::pair<BufferId, Item*> AllocWithRelease(int64 size, Item* item);

  // get space for an item. Return the actual allocated size
  // because a larger buffer may be allocated to it
  int64 GetSpaceFor(int64 size, Item* item);

  void SelfCHECK(absl::string_view extra_msg = "") {
    if (memory_bound_ == InfiniteMemory) return;
    int64 allocated_size, released_size;
    allocated_size = released_size = 0;
    for (auto iter = allocated_buffers_.begin();
         iter != allocated_buffers_.end(); ++iter) {
      allocated_size += iter->first * iter->second.size();
    }
    for (auto iter = released_intervals_.begin();
         iter != released_intervals_.end(); ++iter) {
      released_size += iter->first * iter->second.size();
    }
    CHECK(allocated_size == memory_usage_);
    CHECK(allocated_size + released_size + free_memory_ == memory_bound_)
        << "In consistency:\n"
        << "memory bound is: " << memory_bound_
        << ",\n but allocated size is: " << allocated_size
        << ", released size is: " << released_size
        << ", free memory size is: " << free_memory_;
  }

  int64 free_memory_;
  HloComputation* computation_;
  InstructionList& instruction_list_;
  int64 memory_bound_;
  const HloSwapInsertion::ShapeSizeFunction& size_function_;
  const TuplePointsToAnalysis& points_to_analysis_;

  int64 memory_usage_;

  Item* prepare_for_ = nullptr;
  std::vector<Buffer> buffers_;
  std::vector<Item*> swap_out_inst_, last_use_inst_;

  // allocated buffers are stored by a list(btree_set) of heaps(priority_queue).
  // Each heap contains allocated buffers with the same size,
  // The list is in the order of the size
  std::map<int64, Heap> allocated_buffers_;

  std::map<int64, std::queue<Item*>> released_intervals_;
};
};  // namespace

MemoryRecorder::MemoryRecorder(
    HloComputation* computation, InstructionList& inst_list,
    const TuplePointsToAnalysis& points_to_analysis, int64 memory_bound,
    const HloSwapInsertion::ShapeSizeFunction& size_function)
    : computation_(computation),
      instruction_list_(inst_list),
      size_function_(size_function),
      memory_bound_(memory_bound),
      points_to_analysis_(points_to_analysis) {
  free_memory_ = memory_bound_;
  memory_usage_ = 0;
  PointsToSet::BufferSet live_out_set =
      points_to_analysis.GetPointsToSet(computation->root_instruction())
          .CreateFlattenedSet();
  std::map<const LogicalBuffer*, BufferId> logical_buffer_to_id;

  for (auto* item = inst_list.first(); item != nullptr;
       item = inst_list.next(item)) {
    const HloInstruction* const instruction = item->instruction;

    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
      Buffer* buffer;
      if (instruction->opcode() == HloOpcode::kWhile) {
        CHECK(false) << "while is not implemented now";
        // TODO
        const PointsToSet& operand_points_to =
            points_to_analysis.GetPointsToSet(instruction->operand(0));
        CHECK_EQ(operand_points_to.element(logical_buffer->index()).size(), 1);
        const LogicalBuffer* source_logical_buffer =
            operand_points_to.element(logical_buffer->index())[0];
        buffer = &buffers_.at(logical_buffer_to_id.at(source_logical_buffer));

        // Mark buffer as has indirect use and live out.
        buffer->live_out =
            buffer->live_out || ContainsKey(live_out_set, logical_buffer);

        // Add users of while to Buffer users.
        bool unused;
        // for (ItemUse& user_item : GetUsers(instruction_list, logical_buffer,
        //                                    points_to_analysis, &unused)) {
        //   auto existing_user_it = absl::c_find_if(
        //       buffer->users,
        //       [&](const ItemUse& use) { return user_item.user == use.user;
        //       });
        //   if (existing_user_it == buffer->users.end()) {
        //     buffer->unfinished_user_count++;
        //     user_item.user->buffers_used.push_back(buffer->id);
        //     buffer->users.push_back(user_item);
        //   }
        // }
      } else {
        buffer = &CreateBufferFromLogicalBuffer(
            inst_list.GetItem(logical_buffer->instruction()),
            logical_buffer->shape(), ContainsKey(live_out_set, logical_buffer),
            logical_buffer);
        item->buffers_defined.push_back(buffer->id);
      }
      logical_buffer_to_id[logical_buffer] = buffer->id;
    }
  }
}

void MemoryRecorder::AddReleasedInternal(int64 size, Item* item) {
  if (size == 0) {
    return;
  }
  auto iter = released_intervals_.find(size);
  if (iter == released_intervals_.end()) {
    auto inserted = std::queue<Item*>();
    inserted.push(item);
    released_intervals_.insert(std::make_pair(size, inserted));
  } else {
    iter->second.push(item);
  }
}

void MemoryRecorder::RegisterAlloc(BufferId bid, int64 size) {
  if (size == 0) {
    return;
  }
  auto iter = allocated_buffers_.find(size);
  if (iter == allocated_buffers_.end()) {
    Heap inserted;
    inserted.push(&buffers_.at(bid));
    allocated_buffers_.insert(std::make_pair(size, inserted));
  } else {
    iter->second.push(&buffers_.at(bid));
  }
  memory_usage_ += size;
}

void MemoryRecorder::ReleaseAll(const absl::flat_hash_set<int64>& sizes,
                                Item* release_after) {
  std::vector<BufferId> release_id;
  for (int64 size : sizes) {
    if (size == 0) {
      continue;
    }
    auto iter = allocated_buffers_.find(size);
    CHECK(iter != allocated_buffers_.end());
    auto& buffers = iter->second;
    CHECK(!buffers.empty());
    CHECK(buffers.top().next_use() == kInvalidPosition)
        << buffers.top().buffer()->id
        << " has next use at position: " << buffers.top().buffer()->next_use();
    while (!buffers.empty() && buffers.top().next_use() == kInvalidPosition) {
      const Buffer* buffer = buffers.top().buffer();
      release_id.push_back(buffer->id);
      buffers.pop();
      // insert to released set
      AddReleasedInternal(size, release_after);
    }
    if (buffers.empty()) {
      allocated_buffers_.erase(iter);
    }
  }
  for (BufferId bid : release_id) {
    memory_usage_ -= buffers_.at(bid).occupying_size;
    buffers_.at(bid).SetInGPU(nullptr, 0);
  }
  // return release_id;
}

void MemoryRecorder::AdjustAll(const absl::flat_hash_set<int64>& worklist) {
  for (int64 size : worklist) {
    if (size == 0) {
      continue;
    }
    auto iter = allocated_buffers_.find(size);
    CHECK(iter != allocated_buffers_.end());
    iter->second.adjust();
  }
}

bool MemoryRecorder::AllocFreeMemory(int64 size) {
  if (free_memory_ == InfiniteMemory)
    return true;  // the compute peak memory mode
  if (free_memory_ >= size) {
    free_memory_ -= size;
    return true;
  }
  return false;
}

// The return item can never be a swap out because
// swap out is created only in AllocWithRelease and consumed immediately
std::pair<Item*, int64> MemoryRecorder::AllocReleasedMemory(int64 size) {
  auto iter = released_intervals_.lower_bound(size);
  if (iter == released_intervals_.end()) {
    return std::make_pair(nullptr, 0);
  }
  Item* head = iter->second.front();
  int64 interval_len = iter->first;
  iter->second.pop();
  if (iter->second.empty()) {
    released_intervals_.erase(interval_len);
  }
  return std::make_pair(head, interval_len);
}

std::pair<BufferId, Item*> MemoryRecorder::AllocWithRelease(int64 size,
                                                            Item* item) {
  // We allocate buffers sequentially, so we need to avoid a later allocation
  // occupies an earlier allocation.
  // Observe that AllocWithRelease only allocates for:
  // 1. an operand swapped out, whose next use is exactly this instruction;
  // 2. a buffer defined by this instruction;
  // If the first happens, the best fit size has no available buffer(otherwise
  // next use is later) so try another size;(TODO)
  // The second is avoided by RegisterAlloc later;
  auto iter = allocated_buffers_.lower_bound(size);
  if (iter == allocated_buffers_.end()) {
    return std::make_pair(kInvalidBufferId, nullptr);
  }
  auto info = std::make_pair(iter->second.top().buffer()->id,
                             iter->second.top().buffer()->latest_alloc);
  memory_usage_ -= iter->second.top().buffer()->occupying_size;
  iter->second.pop();
  // do not clear empty heap, because the interval will be consumed immediately
  return info;
}

int64 MemoryRecorder::GetSpaceFor(int64 size, Item* item) {
  // TODO: more info: constant, entry parameter, live out...
  // try free memory first
  if (AllocFreeMemory(size)) {
    return size;
  }
  // try already released memory
  {
    auto release_after = AllocReleasedMemory(size);
    Item* release_after_inst = release_after.first;
    if (release_after_inst != nullptr) {
      VLOG(4) << "Alloc " << HumanReadableNumBytes(size) << " for "
              << item->instruction->ToShortString()
              << " with buffer released by "
              << release_after.first->instruction->ToShortString();
      instruction_list_.AddEdge(release_after_inst, item);
      return release_after.second;
    }
  }
  // cannot alloc at the released interval, swap out a buffer
  auto release = AllocWithRelease(size, item);
  BufferId release_bid = release.first;
  Item* release_inst = release.second;
  Buffer& release_buffer = buffers_.at(release_bid);
  int64 interval_size = release_buffer.occupying_size;
  VLOG(3) << "Alloc " << HumanReadableNumBytes(size) << " for "
          << item->instruction->ToShortString()
          << " with swapping out buffer of"
          << release_inst->instruction->ToShortString();

  if (swap_out_inst_.at(release_bid) != nullptr) {
    Item* last_use = last_use_inst_.at(release_bid);
    // consume immediately, do not register
    instruction_list_.AddEdge(last_use, item);
  } else {
    Item* swap_out = new Item;
    auto operand = release_buffer.defining_instruction->instruction;
    swap_out->is_swap = true;
    HloCustomCallInstruction* swap_out_inst = Cast<HloCustomCallInstruction>(
        computation_->AddInstruction(HloInstruction::CreateCustomCall(
            FormalShape, {operand}, kBuiltinSwapOutTarget,
            /*opaque=*/
            std::to_string(SwapKey++).append(";").append(
                std::to_string(SwapDoneEventKey)))));

    swap_out->instruction = swap_out_inst;
    swap_out_inst->set_custom_call_has_side_effect(true);
    instruction_list_.AddSwapOut(swap_out);

    instruction_list_.AddEdge(release_inst, swap_out);

    Item* last_use = last_use_inst_.at(release_bid);
    if (last_use != nullptr) {
      instruction_list_.AddEdge(last_use, swap_out);
    }
    swap_out_inst_[release_bid] = swap_out;

    // add swap done for the swap out,
    // the buffer is released only after swap done
    Item* swap_done = new Item;
    swap_done->is_swap = true;
    HloCustomCallInstruction* swap_done_inst = Cast<HloCustomCallInstruction>(
        computation_->AddInstruction(HloInstruction::CreateCustomCall(
            FormalShape, {operand}, kBuiltinSwapDoneTarget,
            /*opaque=*/std::to_string(SwapDoneEventKey++))));
    swap_done->instruction = swap_done_inst;
    swap_done_inst->set_custom_call_has_side_effect(true);
    swap_done->buffers_used.push_back(Operand{kInvalidBufferId, swap_out_inst});
    instruction_list_.AddEdge(swap_out, swap_done);
    instruction_list_.AddSwapDone(swap_out, swap_done);

    // add edge from swap out done to the item
    instruction_list_.AddEdge(swap_done, item);
    last_use_inst_[release_bid] = swap_done;
  }
  release_buffer.SetInGPU(nullptr, 0);
  return interval_size;
}

void MemoryRecorder::PrepareForInstruction(Item* item) {
  prepare_for_ = item;
  // prepare for operands
  for (auto& operand : item->buffers_used) {
    int64 bid = operand.bid;
    auto& buffer = buffers_.at(bid);
    if (buffer.InGPU()) {
      if (buffer.IsSwappedIn()) {
        instruction_list_.AddEdge(
            instruction_list_.GetSwapDone(buffer.latest_alloc), item);
        buffer.defining_instruction->instruction->ReplaceUseWith(
            item->instruction, buffer.latest_alloc->instruction);
      }
      last_use_inst_[bid] = item;
      continue;
    }
    // swap in from CPU
    Item* swap_out = swap_out_inst_.at(bid);

    CHECK(swap_out != nullptr);

    Item* swap_in = new Item;
    swap_in->is_swap = true;

    std::string opaque =
        OpaqueByIndex(
            (Cast<HloCustomCallInstruction>(swap_out->instruction)->opaque()),
            0)
            .append(";")
            .append(std::to_string(SwapDoneEventKey));
    HloCustomCallInstruction* swap_in_inst = Cast<HloCustomCallInstruction>(
        computation_->AddInstruction(HloInstruction::CreateCustomCall(
            operand.instruction->shape(), {}, kBuiltinSwapInTarget,
            /*opaque=*/opaque)));

    swap_in->instruction = swap_in_inst;
    swap_in_inst->set_custom_call_has_side_effect(true);
    instruction_list_.AddSwapIn(swap_in);

    instruction_list_.AddEdge(swap_out, swap_in);
    // swap in directly sync(skip swap done)

    Item* swap_done = new Item;
    swap_done->is_swap = true;
    HloCustomCallInstruction* swap_done_inst = Cast<HloCustomCallInstruction>(
        computation_->AddInstruction(HloInstruction::CreateCustomCall(
            FormalShape, {swap_in_inst}, kBuiltinSwapDoneTarget,
            /*opaque=*/std::to_string(SwapDoneEventKey++))));
    swap_done->instruction = swap_done_inst;
    swap_done_inst->set_custom_call_has_side_effect(true);
    swap_done->buffers_used.push_back(Operand{kInvalidBufferId, swap_in_inst});
    instruction_list_.AddSwapDone(swap_in, swap_done);

    int64 interval_size = GetSpaceFor(AllocatedSize(buffer), swap_in);
    buffer.SetInGPU(swap_in, interval_size);
    RegisterAlloc(bid, interval_size);

    operand.instruction->ReplaceUseWith(item->instruction, swap_in_inst);
    instruction_list_.AddEdge(swap_done, item);
    last_use_inst_.at(bid) = item;
  }
  SelfCHECK();
  // prepare for results
  absl::InlinedVector<std::pair<BufferId, int64>, 3> intervals_for_defined;
  for (auto bid : item->buffers_defined) {
    auto& buffer = buffers_.at(bid);
    int interval_size = GetSpaceFor(AllocatedSize(buffer), item);
    buffer.SetInGPU(item, interval_size);
    intervals_for_defined.push_back(std::make_pair(bid, interval_size));
  }
  // Register allocation later to avoid: a defined buffer occupies the space of
  // another.
  for (auto& interval : intervals_for_defined) {
    RegisterAlloc(interval.first, interval.second);
  }
  // computation cost
  SelfCHECK();
}

void MemoryRecorder::RecycleAfterInstruction(Item* item) {
  prepare_for_ = nullptr;
  absl::flat_hash_set<int64> worklist;
  absl::flat_hash_set<int64> to_adjust_sizes;
  for (auto& operand : item->buffers_used) {
    int64 bid = operand.bid;
    auto& buffer = buffers_.at(bid);

    ++buffer.next_use_index;
    to_adjust_sizes.insert(buffer.occupying_size);
    if (buffer.next_use() == kInvalidPosition) {
      worklist.insert(buffer.occupying_size);
    }
  }

  for (auto bid : item->buffers_defined) {
    auto& buffer = buffers_.at(bid);
    if (buffer.next_use() == kInvalidPosition) {
      worklist.insert(buffer.occupying_size);
    }
  }
  AdjustAll(to_adjust_sizes);
  ReleaseAll(worklist, item);
  SelfCHECK();
}

StatusOr<int64> HloSwapInsertion::ComputePeakMemory(
    HloComputation* computation, const HloInstructionSequence& order) const {
  InstructionList instruction_list(order);
  MemoryRecorder tracker(computation, instruction_list, *points_to_analysis_,
                         InfiniteMemory, size_function_);
  int64 peak_memory = tracker.memory_usage();
  for (auto item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    tracker.PrepareForInstruction(item);
    peak_memory =
        std::max<int64>(peak_memory, tracker.memory_usage() + callee_usage);
    tracker.RecycleAfterInstruction(item);
  }
  VLOG(1) << "Peak memory for " << computation->name() << ": "
          << HumanReadableNumBytes(peak_memory);
  return peak_memory;
}

StatusOr<int64> HloSwapInsertion::CalledComputationsMemoryUsage(
    const HloInstruction* instruction) const {
  const CallSite* callsite =
      call_graph_->GetNode(instruction->parent()).GetCallSite(instruction);
  if (callsite == nullptr || callsite->context() == CallContext::kParallel) {
    return 0;
  }
  int64 callee_usage = 0;
  for (const HloComputation* computation : callsite->called_computations()) {
    TF_RET_CHECK(ContainsKey(computation_peak_memory_, computation));
    callee_usage += computation_peak_memory_.at(computation);
  }
  return callee_usage;
}

StatusOr<bool> HloSwapInsertion::SwapInsertionComputation(
    HloComputation* computation, HloSchedule* schedule,
    int64 memory_limit_bytes) {
  InstructionList instruction_list(schedule->sequence(computation));
  MemoryRecorder tracker(computation, instruction_list, *points_to_analysis_,
                         memory_limit_bytes_, size_function_);

  const CallGraphNode& call_graph_node = call_graph_->GetNode(computation);

  VLOG(1) << "memory limit is: " << memory_limit_bytes_;

  int64 peak_memory = tracker.memory_usage();
  bool changed = false;

  for (auto* item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;

    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));

    tracker.PrepareForInstruction(item);
    peak_memory =
        std::max<int64>(peak_memory, tracker.memory_usage() + callee_usage);
    VLOG(1) << "memory usage after computation is: " << tracker.memory_usage();
    // // callee usage is too large, try to swap the callee.(todo)
    // const CallSite* callsite = call_graph_node.GetCallSite(instruction);
    // if (callsite != nullptr &&
    //     callsite->context() == CallContext::kSequential &&
    //     callee_usage + tracker.memory_usage() > memory_limit_bytes_) {
    //   for (HloComputation* called_computation :
    //        callsite->called_computations()) {
    //     int64 subcomputation_memory_limit_bytes =
    //         std::max<int64>(0, memory_limit_bytes_ - tracker.memory_usage());
    //     VLOG(3) << "dive into subcomputation";
    //     TF_ASSIGN_OR_RETURN(
    //         bool subcomputation_changed,
    //         SwapInsertionComputation(called_computation, schedule,
    //                                  subcomputation_memory_limit_bytes));
    //     changed |= subcomputation_changed;
    //   }
    // }
    tracker.RecycleAfterInstruction(item);
    VLOG(1) << "memory usage after recycle is: " << tracker.memory_usage();
  }
  std::cerr << peak_memory << "\n";
  
  schedule->set_sequence(computation, std::move(instruction_list.toSequence()));
  TF_RETURN_IF_ERROR(schedule->Verify());
  return true;
}

StatusOr<bool> HloSwapInsertion::Run(HloModule* module) {
  HloMemoryScheduler scheduler(
      [this](const BufferValue& buffer) {
        return size_function_(buffer.shape());
      },
      ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
  if (!module->has_schedule()) {
    scheduler.Run(module);
  }
  TF_RET_CHECK(module->has_schedule());
  TF_ASSIGN_OR_RETURN(points_to_analysis_, TuplePointsToAnalysis::Run(module));

  int64 module_output_size = 0;
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&module_output_size, module, this](const Shape& subshape,
                                          const ShapeIndex& output_index) {
        module_output_size += size_function_(subshape);
      });

  call_graph_ = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph_->VisitNodes(
      [this, module](const CallGraphNode& node) -> Status {
        if (node.context() == CallContext::kSequential) {
          TF_ASSIGN_OR_RETURN(
              computation_peak_memory_[node.computation()],
              ComputePeakMemory(node.computation(), module->schedule().sequence(
                                                        node.computation())));
        }
        return Status::OK();
      },
      /*visit_unreachable_nodes=*/false));

  // The peak memory usage of the module equals the peak memory use of the entry
  // computation plus the output size of the computation. This is because the
  // peak memory for a computation does not include the output as this is
  // typically accounted for in the caller.
  const int64 before_peak_memory =
      computation_peak_memory_.at(module->entry_computation()) +
      module_output_size;
  VLOG(1) << "Peak memory usage of module (before): "
          << HumanReadableNumBytes(before_peak_memory);

  if (before_peak_memory < memory_limit_bytes_) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      bool changed,
      SwapInsertionComputation(module->entry_computation(), &module->schedule(),
                               memory_limit_bytes_));
  // TODO: replace by a special scheduler instead
  return changed;
}

};  // namespace xla