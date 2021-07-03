#include "tensorflow/compiler/xla/service/hlo_swap_insertion.h"

#include <algorithm>
#include <iterator>
#include <memory>
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

// TODO(yonghao): has_indirect_use: this leads to redundant Buffers,
// e.g. Tuple and elements inside are different buffers sharing the same memory.
// todo: store the space for parameter specifically, instead of
// wasting time and space for it;
namespace xla {

namespace {

using ::tensorflow::strings::HumanReadableNumBytes;

using BufferId = int64;
using BufferIdList = absl::InlinedVector<BufferId, 3>;

static const Shape keyShape = ShapeUtil::MakeNil();
static int64 swapInEventKey = 0;
// this key is for SwapDone, since currently control flow dependency is lost in
// MLIR and thunk_schedule.
class Item {
 private:
  Item* next = nullptr;
  Item* prev = nullptr;

 public:
  HloInstruction* instruction;
  BufferIdList buffers_defined, buffers_output, buffers_used;
  // the order of this computation for all not-swap instructions
  int64 position;
  bool isSwap = false;

  friend class InstructionList;
};

struct ItemUse {
  Item* user;
  int64 operand_number;
  absl::optional<int64> index;

  ItemUse(Item* user, int64 op_num, absl::optional<int64> index)
      : user(user), operand_number(op_num), index(index) {}
  bool operator==(const ItemUse& other) const {
    return user == other.user && operand_number == other.operand_number &&
           index == other.index;
  }
};

using UsesList = absl::InlinedVector<ItemUse, 3>;

class InstructionList {
 public:
  explicit InstructionList(const HloInstructionSequence& order) {
    int64 position = 0;
    Item* last = nullptr;
    for (HloInstruction* inst : order.instructions()) {
      // Add a new item to the linked list.
      Item* item = new Item;
      item->next = nullptr;
      item->prev = last;
      if (last == nullptr) {
        first_ = item;
      } else {
        last->next = item;
      }
      last = item;

      // Initially position numbers are uniquely assigned in order. Later as
      // instructions are added with InsertBefore* methods, some instructions
      // may have duplicate position numbers, but the values will be guaranteed
      // to be monotonically increasing through the list, and so is still useful
      // for quickly(-ish) determining the order of arbitrary instructions in
      // the list.
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
  }

  void addEdge(Item* from, Item* to) {
    if (from->isSwap || to->isSwap) {
      from->instruction->AddControlDependencyTo(to->instruction);
      VLOG(3) << "\tadd edge from: " << from->instruction->ToString()
              << ", to: " << to->instruction->ToString();
    }
  };

  Item* first() const { return first_; }

  Item* GetItem(const HloInstruction* inst) const {
    auto iter = item_map_.find(inst);
    CHECK(iter != item_map_.end()) << "Did not find " << inst->name();
    return iter->second;
  }

  Item* next(Item* item) const { return item->next; }

  // TODO: HloInstructionSequence sequence() {}

 private:
  Item* first_;

  absl::flat_hash_map<const HloInstruction*, Item*> item_map_;
};

// Return the items which use the given LogicalBuffer. Sets
// has_indirect_users to whether any of the uses is indirect. A use is indirect
// if the instruction defining logical_buffer is not an operand of the use. This
// can happen via buffer aliasing (eg, tuples).
UsesList GetUsers(const InstructionList& instruction_list,
                  const LogicalBuffer* logical_buffer,
                  const TuplePointsToAnalysis& points_to_analysis,
                  bool* has_indirect_users) {
  UsesList users;
  // To identify uses iterate through all HloInstruction users of the
  // BufferAliases of the logical buffer.
  *has_indirect_users = false;
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
      if (buffer_alias.instruction() != logical_buffer->instruction() /*&&
          !IsSupportedIndirectUser(buffer_alias.instruction())*/) {
        // has_indirect_user
        // LOG(WARNING) << user->ToShortString()
        //              << " has indirect uses, may lead to inaccurate result";
        // LOG(WARNING) << "buffer alias inst is: "
        //              << buffer_alias.instruction()->ToShortString()
        //              << "; logical buffer inst is: "
        //              << logical_buffer->instruction()->ToShortString()
        //              << "; buffer is: " << logical_buffer->ToString();
        *has_indirect_users = true;
      }
      // A buffer may be used by the instruction via more than one alias. For
      // example, a buffer which appears in more than one element of a tuple.
      Item* user_item = instruction_list.GetItem(user);
      absl::optional<int64> user_index =
          logical_buffer->index().size() != 1
              ? absl::nullopt
              : absl::make_optional(logical_buffer->index().back());
      for (int64 op_idx : user->OperandIndices(buffer_alias.instruction())) {
        if (!absl::c_linear_search(
                users,
                ItemUse{user_item, static_cast<int>(op_idx), user_index})) {
          users.push_back(
              ItemUse{user_item, static_cast<int>(op_idx), user_index});
        }
      }
    }
  }
  return users;
}

class MemoryRecorder {
 public:
  MemoryRecorder(HloComputation* computation, InstructionList& inst_list,
                 const TuplePointsToAnalysis& points_to_analysis,
                 int64 memory_bound,
                 const HloSwapInsertion::ShapeSizeFunction& size_function);
  // try to allocate on free memory
  bool allocFreeMemory(int64 size);
  // try to allocate on released memory
  // if success, the second is the last occurance: a swap out or discard
  std::vector<Item*> allocReleasedMemory(int64& size);

  Status PrepareForInstruction(Item* item, int64 callee_usage);

  Status RecycleAfterInstruction(Item* item);

  int64 memory_usage() const { return memory_usage_; }

 private:
  struct Buffer {
    // The unique id of this Buffer. This value is equal to the buffer's index
    // in the vector buffers_.
    const BufferId id;

    // The instruction which defines this buffer. Not a swapIn
    Item* defining_instruction;

    Item* latest_alloc;

    Shape shape;

    // The materialized size of the buffer in bytes.
    const int64 size;

    // Whether this buffer is live-out of the computation.
    bool live_out;

    // Whether this buffer has indirect uses. Ie, an instruction which is not a
    // user of defining_instruction uses this buffer. This can occur due to
    // buffer aliasing (eg, tuples). We only consider get-tuple-element and
    // bitcast, and do not swap out other indirect uses
    bool has_indirect_uses;

    // Position in the tuple this buffer definition lives in.
    ShapeIndex index;

    // The instructions which use this buffer.
    UsesList users;

    // The number of users (HloInstructions) of this buffer which have not yet
    // been placed in the sequence.
    int64 unfinished_user_count;

    // Never swaps out a parameter
    bool isParameter;

    const LogicalBuffer* logical_buffer;

    // return if the buffer is already in GPU
    bool inGPU() { return latest_alloc != nullptr; }

    bool isSwappedIn() { return latest_alloc != defining_instruction; }

    // set if the buffer is in GPU
    void setInGPU(Item* swap_in_inst) {
      // check: defining_instruction == nullptr << "already in with define
      // instruction " << defining_instruction->ToString() << " while meeting
      // another define instruction " << swap_in_inst->ToString();
      latest_alloc = swap_in_inst;
    }
  };

  // release with some policy, return the item defines the released buffer
  std::vector<std::pair<BufferId, Item*>> canAllocWithRelease(
      int64 size, Item* item) const {
    // todo: not only FIFO
    std::vector<std::pair<BufferId, Item*>> result;
    for (auto& p : alloced) {
      // cannot release a will used buffer
      if (absl::c_linear_search(item->buffers_used, p.first)) continue;
      if (absl::c_linear_search(preparing_for->buffers_used, p.first)) continue;
      int64 bufferSize = AllocatedSize(buffers_.at(p.first));
      if (bufferSize == 0 || !buffers_.at(p.first).index.empty()) continue;
      result.push_back(p);
      if (size <= bufferSize) {
        size = 0;
        break;
      } else {
        size -= bufferSize;
      }
    }
    CHECK(size == 0) << "cannot alloc enough space";
    return result;
  }

  // register a swap in to GPU/computed by exec: record the size and producer
  void registerAlloc(Item* define, BufferId bid, int64 size) {
    alloced.push_back(std::make_pair(bid, define));
    memory_usage_ += size;
  }

  // register a swap out to CPU/release after exec: record the size and
  // instruction
  void registerRelease(Item* lastUse, BufferId bid, int64 size) {
    // todo: avoid linear search in the alloced
    VLOG(3) << "\treleasing " << bid;
    for (auto iter = alloced.begin(); iter != alloced.end(); ++iter) {
      if (iter->first == bid) {
        alloced.erase(iter);
        break;
      }
    }
    // todo: merge different releases for the same instruction
    if (size != 0) {
      released_after.emplace_back(std::make_pair(lastUse, size));
      memory_usage_ -= size;
    }
  }

  // swap out already located buffers to prepare for the given buffer
  void getSpaceFor(int64 size, Item* item) {
    if (!allocFreeMemory(size)) {
      // no enough free memory, try released memory
      auto released = allocReleasedMemory(size);
      // can swap in only after the swap out/discard ends
      for (auto rb : released) {
        instruction_list.addEdge(rb, item);
      }
      if (size == 0) {
        VLOG(4) << "\tcan alloc with already released memory";
      } else {
        // find existed buffers and release them
        int64 rest_size = size;
        auto toReleasePairs = canAllocWithRelease(size, item);
        VLOG(3) << "\tswap out when getting space for instruction: "
                << item->instruction->ToString();

        for (auto& toRelease : toReleasePairs) {
          BufferId toReleaseBid = toRelease.first;
          Buffer& toReleaseBuffer = buffers_.at(toReleaseBid);
          // CHECK(toReleasBuffer != nullptr) << "cannot alloc enough memory"
          rest_size -= AllocatedSize(toReleaseBuffer);
          toReleaseBuffer.setInGPU(nullptr);
          if (swap_out_inst.at(toReleaseBid) != nullptr) {
            // is already swapped out, only need to discard
            Item* lastUseInst = last_use_inst.at(toReleaseBid);
            registerRelease(lastUseInst, toReleaseBid,
                            rest_size > 0 ? 0 : -rest_size);
            instruction_list.addEdge(lastUseInst, item);
          } else {
            // add the swap out to CPU instruction
            Item* swapOut = new Item;
            swapOut->isSwap = true;
            swapOut->instruction =
                computation_->AddInstruction(HloInstruction::CreateCustomCall(
                    keyShape,
                    {toReleaseBuffer.defining_instruction->instruction},
                    "__builtin$SwapOut",
                    /*opaque=*/std::to_string(swap_key_++)));
            Cast<HloCustomCallInstruction>(swapOut->instruction)
                ->set_custom_call_has_side_effect(true);
            // std::cerr << "\tcreate swap out for buffer: " << toReleaseBid
            //           << "\n";
            registerRelease(swapOut, toReleaseBid,
                            rest_size > 0 ? 0 : -rest_size);
            // add control flow edge from its producer to the swap instruction
            instruction_list.addEdge(toRelease.second, swapOut);
            Item* lastUseItem = last_use_inst.at(toReleaseBid);
            if (lastUseItem != nullptr)
              instruction_list.addEdge(lastUseItem, swapOut);
            // todo: instead, make Item* in released_after an inlined
            // buffer<Item *> and add this to the buffer
            last_use_inst.at(toReleaseBid) = swapOut;

            instruction_list.addEdge(swapOut, item);
            swap_out_inst[toReleaseBid] = swapOut;
          }
        }
        memory_usage_ -= size;
      }
    }
  }

  Buffer& CreateBufferFromLogicalBuffer(
      const LogicalBuffer* logical_buffer,
      const TuplePointsToAnalysis& points_to_analysis, bool live_out) {
    bool has_indirect_uses = false;
    UsesList users = GetUsers(instruction_list, logical_buffer,
                              points_to_analysis, &has_indirect_uses);
    return NewBuffer(instruction_list.GetItem(logical_buffer->instruction()),
                     logical_buffer->shape(), logical_buffer->index(),
                     std::move(users), live_out, has_indirect_uses,
                     logical_buffer);
  }

  Buffer& NewBuffer(Item* defining_instruction, const Shape& shape,
                    const ShapeIndex& index, UsesList&& uses, bool live_out,
                    bool has_indirect_uses,
                    const LogicalBuffer* logical_buffer) {
    int buffer_id = buffers_.size();
    auto get_num_of_unique_users = [](const UsesList& uses) -> int64 {
      absl::flat_hash_set<Item*> users_set;
      for (const ItemUse& use : uses) {
        users_set.insert(use.user);
      }
      return users_set.size();
    };
    buffers_.push_back(Buffer{
        buffer_id, defining_instruction, defining_instruction, shape,
        size_function_(shape), live_out, has_indirect_uses, index, uses,
        get_num_of_unique_users(uses),
        defining_instruction->instruction->opcode() == HloOpcode::kParameter,
        logical_buffer});
    swap_out_inst.push_back(nullptr);
    last_use_inst.push_back(nullptr);
    return buffers_.back();
  }

  int64 AllocatedSize(const Buffer& buffer) const {
    HloInstruction* inst = buffer.defining_instruction->instruction;
    HloOpcode def_opcode = inst->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
      return buffer.size;
    }
  }

  void SelfCHECK(absl::string_view extra_msg = "") {
    int64 alloced_memory = 0, released_memory = 0;
    std::string allocated = "";
    VLOG(4) << "start self checking...";
    for (auto iter = released_after.begin(); iter != released_after.end();
         ++iter) {
      released_memory += iter->second;
    }
    for (auto iter = alloced.begin(); iter != alloced.end(); ++iter) {
      alloced_memory += AllocatedSize(buffers_[iter->first]);
      allocated.append(" ");
      allocated.append(std::to_string(iter->first));
    }
    VLOG(1) << "\tfree: " << free_memory << ", released: " << released_memory
            << ", alloced: " << alloced_memory;
    if (memory_bound_ == -1) return;
    CHECK(alloced_memory == memory_usage_);
    CHECK(alloced_memory + released_memory + free_memory == memory_bound_)
        << "memory leak. "
        << "\nallocated: " << alloced_memory
        << ", released: " << released_memory << ", free: " << free_memory
        << ", bound: " << memory_bound_ << "\nmeta: " << extra_msg;
  }

  HloComputation* computation_;
  InstructionList& instruction_list;
  std::vector<Buffer> buffers_;
  const HloSwapInsertion::ShapeSizeFunction& size_function_;
  const TuplePointsToAnalysis& points_to_analysis_;
  // last_use_inst is assigned only after swap out is assigned
  std::vector<Item*> swap_out_inst, last_use_inst;
  int64 swap_key_ = 0;
  Item* preparing_for = nullptr;
  int64 memory_bound_;
  int64 free_memory, memory_usage_;
  absl::flat_hash_map<Item*, Item*> swap_done_map_;
  std::vector<std::pair<Item*, int64>> released_after;

  std::vector<std::pair<BufferId, Item*>> alloced;
};
};  // namespace

MemoryRecorder::MemoryRecorder(
    HloComputation* computation, InstructionList& inst_list,
    const TuplePointsToAnalysis& points_to_analysis, int64 memory_bound,
    const HloSwapInsertion::ShapeSizeFunction& size_function)
    : computation_(computation),
      instruction_list(inst_list),
      size_function_(size_function),
      memory_bound_(memory_bound),
      points_to_analysis_(points_to_analysis) {
  free_memory = memory_bound;
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
      // TODO: work on the while later
      if (instruction->opcode() == HloOpcode::kWhile) {
        CHECK(false) << "while is ignored as not implemented now";
        // The while instruction defines no new buffers. Instead it reuses the
        // buffers of its operand. Find the Buffer of its operand at the
        // proper ShapeIndex.
        const PointsToSet& operand_points_to =
            points_to_analysis.GetPointsToSet(instruction->operand(0));
        CHECK_EQ(operand_points_to.element(logical_buffer->index()).size(), 1);
        const LogicalBuffer* source_logical_buffer =
            operand_points_to.element(logical_buffer->index())[0];
        buffer =
            &buffers_.at(logical_buffer_to_id.at(source_logical_buffer));

        // Mark buffer as has indirect use and live out.
        buffer->has_indirect_uses = true;
        buffer->live_out =
            buffer->live_out || ContainsKey(live_out_set, logical_buffer);

        // Add users of while to Buffer users.
        bool unused;
        for (ItemUse& user_item : GetUsers(instruction_list, logical_buffer,
                                           points_to_analysis, &unused)) {
          auto existing_user_it = absl::c_find_if(
              buffer->users,
              [&](const ItemUse& use) { return user_item.user == use.user; });
          if (existing_user_it == buffer->users.end()) {
            buffer->unfinished_user_count++;
            user_item.user->buffers_used.push_back(buffer->id);
            buffer->users.push_back(user_item);
          }
        }

      } else {
        buffer = &CreateBufferFromLogicalBuffer(
            logical_buffer, points_to_analysis,
            ContainsKey(live_out_set, logical_buffer));
        item->buffers_defined.push_back(buffer->id);
        for (ItemUse& user : buffer->users) {
          if (!absl::c_linear_search(user.user->buffers_used, buffer->id)) {
            user.user->buffers_used.push_back(buffer->id);
          }
        }
      }
      logical_buffer_to_id[logical_buffer] = buffer->id;
    }

    for (const LogicalBuffer* logical_buffer :
         points_to_analysis.GetPointsToSet(instruction).CreateFlattenedSet()) {
      item->buffers_output.push_back(logical_buffer_to_id[logical_buffer]);
    }
  }
}

bool MemoryRecorder::allocFreeMemory(int64 size) {
  if (free_memory == -1) return true;  // the compute peak memory mode
  if (free_memory >= size) {
    free_memory -= size;
    return true;
  }
  return false;
}

std::vector<Item*> MemoryRecorder::allocReleasedMemory(int64& size) {
  // alloc as much as possible
  std::vector<Item*> result;
  Item* tail = nullptr;
  for (auto iter = released_after.begin(); iter != released_after.end();) {
    result.push_back(iter->first);
    if (size <= iter->second) {
      iter->second -= size;
      if (iter->second == 0) {
        released_after.erase(iter);
      }
      size = 0;
      break;
    } else {
      size -= iter->second;
      auto bak = iter;
      ++iter;
      released_after.erase(bak);
    }
  }
  return result;
}

Status MemoryRecorder::PrepareForInstruction(Item* item, int64 callee_usage) {
  // cannot release buffers used by the preparing_for item.
  preparing_for = item;
  // std::cerr << "prepare for: " << item->instruction->ToShortString() << "\n";
  for (auto bid : item->buffers_used) {
    auto& buffer = buffers_.at(bid);
    if (buffer.inGPU()) {
      if (buffer.isSwappedIn()) {
        instruction_list.addEdge(swap_done_map_[buffer.latest_alloc], item);
        buffer.defining_instruction->instruction->ReplaceUseWith(
            item->instruction, buffer.latest_alloc->instruction);
      }
      last_use_inst.at(bid) = item;
      continue;
    }
    // not in GPU, need a swap in
    // std::cerr << "buffer needs to be swapped in. From: "
    //           << buffer.logical_buffer->instruction()->ToShortString()
    //           << ", idx: " << buffer.index.ToString()
    //           << ", shape: " << buffer.shape.ToString() << "\n";
    Item* swapOut = swap_out_inst.at(bid);
    CHECK(swapOut != nullptr) << "Not in GPU but not swapped out";

    for (const BufferAlias& buffer_alias :
         points_to_analysis_.GetBufferAliases(*(buffer.logical_buffer))) {
      for (const HloInstruction* user : buffer_alias.instruction()->users()) {
        if (user == item->instruction &&
            !points_to_analysis_.DoesNotUseOperandBuffer(
                buffer_alias.instruction(), buffer_alias.index(), user)) {
          Item* swapIn = new Item;
          swapIn->isSwap = true;
          std::string opaque =
              Cast<HloCustomCallInstruction>(swapOut->instruction)->opaque();
          opaque.append(";" + std::to_string(swapInEventKey));
          HloCustomCallInstruction* swapInInst = Cast<HloCustomCallInstruction>(
              computation_->AddInstruction(HloInstruction::CreateCustomCall(
                  buffer_alias.instruction()->shape(), {}, "__builtin$SwapIn",
                  /*opaque=*/opaque)));

          // std::cerr << "buffer alias inst is: "
          //           << buffer_alias.instruction()->ToShortString()
          //           << ", shape is: "
          //           << buffer_alias.instruction()->shape().ToString() <<
          //           "\n";
          swapIn->instruction = swapInInst;
          swapInInst->set_custom_call_has_side_effect(true);
          instruction_list.addEdge(swapOut, swapIn);

          Item* swapDone = new Item;
          swapDone->isSwap = true;
          HloCustomCallInstruction* swapDoneInst =
              Cast<HloCustomCallInstruction>(
                  computation_->AddInstruction(HloInstruction::CreateCustomCall(
                      keyShape, {swapInInst}, "__builtin$SwapDone",
                      /*opaque=*/std::to_string(swapInEventKey++))));
          swapDone->instruction = swapDoneInst;
          swapDoneInst->set_custom_call_has_side_effect(true);

          VLOG(3) << "\tcreate swap in for buffer: " << bid;
          // if already discarded, swap in after discarded
          Item* lastUse = last_use_inst.at(bid);
          if (lastUse != nullptr) {
            instruction_list.addEdge(lastUse, swapIn);
          }

          buffer.setInGPU(swapIn);
          swap_done_map_[swapIn] = swapDone;
          getSpaceFor(AllocatedSize(buffer), swapIn);
          // add swap in to GPU instruction
          registerAlloc(swapIn, bid, buffer.size);
          // add control flow edge from the swap to its user
          instruction_list.addEdge(swapIn, item);
          buffer_alias.instruction()->ReplaceUseWith(item->instruction,
                                                     swapIn->instruction);
          instruction_list.addEdge(swapDone, item);
        }
      }
    }
    last_use_inst.at(bid) = item;
  }
  SelfCHECK("end preparing for used buffers");
  int64 cnt = 0;
  for (auto bid : item->buffers_defined) {
    auto& buffer = buffers_.at(bid);
    // CHECK(buffer.defining_instruction == item);
    VLOG(3) << "\tgetting space for buffers_defined: " << bid
            << ", space size: " << AllocatedSize(buffer);
    getSpaceFor(AllocatedSize(buffer), item);
    registerAlloc(item, bid, AllocatedSize(buffer));
    SelfCHECK(std::string("end preparing for defined buffers, idx: ")
                  .append(std::to_string(cnt++)));
  }
  getSpaceFor(callee_usage, item);
  SelfCHECK("end preparing for instruction");
  return Status::OK();
}

Status MemoryRecorder::RecycleAfterInstruction(Item* item) {
  preparing_for = nullptr;
  for (auto bid : item->buffers_used) {
    auto& buffer = buffers_.at(bid);
    // if no other use, recycle
    if (buffer.isParameter || buffer.live_out) {
      VLOG(4) << "\tbid is: " << bid
              << ", skip because is parameter or live out";
      continue;
    }
    --buffer.unfinished_user_count;
    VLOG(4) << "\tbid is: " << bid
            << ", user cnt is: " << buffer.unfinished_user_count;
    if (buffer.unfinished_user_count == 0)
      registerRelease(item, bid, AllocatedSize(buffer));
  }
  for (auto bid : item->buffers_defined) {
    auto& buffer = buffers_.at(bid);
    // if no other use, recycle
    if (buffer.unfinished_user_count == 0) {
      registerRelease(item, bid, AllocatedSize(buffer));
    }
  }
  SelfCHECK();
  return Status::OK();
}

StatusOr<int64> HloSwapInsertion::ComputePeakMemory(
    HloComputation* computation, const HloInstructionSequence& order) const {
  InstructionList instruction_list(order);
  MemoryRecorder tracker(computation, instruction_list, *points_to_analysis_,
                         -1, size_function_);
  int64 peak_memory = tracker.memory_usage();
  for (auto item = instruction_list.first(); item != nullptr;
       item = instruction_list.next(item)) {
    const HloInstruction* instruction = item->instruction;
    TF_ASSIGN_OR_RETURN(int64 callee_usage,
                        CalledComputationsMemoryUsage(instruction));
    TF_RETURN_IF_ERROR(tracker.PrepareForInstruction(item, 0));
    peak_memory =
        std::max<int64>(peak_memory, tracker.memory_usage() + callee_usage);
    TF_RETURN_IF_ERROR(tracker.RecycleAfterInstruction(item));
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
    // callee usage is too large, try to swap the callee.
    const CallSite* callsite = call_graph_node.GetCallSite(instruction);
    if (callsite != nullptr &&
        callsite->context() == CallContext::kSequential &&
        callee_usage + tracker.memory_usage() > memory_limit_bytes_) {
      // todo: incorrect placement
      for (HloComputation* called_computation :
           callsite->called_computations()) {
        int64 subcomputation_memory_limit_bytes =
            std::max<int64>(0, memory_limit_bytes_ - tracker.memory_usage());
        VLOG(3) << "dive into subcomputation";
        TF_ASSIGN_OR_RETURN(
            bool subcomputation_changed,
            SwapInsertionComputation(called_computation, schedule,
                                     subcomputation_memory_limit_bytes));
        changed |= subcomputation_changed;
      }
    }

    TF_RETURN_IF_ERROR(tracker.PrepareForInstruction(item, callee_usage));
    peak_memory =
        std::max<int64>(peak_memory, tracker.memory_usage() + callee_usage);
    VLOG(1) << "memory usage after computation is: " << tracker.memory_usage();
    TF_RETURN_IF_ERROR(tracker.RecycleAfterInstruction(item));
    VLOG(1) << "memory usage after recycle is: " << tracker.memory_usage();
  }
  std::cerr << peak_memory << "\n";
  return true;  // todo
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
  // reschedule because new instructions are inserted.
  TF_ASSIGN_OR_RETURN(changed, scheduler.Run(module));
  // TODO: replace by a special scheduler instead
  return changed;
}

};  // namespace xla