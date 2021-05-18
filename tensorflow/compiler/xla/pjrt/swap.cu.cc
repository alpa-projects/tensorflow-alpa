#include "tensorflow/compiler/xla/pjrt/swap.h"

#include <stdlib.h>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/shape_util.h"

#ifdef GOOGLE_CUDA
namespace xla{
// 1. parse the Shape: ShapeUtil::ForEachSubshape(not only leaf shapes)/ShapeUtil::GetLeafShapes
// 2. get the dtype(another opaque?)    TODO
// 3. use the dtype to cast     TODO
// 4. use the Shape to do Memcpy: the length of buffers are the same as Shapes

// assume only float is solved
// TODO: cudaStream_t

// ShapeUtil::ByteSizeOf

// the swap recorder uses a linked list to record inserted bufferInfoMap, the key is the address to help find the result

namespace {

using bufferInfoMap = absl::flat_hash_map<const ShapeIndex, void *>;
using key_t = int64;


class SwapRecorder {
private: 
  struct Node {
    Node *prev;
    Node *next;
    bufferInfoMap *value;
    Node(Node *prev_, Node *next_, bufferInfoMap *value_)
      :prev(prev_), next(next_), value(value_) {}
  };
public: 
  explicit SwapRecorder() {
    head = nullptr;
  }
  ~SwapRecorder() {
    // TODO: release all memory in `delete value`
    Node *cur = head;
    while (cur != nullptr) {
      Node *tmp = cur;
      cur = cur->next;
      if (tmp->value != nullptr) {
          delete tmp->value;
      }
      delete tmp;
    }
  }
  key_t insert(bufferInfoMap *value) {
    // TODO: acquire the mutex
    Node *node = new Node(nullptr, nullptr, value);
    if (head != nullptr) {
        node->prev = head;
        head->next = node;
    } else {
        head = node;
    }
    // TODO: release the mutex
    return reinterpret_cast<int64>(node);
  }
  bufferInfoMap *at(key_t key) {
    Node *node = reinterpret_cast<Node *>(key);
    return node->value;
  }
private: 
  Node *head;
};
};

SwapRecorder swap_recorder;

const Shape makeShape(std::string message) {
  ShapeProto proto;
  CHECK(proto.ParseFromString(message)) << "Cannot parse the proto from message string";
  return Shape(proto);
}

void GPUSwapOut(
  cudaStream_t stream, 
  void **buffers, 
  const char *opaque, size_t opaque_len) {
  std::string message(opaque, opaque_len);
  const Shape shape = makeShape(message);
  CHECK_EQ(shape, ShapeUtil::MakeShape(S32, {1024}));
  // const Shape shape = ShapeUtil::MakeShape(S32, {1024});
  auto leafShapes = ShapeUtil::GetLeafShapes(shape);
  int cnt = 0;
  bufferInfoMap *map = new bufferInfoMap;
  for (auto &leafShape : leafShapes) {
    // CHECK(!leafShape.shape.IsTuple());
    const ShapeIndex index = leafShape.index;
    int64 size = ShapeUtil::ByteSizeOf(leafShape.shape);
    void *ptr = malloc(size);
    cudaMemcpy(
      ptr, buffers[cnt++], size, 
      cudaMemcpyDeviceToHost
    );
    map->operator[](index) = ptr;
  }
  key_t key = swap_recorder.insert(map);
  cudaMemcpy(
    buffers[cnt], &key, sizeof(key_t), 
    cudaMemcpyHostToDevice
  );
}

void GPUSwapIn(
  cudaStream_t stream, 
  void **buffers, 
  const char *opaque, size_t opaque_len) {
  std::string message(opaque, opaque_len);
  const Shape shape = makeShape(message);
  auto leafShapes = ShapeUtil::GetLeafShapes(shape);

  key_t key;
  int cnt = 0;
  cudaMemcpy(
    &key, buffers[cnt++], 
    sizeof(key_t), cudaMemcpyDeviceToHost
  );
  bufferInfoMap *map = swap_recorder.at(key);
  for (auto &leafShape : leafShapes) {
    auto &index = leafShape.index;
    int64 size = ShapeUtil::ByteSizeOf(leafShape.shape);
    void *ptr = map->operator[](index);
    cudaMemcpy(
      buffers[cnt++], ptr, 
      size, cudaMemcpyHostToDevice
    );
  }
}

};  // namespace xla
#endif