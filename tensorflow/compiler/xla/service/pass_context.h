#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PASS_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PASS_CONTEXT_H_

#include <string>
#include <utility>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// A global context to pass arguments from python to xla passes
namespace pass_context {

class AnyObject {
 public:
  enum class Type : uint8 {
    kDouble,
    kInt,
    kString,
    kNone,
  };

  Type type;
  double double_val;
  int64 int_val;
  std::string str_val;
};

void SetPassContext(pybind11::dict dict);

void ClearPassContext();

int64 GetInt(const std::string& name, int default_value);

bool GetBool(const std::string& name, bool default_value);

std::string GetString(const std::string& name, const std::string& default_value);

}  // namespace pass_context
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PASS_CONTEXT_H_
