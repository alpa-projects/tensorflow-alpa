#include "tensorflow/compiler/xla/service/pass_context.h"

#include "absl/container/flat_hash_map.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace pass_context {

namespace py = pybind11;

absl::flat_hash_map<std::string, AnyObject> current_context;

void SetPassContext(py::dict dict) {
  for (auto item : dict) {
    std::string name = py::str(item.first);
    AnyObject obj;

    if (py::isinstance<py::int_>(item.second)) {
      obj.type = AnyObject::Type::kInt;
      obj.int_val = py::cast<int>(item.second);
    } else if (py::isinstance<py::float_>(item.second)) {
      obj.type = AnyObject::Type::kDouble;
      obj.double_val = py::cast<double>(item.second);
    } else if (py::isinstance<py::str>(item.second)) {
      obj.type = AnyObject::Type::kString;
      obj.str_val = py::cast<std::string>(item.second);
    } else {
      LOG(FATAL) << "Invalid value: " << py::str(item.second);
    }

    current_context[name] = obj;
  }
}

void ClearPassContext() {
  current_context.clear();
}

int64 GetInt(const std::string& name, int default_value) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    return default_value;
  } else {
    const AnyObject& obj = iter->second;
    switch (obj.type) {
      case AnyObject::Type::kDouble:
        return static_cast<int64>(obj.double_val);
      case AnyObject::Type::kInt:
        return obj.int_val;
      default:
        LOG(FATAL) << "Get value with an invalid type: " << name;
    }
  }
}

bool GetBool(const std::string& name, bool default_value) {
  return static_cast<bool>(GetInt(name, default_value));
}

std::string GetString(const std::string& name, const std::string& default_value) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    return default_value;
  } else {
    const AnyObject& obj = iter->second;
    switch (obj.type) {
      case AnyObject::Type::kString:
        return obj.str_val;
      default:
        LOG(FATAL) << "Get value with an invalid type: " << name;
    }
  }
}

}  // namespace pass_context
}  // namespace xla

