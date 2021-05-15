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
      obj.int_val = py::cast<int64>(item.second);
    } else if (py::isinstance<py::float_>(item.second)) {
      obj.type = AnyObject::Type::kDouble;
      obj.double_val = py::cast<double>(item.second);
    } else if (py::isinstance<py::str>(item.second)) {
      obj.type = AnyObject::Type::kString;
      obj.str_val = py::cast<std::string>(item.second);
    } else if (py::isinstance<py::list>(item.second) ||
               py::isinstance<py::tuple>(item.second)) {
      auto tuple_val = py::cast<py::tuple>(item.second);
      if (py::isinstance<py::int_>(tuple_val[0])) {
        obj.type = AnyObject::Type::kIntVector;
        obj.int_vector_val.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          obj.int_vector_val.push_back(py::cast<int64>(tuple_val[i]));
        }
      } else if (py::isinstance<py::float_>(tuple_val[0])) {
        obj.type = AnyObject::Type::kDoubleVector;
        obj.double_vector_val.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          obj.double_vector_val.push_back(py::cast<double>(tuple_val[i]));
        }
      } else {
        LOG(FATAL) << "Invalid value in a tuple/list: "
                   << py::str(item.second);
      }
    } else {
      LOG(FATAL) << "Invalid value: " << py::str(item.second);
    }

    current_context[name] = obj;
  }
}

void ClearPassContext() {
  current_context.clear();
}

int64 GetInt(const std::string& name, int64 default_value) {
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
        LOG(FATAL) << "Get value of '" << name << "' with an invalid type";
    }
  }
}

double GetDouble(const std::string& name) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    LOG(FATAL) << "Cannot find " << name << " in the pass context";
  } else {
    const AnyObject& obj = iter->second;
    switch (obj.type) {
      case AnyObject::Type::kDouble:
        return obj.double_val;
      case AnyObject::Type::kInt:
        return static_cast<double>(obj.int_val);
      default:
        LOG(FATAL) << "Get value of '" << name << "' with an invalid type";
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
        LOG(FATAL) << "Get value of '" << name << "' with an invalid type";
    }
  }
}

std::vector<int64> GetIntVector(const std::string& name) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    LOG(FATAL) << "Cannot find " << name << " in the pass context";
  } else {
    const AnyObject& obj = iter->second;
    switch (obj.type) {
      case AnyObject::Type::kIntVector:
        return obj.int_vector_val;
      default:
        LOG(FATAL) << "Get value of '" << name << "' with an invalid type";
    }
  }
}

std::vector<double> GetDoubleVector(const std::string& name) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    LOG(FATAL) << "Cannot find " << name << " in the pass context";
  } else {
    const AnyObject& obj = iter->second;
    switch (obj.type) {
      case AnyObject::Type::kDoubleVector:
        return obj.double_vector_val;
      default:
        LOG(FATAL) << "Get value of '" << name << "' with an invalid type";
    }
  }
}

}  // namespace pass_context
}  // namespace xla

