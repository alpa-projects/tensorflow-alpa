#include "tensorflow/compiler/xla/service/pass_context.h"

#include "absl/container/flat_hash_map.h"
#include "absl/types/any.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace pass_context {

namespace py = pybind11;

absl::flat_hash_map<std::string, absl::any> current_context;

void SetPassContext(py::dict dict) {
  for (auto item : dict) {
    std::string name = py::str(item.first);
    absl::any obj;

    if (py::isinstance<py::bool_>(item.second)) {
      obj = absl::any(py::cast<bool>(item.second));
    } else if (py::isinstance<py::int_>(item.second)) {
      obj = absl::any(py::cast<int64>(item.second));
    } else if (py::isinstance<py::float_>(item.second)) {
      obj = absl::any(py::cast<double>(item.second));
    } else if (py::isinstance<py::str>(item.second)) {
      obj = absl::any(py::cast<std::string>(item.second));
    } else if (py::isinstance<py::list>(item.second) ||
               py::isinstance<py::tuple>(item.second)) {
      auto tuple_val = py::cast<py::tuple>(item.second);
      // Infer the type according to the first element of the tuple.
      if (!tuple_val.empty() &&  py::isinstance<py::int_>(tuple_val[0])) {
        std::vector<int64> int_vector;
        int_vector.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          int_vector.push_back(py::cast<int64>(tuple_val[i]));
        }
        obj = absl::any(std::move(int_vector));
      } else if (!tuple_val.empty() && py::isinstance<py::float_>(tuple_val[0])) {
        std::vector<double> double_vector;
        double_vector.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          double_vector.push_back(py::cast<double>(tuple_val[i]));
        }
        obj = absl::any(std::move(double_vector));
      } else if (!tuple_val.empty() && py::isinstance<py::str>(tuple_val[0])) {
        std::vector<std::string> str_vector;
        str_vector.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          str_vector.push_back(py::cast<std::string>(tuple_val[i]));
        }
        obj = absl::any(std::move(str_vector));
      } else {
        obj = absl::any(py::cast<py::object>(item.second));
      }
    } else {
      obj = absl::any(py::cast<py::object>(item.second));
    }

    current_context[name] = std::move(obj);
  }
}

void ClearPassContext() {
  current_context.clear();
}

template <typename T>
T GetWithDefaultValue(const std::string& name, const T& default_value) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    return default_value;
  } else {
    try {
      return absl::any_cast<T>(iter->second);
    } catch(const absl::bad_any_cast& e) {
      LOG(FATAL) << "Bad cast of '" << name;
    }
  }
}

template <typename T>
T GetWithoutDefaultValue(const std::string& name) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    LOG(FATAL) << "Cannot find " << name << " in the pass context";
  } else {
    try {
      return absl::any_cast<T>(iter->second);
    } catch(const absl::bad_any_cast& e) {
      LOG(FATAL) << "Bad cast of '" << name;
    }
  }
}

int64 GetInt(const std::string& name, int64 default_value) {
  return GetWithDefaultValue<int64>(name, default_value);
}

bool GetBool(const std::string& name, bool default_value) {
  return GetWithDefaultValue<bool>(name, default_value);
}

double GetDouble(const std::string& name) {
  return GetWithoutDefaultValue<double>(name);
}

std::string GetString(const std::string& name, const std::string& default_value) {
  return GetWithDefaultValue<std::string>(name, default_value);
}

std::vector<int64> GetIntVector(const std::string& name) {
  return GetWithoutDefaultValue<std::vector<int64>>(name);
}

std::vector<double> GetDoubleVector(const std::string& name) {
  return GetWithoutDefaultValue<std::vector<double>>(name);
}

std::vector<std::string> GetStringVector(const std::string& name) {
  return GetWithoutDefaultValue<std::vector<std::string>>(name);
}

py::object GetPyObject(const std::string& name) {
  return GetWithoutDefaultValue<py::object>(name);
}

}  // namespace pass_context
}  // namespace xla
