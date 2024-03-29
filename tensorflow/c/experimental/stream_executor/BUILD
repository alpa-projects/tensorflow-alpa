# Description:
# StreamExecutor C API.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    licenses = ["notice"],
)

filegroup(
    name = "headers",
    srcs = [
        "stream_executor.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "stream_executor_hdrs",
    hdrs = ["stream_executor.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "stream_executor",
    srcs = ["stream_executor.cc"],
    hdrs = ["stream_executor.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":stream_executor_internal",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla/stream_executor:executor_cache",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:platform",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_internal",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl",
        "//tensorflow/compiler/xla/stream_executor:timer",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime/device:device_utils",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:strcat",
    ],
)

cc_library(
    name = "stream_executor_internal",
    hdrs = [
        "stream_executor.h",
        "stream_executor_internal.h",
    ],
    visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/core/common_runtime/pluggable_device:__subpackages__",
    ],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla/stream_executor:executor_cache",
        "//tensorflow/compiler/xla/stream_executor/lib",
    ],
)

tf_cc_test(
    name = "stream_executor_test",
    srcs = ["stream_executor_test.cc"],
    deps = [
        ":stream_executor",
        ":stream_executor_internal",
        ":stream_executor_test_util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "stream_executor_test_util",
    srcs = ["stream_executor_test_util.cc"],
    hdrs = ["stream_executor_test_util.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":stream_executor_hdrs",
        "//tensorflow/c:tf_status",
    ],
)
