filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nccl",
    srcs = ["libnccl_static.a"],
    hdrs = ["nccl.h"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart_static",
    ],
)

genrule(
    name = "nccl-files",
    outs = [
        "libnccl_static.a",
        "nccl.h",
    ],
    cmd = """
cp "%{nccl_header_dir}/nccl.h" "$(@D)/nccl.h" &&
cp "%{nccl_library_dir}/libnccl_static.a" \
  "$(@D)/libnccl_static.a"
""",
)
