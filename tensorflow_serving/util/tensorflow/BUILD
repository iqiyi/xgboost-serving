# Description: tensorflow utils.

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "**/google_*",
        ],
    ),
)

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "periodic_function_dynamic",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    deps = [
        # "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

