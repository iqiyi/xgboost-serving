# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_binary(
    name = "xgboost_client_cc",
    srcs = [
        "xgboost_client.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
        "@com_google_protobuf//:protobuf_lite",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "alphafm_client_cc",
    srcs = [
        "alphafm_client.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
        "@com_google_protobuf//:protobuf_lite",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "alphafm_softmax_client_cc",
    srcs = [
        "alphafm_softmax_client.cc",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto",
        "@com_google_protobuf//:protobuf_lite",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
