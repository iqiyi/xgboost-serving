# Description: TensorFlow Serving alphafm servable.

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
        ],
    ),
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "alphafm_source_adapter_proto",
    srcs = ["alphafm_source_adapter.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
    ],
)

cc_library(
    name = "alphafm_model",
    srcs = ["alphafm_model.cc"],
    hdrs = ["alphafm_model.h"],
    visibility = [
	"//visibility:public",
    ],
    deps = [
	"//tensorflow_serving/apis:predict_proto",
	"//tensorflow_serving/util/strings:numeric",
	"//tensorflow_serving/util/strings:split",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    copts = [
        "-fopenmp",
    ],
    linkopts = ["-lgomp"],
)

cc_library(
    name = "feature_mapping",
    srcs = ["feature_mapping.cc"],
    hdrs = ["feature_mapping.h"],
    visibility = [
	"//visibility:public",
    ],
    deps = [
	"//tensorflow_serving/util/strings:numeric",
	"//tensorflow_serving/util/strings:split",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "alphafm_bundle",
    srcs = ["alphafm_bundle.cc"],
    hdrs = ["alphafm_bundle.h", "//tensorflow_serving/servables/xgboost:xgboost_constants.h"],
    visibility = [
	"//visibility:public",
    ],
    deps = [
        "@xgboost//:xgboost",
        "@xgboost//:dmlc",
        ":alphafm_model",
        ":feature_mapping",
        "//tensorflow_serving/util:file_probing_env",
    ],
)

cc_library(
    name = "alphafm_source_adapter",
    srcs = ["alphafm_source_adapter.cc"],
    hdrs = ["alphafm_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@xgboost//:xgboost",
	":alphafm_bundle",
	":alphafm_source_adapter_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
    ],
    alwayslink = 1,
)

cc_library(
    name = "predict_impl",
    srcs = ["predict_impl.cc"],
    hdrs = ["predict_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
	":alphafm_bundle",
	"//tensorflow_serving/apis:predict_proto",
	"//tensorflow_serving/servables/xgboost:util",
	"//tensorflow_serving/model_servers:server_core",
	"@brpc//:bvar",
    ]
)

cc_test(
    name = "alphafm_model_test",
    size = "medium",
    srcs = ["alphafm_model_test.cc"],
    data = ["//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.fm"],
    deps = [
        ":alphafm_model",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "feature_mapping_test",
    size = "medium",
    srcs = ["feature_mapping_test.cc"],
    data = ["//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.leaf_mapping"],
    deps = [
        ":feature_mapping",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "alphafm_bundle_test",
    size = "medium",
    srcs = ["alphafm_bundle_test.cc"],
    data = [
                "//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.model",
                "//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.fm",
                "//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.leaf_mapping",
	   ],
    deps = [
        ":alphafm_bundle",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "alphafm_source_adapter_test",
    size = "medium",
    srcs = ["alphafm_source_adapter_test.cc"],
    deps = [
        ":alphafm_source_adapter",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:test",
	"//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
	"//tensorflow_serving/core:aspired_versions_manager",
	"//tensorflow_serving/core:availability_preserving_policy",
    ],
)

cc_test(
    name = "predict_impl_test",
    size = "medium",
    srcs = ["predict_impl_test.cc"],
    data = [
        "//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.model",
        "//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.fm",
        "//tensorflow_serving/servables/alphafm/testdata:test_model/1/deploy.leaf_mapping",
    ],
    deps = [
        ":predict_impl",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
