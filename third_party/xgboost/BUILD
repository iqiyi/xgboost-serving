# Description: XGBoost BUILD File.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)

include_files = [
    "xgboost/include/xgboost/c_api.h",
]

lib_files = [
    "xgboost/lib/libxgboost.a",
    "xgboost/lib/libdmlc.a",
]

genrule(
    name = "xgboost-srcs",
    srcs = ["@xgboost//:CMakeLists.txt"],
    outs = include_files + lib_files,
    cmd = "\n".join([
        "export INSTALL_DIR=$$(pwd)/$(@D)/xgboost",
        "export TMP_DIR=$$(mktemp -d -t xgboost.XXXXXX)",
        "mkdir -p $$TMP_DIR",
        "cp -R $$(pwd)/external/xgboost/* $$TMP_DIR",
        "cd $$TMP_DIR",
        "mkdir -p build",
        "cd build",
        "cmake .. -DBUILD_STATIC_LIB:bool=ON",
        "make -j4",
        "mkdir -p $$INSTALL_DIR/lib",
        "mkdir -p $$INSTALL_DIR/include",
        "cp -rf ../lib/libxgboost.a $$INSTALL_DIR/lib/",
        "cp dmlc-core/libdmlc.a $$INSTALL_DIR/lib/",
        "cp -rf $$TMP_DIR/include/* $$INSTALL_DIR/include/",
        "rm -rf $$TMP_DIR",
    ]),
)

cc_library(
    name = "xgboost",
    srcs = ["xgboost/lib/libxgboost.a"],
    hdrs = include_files,
    includes = ["xgboost/include"],
    linkopts = ["-lpthread", "-lgomp"],
    linkstatic = 1,
)

cc_library(
    name = "dmlc",
    srcs = ["xgboost/lib/libdmlc.a"],
    linkstatic = 1,
)
