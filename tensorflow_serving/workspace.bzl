# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def tf_serving_workspace():
    """All TensorFlow Serving external dependencies."""

    tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

    # ===== gRPC dependencies =====
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # gRPC wants the existence of a cares dependence but its contents are not
    # actually important since we have set GRPC_ARES=0 in tools/bazel.rc
    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )

    # ===== RapidJSON (rapidjson.org) dependencies =====
    http_archive(
        name = "com_github_tencent_rapidjson",
        urls = [
            "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
        ],
        sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
        strip_prefix = "rapidjson-1.1.0",
        build_file = "@//third_party/rapidjson:BUILD",
    )

    # ===== libevent (libevent.org) dependencies =====
    http_archive(
        name = "com_github_libevent_libevent",
        urls = [
            "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip",
        ],
        sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
        strip_prefix = "libevent-release-2.1.8-stable",
        build_file = "@//third_party/libevent:BUILD",
    )

    # ===== Override TF defined `com_google_absl` (we need a recent version).
    tf_http_archive(
        name = "com_google_absl",
        build_file = str(Label("@org_tensorflow//third_party:com_google_absl.BUILD")),
        sha256 = "b6aa25c8283cca9de282bb7f5880b04492af76213b2f48c135c4963c6333a21e",
        strip_prefix = "abseil-cpp-36d37ab992038f52276ca66b9da80c1cf0f57dc2",
        urls = [
            "http://mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/36d37ab992038f52276ca66b9da80c1cf0f57dc2.tar.gz",
        ],
    )

    # ==== XGBoost dependencies ====
    http_archive(
        name = "xgboost",
        urls = [
            "https://github.com/dmlc/xgboost/releases/download/v1.5.0/xgboost.tar.gz",
        ],
        sha256 = "25ee3adb9925d0529575c0f00a55ba42202a1cdb5fdd3fb6484b4088571326a5",
        build_file = "@//third_party/xgboost:BUILD",
        strip_prefix = "xgboost",
    )
    # ==== brpc dependencies ====
    http_archive(
        name = "com_github_google_leveldb",
        build_file = "@//third_party/leveldb:BUILD",
        strip_prefix = "leveldb-a53934a3ae1244679f812d998a4f16f2c7f309a6",
        url = "https://github.com/google/leveldb/archive/a53934a3ae1244679f812d998a4f16f2c7f309a6.tar.gz"
    )
    git_repository(
        name = "brpc",
        remote = "https://github.com/apache/incubator-brpc",
        tag = "0.9.7",
    )
    git_repository(
        name = "com_github_gflags_gflags",
        remote = "https://github.com/gflags/gflags",
        tag = "v2.2.2",
    )
    git_repository(
        name = "com_github_glog_glog",
        remote = "https://github.com/google/glog",
        tag = "v0.4.0",
    )
