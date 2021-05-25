# Description: Example xgboost model export.

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
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

py_runtime(
    name = "xgboost_python3_runtime",
    interpreter_path = "/usr/local/bin/python3.7",
    files = [],
)

py_binary(
    name = "export_test_model",
    srcs = [
        "export_test_model.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
    ],
    data = [
        "agaricus.txt.train",
	"agaricus.txt.test",
    ],
)

