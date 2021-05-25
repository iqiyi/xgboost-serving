# Description: string utils.

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
    name = "numeric",
    srcs = ["numeric.cc"],
    hdrs = ["numeric.h"],
)

cc_library(
    name = "string_piece",
    srcs = ["string_piece.cc"],
    hdrs = ["string_piece.h"],
)

cc_library(
    name = "split",
    srcs = ["split.cc"],
    hdrs = ["split.h"],
    deps = [
        ":string_piece",
    ],
)
