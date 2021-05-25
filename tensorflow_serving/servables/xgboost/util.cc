/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/servables/xgboost/util.h"

#include "google/protobuf/wrappers.pb.h"
// #include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {
namespace {

auto *example_counts = monitoring::Sampler<1>::New(
    {"/tensorflow/serving/request_example_counts",
     "The number of tensorflow.Examples per request.", "model"},
    // It's 15 buckets with the last bucket being 2^14 to DBL_MAX;
    // so the limits are [1, 2, 4, 8, ..., 16 * 1024, DBL_MAX].
    monitoring::Buckets::Exponential(1, 2, 15));

auto *example_count_total = monitoring::Counter<1>::New(
    "/tensorflow/serving/request_example_count_total",
    "The total number of tensorflow.Examples.", "model");
} // namespace

namespace internal {

monitoring::Sampler<1> *GetExampleCounts() { return example_counts; }

monitoring::Counter<1> *GetExampleCountTotal() { return example_count_total; }

} // namespace internal

void RecordRequestExampleCount(const string &model_name, size_t count) {
  example_counts->GetCell(model_name)->Add(count);
  example_count_total->GetCell(model_name)->IncrementBy(count);
}

void MakeModelSpec(const string &model_name,
                   const optional<string> &signature_name,
                   const optional<int64> &version, ModelSpec *model_spec) {
  model_spec->Clear();
  model_spec->set_name(model_name);
  if (version) {
    model_spec->mutable_version()->set_value(*version);
  }
}

} // namespace serving
} // namespace tensorflow
