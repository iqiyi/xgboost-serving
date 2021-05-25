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

#ifndef TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_UTIL_H_
#define TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow_serving/util/optional.h"

namespace tensorflow {
namespace serving {

// Implementation details mainly used for testing; please don't depend on it.
namespace internal {

monitoring::Sampler<1> *GetExampleCounts();

monitoring::Counter<1> *GetExampleCountTotal();

} // namespace internal

// Records the example count of this request with the metric tracking the
// histogram of number of examples per request.
void RecordRequestExampleCount(const string &model_name, size_t count);

// Populates given model_spec based on the model name and optional
// signature/version information.
// If signature_name has a value and is empty, model_spec's signature_name is
// set to tensorflow::kDefaultServingSignatureDefKey.
void MakeModelSpec(const string &model_name,
                   const optional<string> &signature_name,
                   const optional<int64> &version, ModelSpec *model_spec);

} // namespace serving
} // namespace tensorflow

#endif // TENSORFLOW_SERVING_SERVABLES_TENSORFLOW_UTIL_H_
