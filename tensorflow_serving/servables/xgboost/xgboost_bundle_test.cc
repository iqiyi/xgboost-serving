/*

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

#include "tensorflow_serving/servables/xgboost/xgboost_bundle.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {
TEST(XgboostBundleTest, Basic) {
  XgboostBundle bundle;
  TF_ASSERT_OK(bundle.LoadXgboostModel(
      "tensorflow_serving/servables/xgboost/testdata/test_model/1"));
}
} // namespace
} // namespace serving
} // namespace tensorflow
