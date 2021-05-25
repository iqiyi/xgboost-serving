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

#include "tensorflow_serving/servables/alphafm/feature_mapping.h"

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
TEST(FeatureMappingTest, Basic) {
  FeatureMapping feature_mapping("tensorflow_serving/servables/alphafm/"
                                 "testdata/test_model/1/deploy.leaf_mapping");
  uint32_t tree_num = 0;
  uint32_t leaf_index = 31;
  uint64_t feature_id;
  Status status =
      feature_mapping.GetFeatureId(tree_num, leaf_index, &feature_id);
  EXPECT_EQ(feature_id, 4097);
}
} // namespace
} // namespace serving
} // namespace tensorflow
