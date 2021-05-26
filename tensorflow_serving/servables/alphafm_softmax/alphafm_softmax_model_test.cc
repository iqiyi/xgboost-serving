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

#include "tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_model.h"

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
TEST(AlphafmSoftmaxModelTest, Basic) {
  AlphafmSoftmaxModel alphafm_softmax_model("tensorflow_serving/servables/"
                                            "alphafm_softmax/testdata/"
                                            "test_model/1/deploy.fm");
  FeatureScoreVector xgboost_feature_score_vector;
  FeatureScoreVector fm_feature_score_vector;
  FeatureScoreVector leaf_mapping_feature_score_vector;
  auto xgboost_feature_score = xgboost_feature_score_vector.add_feature_score();
  auto xgboost_feature_score_2 =
      xgboost_feature_score_vector.add_feature_score();
  auto fm_feature_score = fm_feature_score_vector.add_feature_score();
  auto fm_feature_score_2 = fm_feature_score_vector.add_feature_score();
  auto leaf_mapping_feature_score =
      leaf_mapping_feature_score_vector.add_feature_score();
  auto leaf_mapping_feature_score_2 =
      leaf_mapping_feature_score_vector.add_feature_score();
  std::vector<uint64_t> xgboost_id{104, 110, 112, 114, 116,
                                   117, 119, 120, 122, 124};
  std::vector<float> xgboost_score{1,        1, 0.439716, 0.646667, 0.555556,
                                   0.727273, 1, 0.883929, 1,        0.64390756};
  std::vector<uint64_t> fm_id{359490963259392057, 1961623260467125119,
                              2748734556781667681, 4310738850516811781};
  std::vector<float> fm_score{1.0, 1.0, 1.0, 1.0};
  std::vector<uint64_t> leaf_mapping_id{
      4119, 4155, 4171, 4221, 4235, 4283, 4297, 4325, 4374, 4400, 4439,
      4476, 4497, 4540, 4568, 4606, 4611, 4667, 4702, 4724, 4746, 4779,
      4824, 4861, 4880, 4910, 4925, 4979, 5017, 5019, 5080, 5097, 5129,
      5167, 5189, 5217, 5254, 5293, 5326, 5357, 5378, 5399, 5440, 5468,
      5520, 5538, 5589, 5620, 5623, 5675, 5717, 5741, 5761, 5806, 5810,
      5842, 5876, 5937, 5956, 6000, 6008, 6054, 6070, 6096, 6137, 6185,
      6193, 6253, 6280, 6306, 6345, 6374, 6393, 6414, 6431, 6463, 6493,
      6551, 6561, 6596, 6636, 6662, 6705, 6733, 6755, 6797, 6804, 6849,
      6882, 6898, 6926, 6971, 7005, 7023, 7039, 7075, 7124, 7156, 7173,
      7217, 7247, 7255, 7287, 7321, 7380, 7389, 7421, 7476, 7482, 7536};
  std::vector<float> leaf_mapping_score(110, 1.0);
  *(xgboost_feature_score->mutable_id()) = {xgboost_id.begin(),
                                            xgboost_id.end()};
  *(xgboost_feature_score->mutable_score()) = {xgboost_score.begin(),
                                               xgboost_score.end()};
  *(xgboost_feature_score_2->mutable_id()) = {xgboost_id.begin(),
                                              xgboost_id.end()};
  *(xgboost_feature_score_2->mutable_score()) = {xgboost_score.begin(),
                                                 xgboost_score.end()};
  *(fm_feature_score->mutable_id()) = {fm_id.begin(), fm_id.end()};
  *(fm_feature_score->mutable_score()) = {fm_score.begin(), fm_score.end()};
  *(fm_feature_score_2->mutable_id()) = {fm_id.begin(), fm_id.end()};
  *(fm_feature_score_2->mutable_score()) = {fm_score.begin(), fm_score.end()};
  *(leaf_mapping_feature_score->mutable_id()) = {leaf_mapping_id.begin(),
                                                 leaf_mapping_id.end()};
  *(leaf_mapping_feature_score->mutable_score()) = {leaf_mapping_score.begin(),
                                                    leaf_mapping_score.end()};
  *(leaf_mapping_feature_score_2->mutable_id()) = {leaf_mapping_id.begin(),
                                                   leaf_mapping_id.end()};
  *(leaf_mapping_feature_score_2->mutable_score()) = {
      leaf_mapping_score.begin(), leaf_mapping_score.end()};
  std::vector<std::vector<float>> result;
  TF_ASSERT_OK(alphafm_softmax_model.Predict(
      fm_feature_score_vector, xgboost_feature_score_vector,
      leaf_mapping_feature_score_vector, result));
  EXPECT_THAT(std::to_string(result[0][0]).substr(0, 6), "0.1879");
}
} // namespace
} // namespace serving
} // namespace tensorflow
