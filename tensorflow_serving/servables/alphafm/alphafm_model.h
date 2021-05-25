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

#ifndef TENSORFLOW_SERVING_SERVABLES_ALPHAFM_ALPHAFM_MODEL_H_
#define TENSORFLOW_SERVING_SERVABLES_ALPHAFM_ALPHAFM_MODEL_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/predict.pb.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tensorflow {
namespace serving {
class AlphaFmModel {
public:
  AlphaFmModel();
  AlphaFmModel(std::string model_path);
  Status LoadModel(std::string model_path);
  void UnloadModel();
  Status Predict(const FeatureScoreVector &fm_feature_score_vector,
                 const FeatureScoreVector &xgboost_feature_score_vector,
                 const FeatureScoreVector &leaf_mapping_feature_score_vector,
                 std::vector<float> &fm_result);
  ~AlphaFmModel();

private:
  struct WeightIv {
    float weight;
    std::vector<float> implicit_vector;
    float implicit_vector_sq{0.0};
    bool exist{false};
  };
  std::unordered_map<uint64_t, WeightIv> feature_id_weight_iv_map_;
  float bias_weight_;
  uint32_t iv_size_{0};
  Status Predict(const FeatureScore &fm_feature_score,
                 const FeatureScore &xgboost_feature_score,
                 const FeatureScore &leaf_mapping_feature_score, float *result);
  Status PredictTest(const FeatureScore &fm_feature_score,
                     const FeatureScore &xgboost_feature_score,
                     const FeatureScore &leaf_mapping_feature_score,
                     float *result);
};
} // namespace serving
} // namespace tensorflow
#endif // TENSORFLOW_SERVING_SERVABLES_ALPHAFM_ALPHAFM_MODEL_H_
