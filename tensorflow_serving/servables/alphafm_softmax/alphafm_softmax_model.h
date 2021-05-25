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

#ifndef TENSORFLOW_SERVING_SERVABLES_ALPHAFM_SOFTMAX_ALPHAFM_SOFTMAX_MODEL_H_
#define TENSORFLOW_SERVING_SERVABLES_ALPHAFM_SOFTMAX_ALPHAFM_SOFTMAX_MODEL_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_serving/apis/predict.pb.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tensorflow {
namespace serving {
class AlphafmSoftmaxModel {
public:
  AlphafmSoftmaxModel();
  AlphafmSoftmaxModel(std::string model_path);
  int GetClassNum() const;
  int GetFactorNum() const;
  Status LoadModel(std::string model_path);
  void UnloadModel();
  Status Predict(const FeatureScoreVector &fm_feature_score_vector,
                 const FeatureScoreVector &xgboost_feature_score_vector,
                 const FeatureScoreVector &leaf_mapping_feature_score_vector,
                 std::vector<std::vector<float>> &fm_result);
  ~AlphafmSoftmaxModel();

private:
  //每一个特征维度的每一个类别的模型单元
  struct ftrl_model_class_unit {
    float wi;
    float w_ni;
    float w_zi;
    std::vector<float> vi;
    std::vector<float> v_ni;
    std::vector<float> v_zi;

    ftrl_model_class_unit(int factor_num,
                          const std::vector<string> &modelLineSeg, int start) {
      vi.resize(factor_num);
      v_ni.resize(factor_num);
      v_zi.resize(factor_num);
      wi = stod(modelLineSeg[start + 1]);
      w_ni = stod(modelLineSeg[start + 2 + factor_num]);
      w_zi = stod(modelLineSeg[start + 3 + factor_num]);
      for (int f = 0; f < factor_num; ++f) {
        vi[f] = stod(modelLineSeg[start + 2 + f]);
        v_ni[f] = stod(modelLineSeg[start + 4 + factor_num + f]);
        v_zi[f] = stod(modelLineSeg[start + 4 + 2 * factor_num + f]);
      }
    }
  };

  //每一个特征维度的模型单元
  struct ftrl_model_unit {
    int class_num;
    std::vector<ftrl_model_class_unit> mcu;
    ftrl_model_unit(int cn, int factor_num,
                    const std::vector<string> &modelLineSeg) {
      class_num = cn;
      for (int i = 0; i < class_num; ++i) {
        int start = i * (3 + 3 * factor_num);
        mcu.push_back(ftrl_model_class_unit(factor_num, modelLineSeg, start));
      }
    }
  };

  struct ftrl_model {
    ftrl_model_unit *muBias;
    std::unordered_map<uint64_t, ftrl_model_unit *> muMap;
    int class_num;
    int factor_num;

    ftrl_model();
    ~ftrl_model();
    std::vector<float>
    getScore(const std::vector<std::pair<uint64_t, float>> &x);
    bool loadModel(std::string model_path);

    int get_class_num() const;
    int get_factor_num() const;
    float get_wi(std::unordered_map<uint64_t, ftrl_model_unit *> &theta,
                 const uint64_t &index, int classIndex);
    float get_vif(std::unordered_map<uint64_t, ftrl_model_unit *> &theta,
                  const uint64_t &index, int f, int classIndex);
  };
  std::unique_ptr<ftrl_model> ftrl_model_;

  Status Predict(const FeatureScore &fm_feature_score,
                 const FeatureScore &xgboost_feature_score,
                 const FeatureScore &leaf_mapping_feature_score,
                 std::vector<float> &result);
};
} // namespace serving
} // namespace tensorflow
#endif // TENSORFLOW_SERVING_SERVABLES_ALPHAFM_SOFTMAX_ALPHAFM_SOFTMAX_MODEL_H_
