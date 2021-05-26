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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/util/strings/numeric.h"
#include "tensorflow_serving/util/strings/split.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#if defined(_OPENMP)
#include "omp.h"
#endif // defined(_OPENMP)

namespace tensorflow {
namespace serving {

AlphafmSoftmaxModel::ftrl_model::ftrl_model() {}

AlphafmSoftmaxModel::ftrl_model::~ftrl_model() {
  if (muBias) {
    delete muBias;
    muBias = nullptr;
  }
  muMap.clear();
}

int AlphafmSoftmaxModel::ftrl_model::get_class_num() const { return class_num; }

int AlphafmSoftmaxModel::ftrl_model::get_factor_num() const {
  return factor_num;
}

std::vector<float> AlphafmSoftmaxModel::ftrl_model::getScore(
    const std::vector<std::pair<uint64_t, float>> &x) {
  std::vector<float> scoreVec(class_num);
  float denominator = 0.0;
  float maxResult = std::numeric_limits<float>::lowest();
  for (int k = 0; k < class_num; ++k) {
    float result = 0;
    result += muBias->mcu[k].wi;
    for (int i = 0; i < x.size(); ++i) {
      result += get_wi(muMap, x[i].first, k) * x[i].second;
    }
    float sum, sum_sqr, d;
    for (int f = 0; f < factor_num; ++f) {
      sum = sum_sqr = 0.0;
      for (int i = 0; i < x.size(); ++i) {
        d = get_vif(muMap, x[i].first, f, k) * x[i].second;
        sum += d;
        sum_sqr += d * d;
      }
      result += 0.5 * (sum * sum - sum_sqr);
    }
    scoreVec[k] = result;
    if (result > maxResult)
      maxResult = result;
  }
  for (int k = 0; k < class_num; ++k) {
    scoreVec[k] -= maxResult;
    scoreVec[k] = exp(scoreVec[k]);
    denominator += scoreVec[k];
  }
  for (int k = 0; k < class_num; ++k) {
    scoreVec[k] /= denominator;
  }
  return scoreVec;
}

float AlphafmSoftmaxModel::ftrl_model::get_wi(
    std::unordered_map<uint64_t, ftrl_model_unit *> &theta,
    const uint64_t &index, int classIndex) {
  std::unordered_map<uint64_t, ftrl_model_unit *>::iterator iter =
      theta.find(index);
  if (iter == theta.end()) {
    return 0.0;
  } else {
    return iter->second->mcu[classIndex].wi;
  }
}

float AlphafmSoftmaxModel::ftrl_model::get_vif(
    std::unordered_map<uint64_t, ftrl_model_unit *> &theta,
    const uint64_t &index, int f, int classIndex) {
  std::unordered_map<uint64_t, ftrl_model_unit *>::iterator iter =
      theta.find(index);
  if (iter == theta.end()) {
    return 0.0;
  } else {
    return iter->second->mcu[classIndex].vi[f];
  }
}

bool AlphafmSoftmaxModel::ftrl_model::loadModel(std::string model_path) {
  std::ifstream infile(model_path, std::ios::in | std::ios::binary);
  string line;
  if (!getline(infile, line)) {
    return false;
  }
  std::vector<string> strVec;
  ::SplitStringUsing(line, " ", &strVec);
  if (0 != ((strVec.size() - 1) % 3)) {
    return false;
  }
  class_num = (strVec.size() - 1) / 3;
  muBias = new ftrl_model_unit(class_num, 0, strVec);
  while (getline(infile, line)) {
    strVec.clear();
    ::SplitStringUsing(line, " ", &strVec);
    factor_num = ((strVec.size() - 1) / class_num - 3) / 3;
    string &index = strVec[0];
    uint64_t index_id;
    if (!::safe_strtoull(index, &index_id)) {
      return false;
    }
    ftrl_model_unit *pMU = new ftrl_model_unit(class_num, factor_num, strVec);
    muMap[index_id] = pMU;
  }
  return true;
}

AlphafmSoftmaxModel::AlphafmSoftmaxModel() {
  LOG(INFO) << "Call the constructor AlphafmSoftmaxModel().";
  ftrl_model_.reset(new AlphafmSoftmaxModel::ftrl_model());
}

AlphafmSoftmaxModel::AlphafmSoftmaxModel(std::string model_path) {
  LOG(INFO) << "Call the constructor AlphafmSoftmaxModel(model_path).";
  ftrl_model_.reset(new AlphafmSoftmaxModel::ftrl_model());
  Status status = LoadModel(model_path);
  if (status.ok()) {
    LOG(INFO) << "Load the AlphafmSoftmax Model successfully.";
  } else {
    LOG(ERROR) << "Failed to load the AlphafmSoftmax Model.";
  }
}

int AlphafmSoftmaxModel::GetClassNum() const {
  return ftrl_model_->get_class_num();
}

int AlphafmSoftmaxModel::GetFactorNum() const {
  return ftrl_model_->get_factor_num();
}

AlphafmSoftmaxModel::~AlphafmSoftmaxModel() {
  LOG(INFO) << "Call the destructor ~AlphafmSoftmaxModel().";
  UnloadModel();
}

Status AlphafmSoftmaxModel::LoadModel(std::string model_path) {
  LOG(INFO) << "Call the method LoadModel(model_path).";
  bool status = ftrl_model_->loadModel(model_path);
  return (true == status)
             ? Status::OK()
             : errors::Unknown("Failed to LoadModel: " + model_path);
}

void AlphafmSoftmaxModel::UnloadModel() {
  LOG(INFO) << "Call the method UnloadModel().";
  ftrl_model_.reset(nullptr);
}

Status
AlphafmSoftmaxModel::Predict(const FeatureScore &fm_feature_score,
                             const FeatureScore &xgboost_feature_score,
                             const FeatureScore &leaf_mapping_feature_score,
                             std::vector<float> &result) {
  std::vector<std::pair<uint64_t, float>> inputs;
  for (int i = 0; i < fm_feature_score.id_size(); i++) {
    inputs.push_back({fm_feature_score.id(i), fm_feature_score.score(i)});
  }
  for (int i = 0; i < xgboost_feature_score.id_size(); i++) {
    inputs.push_back(
        {xgboost_feature_score.id(i), xgboost_feature_score.score(i)});
  }
  for (int i = 0; i < leaf_mapping_feature_score.id_size(); i++) {
    inputs.push_back({leaf_mapping_feature_score.id(i),
                      leaf_mapping_feature_score.score(i)});
  }
  result = ftrl_model_->getScore(inputs);
  return Status::OK();
}

Status AlphafmSoftmaxModel::Predict(
    const FeatureScoreVector &fm_feature_score_vector,
    const FeatureScoreVector &xgboost_feature_score_vector,
    const FeatureScoreVector &leaf_mapping_feature_score_vector,
    std::vector<std::vector<float>> &fm_result) {
  for (int i = 0; i < fm_feature_score_vector.feature_score_size(); i++) {
    std::vector<float> result;
    if (Predict(fm_feature_score_vector.feature_score(i),
                xgboost_feature_score_vector.feature_score(i),
                leaf_mapping_feature_score_vector.feature_score(i), result)
            .ok()) {
      fm_result.push_back(result);
    } else {
      LOG(ERROR) << "Failed to call the Predict method.";
      return errors::Unknown("Failed to call the Predict method.");
    }
  }
  return Status::OK();
}

} // namespace serving
} // namespace tensorflow
