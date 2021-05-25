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

#include "tensorflow_serving/servables/alphafm/alphafm_model.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/util/strings/numeric.h"
#include "tensorflow_serving/util/strings/split.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#if defined(_OPENMP)
#include "omp.h"
#endif // defined(_OPENMP)

namespace tensorflow {
namespace serving {

AlphaFmModel::AlphaFmModel() {
  LOG(INFO) << "Call the constructor AlphaFmModel().";
}

AlphaFmModel::AlphaFmModel(std::string model_path) {
  LOG(INFO) << "Call the constructor AlphaFmModel(model_path).";
  Status status = LoadModel(model_path);
  if (status.ok()) {
    LOG(INFO) << "Load the AlphaFm Model successfully.";
  } else {
    LOG(ERROR) << "Failed to load the AlphaFm Model.";
  }
}

AlphaFmModel::~AlphaFmModel() {
  LOG(INFO) << "Call the destructor ~AlphaFmModel().";
  UnloadModel();
}

Status AlphaFmModel::LoadModel(std::string model_path) {
  LOG(INFO) << "Call the method LoadModel(model_path).";
  std::ifstream infile(model_path, std::ios::in | std::ios::binary);
  if (!infile) {
    LOG(ERROR) << "Failed to open the file " + model_path;
    return errors::Unknown("Failed to open the file: " + model_path);
  }
  std::string line;
  getline(infile, line);
  if (line.empty()) {
    LOG(ERROR) << "No bias weight line found in the fm model.";
    return errors::Unknown("No bias weight line found in the fm model.");
  }
  std::vector<std::string> split_vec;
  ::SplitStringUsing(line, " ", &split_vec);
  if (split_vec.size() != 2) {
    LOG(ERROR) << "Fm model line parsing error: " << line;
    return errors::Unknown("Fm model line parsing error: " + line);
  }
  if (!::safe_strtof(split_vec[1], &bias_weight_)) {
    LOG(ERROR) << "Parse bias_weight_ failed.";
    return errors::Unknown("Parse bias_weight_ failed.");
  }
  while (!infile.eof()) {
    getline(infile, line);
    if (line.empty()) {
      LOG(WARNING) << "Empty line";
      continue;
    }
    std::vector<std::string> split_vec;
    ::SplitStringUsing(line, " ", &split_vec);
    if (split_vec.size() < 3) {
      LOG(WARNING) << "Fm model line parsing error: " << line;
      continue;
    }
    uint64_t feature_id;
    if (!::safe_strtoull(split_vec[0], &feature_id)) {
      LOG(WARNING) << "Fm model line parsing error: " << line;
      continue;
    }
    WeightIv weight_iv;
    weight_iv.exist = true;
    float weight;
    if (!::safe_strtof(split_vec[1], &weight)) {
      LOG(WARNING) << "Fm model line parsing error: " << line;
      continue;
    }
    weight_iv.weight = weight;
    for (int i = 2; i < split_vec.size(); i++) {
      if (!split_vec[i].empty()) {
        float implicit_score;
        if (!::safe_strtof(split_vec[i], &implicit_score)) {
          LOG(WARNING) << "Fm model line parsing error: " << line;
          continue;
        }
        weight_iv.implicit_vector.push_back(implicit_score);
        weight_iv.implicit_vector_sq += implicit_score * implicit_score;
      }
    }
    if (!iv_size_) {
      iv_size_ = weight_iv.implicit_vector.size();
    } else if (iv_size_ != weight_iv.implicit_vector.size()) {
      LOG(ERROR) << "Implicit_vector size error: " << line;
      return errors::Unknown("Implicit_vector size error: " + line);
    }
    feature_id_weight_iv_map_.emplace(feature_id, std::move(weight_iv));
  }
  if (feature_id_weight_iv_map_.empty()) {
    LOG(ERROR) << "Empty model file";
    return errors::Unknown("Empty model file");
  }
  return Status::OK();
}

void AlphaFmModel::UnloadModel() {
  LOG(INFO) << "Call the method UnloadModel().";
  feature_id_weight_iv_map_.clear();
}

Status AlphaFmModel::PredictTest(const FeatureScore &fm_feature_score,
                                 const FeatureScore &xgboost_feature_score,
                                 const FeatureScore &leaf_mapping_feature_score,
                                 float *result) {
  FeatureScore feature_score = xgboost_feature_score;
  feature_score.mutable_id()->MergeFrom(fm_feature_score.id());
  feature_score.mutable_score()->MergeFrom(fm_feature_score.score());
  feature_score.mutable_id()->MergeFrom(leaf_mapping_feature_score.id());
  feature_score.mutable_score()->MergeFrom(leaf_mapping_feature_score.score());
  float weight_sum = 0.0;
  weight_sum += bias_weight_;
  for (int i = 0; i < feature_score.id_size(); i++) {
    auto it = feature_id_weight_iv_map_.find(feature_score.id(i));
    if (it == feature_id_weight_iv_map_.end()) {
      continue;
    }
    weight_sum += feature_score.score(i) * (it->second).weight;
  }
  for (int i = 0; i < feature_score.id_size(); i++) {
    for (int j = i + 1; j < feature_score.id_size(); j++) {
      float tmp = 0.0;
      auto it1 = feature_id_weight_iv_map_.find(feature_score.id(i));
      auto it2 = feature_id_weight_iv_map_.find(feature_score.id(j));
      if (it1 == feature_id_weight_iv_map_.end() ||
          it2 == feature_id_weight_iv_map_.end()) {
        continue;
      }
      for (int t = 0; t < iv_size_; t++) {
        tmp += it1->second.implicit_vector[t] * it2->second.implicit_vector[t];
      }
      weight_sum += tmp * feature_score.score(i) * feature_score.score(j);
    }
  }
  *result = 1.0f / (1.0f + exp(-weight_sum));
  return Status::OK();
}

Status AlphaFmModel::Predict(const FeatureScore &fm_feature_score,
                             const FeatureScore &xgboost_feature_score,
                             const FeatureScore &leaf_mapping_feature_score,
                             float *result) {
  float weight_sum = 0.0;
  float weight_sum_cross = 0.0;
  std::vector<std::pair<float, const WeightIv *>> feature_score_weight_vector;
  feature_score_weight_vector.reserve(fm_feature_score.id_size() +
                                      xgboost_feature_score.id_size() +
                                      leaf_mapping_feature_score.id_size());
  for (int i = 0; i < fm_feature_score.id_size(); i++) {
    const WeightIv *weight_iv_ptr = nullptr;
    auto it = feature_id_weight_iv_map_.find(fm_feature_score.id(i));
    if (it == feature_id_weight_iv_map_.end()) {
      continue;
    }
    weight_iv_ptr = &(it->second);
    if (!weight_iv_ptr->exist) {
      continue;
    }
    weight_sum += fm_feature_score.score(i) * weight_iv_ptr->weight;
    feature_score_weight_vector.emplace_back(fm_feature_score.score(i),
                                             weight_iv_ptr);
    weight_sum_cross -= fm_feature_score.score(i) * fm_feature_score.score(i) *
                        weight_iv_ptr->implicit_vector_sq;
  }
  for (int i = 0; i < xgboost_feature_score.id_size(); i++) {
    const WeightIv *weight_iv_ptr = nullptr;
    auto it = feature_id_weight_iv_map_.find(xgboost_feature_score.id(i));
    if (it == feature_id_weight_iv_map_.end()) {
      continue;
    }
    weight_iv_ptr = &(it->second);
    if (!weight_iv_ptr->exist) {
      continue;
    }
    weight_sum += xgboost_feature_score.score(i) * weight_iv_ptr->weight;
    feature_score_weight_vector.emplace_back(xgboost_feature_score.score(i),
                                             weight_iv_ptr);
    weight_sum_cross -= xgboost_feature_score.score(i) *
                        xgboost_feature_score.score(i) *
                        weight_iv_ptr->implicit_vector_sq;
  }
  for (int i = 0; i < leaf_mapping_feature_score.id_size(); i++) {
    const WeightIv *weight_iv_ptr = nullptr;
    auto it = feature_id_weight_iv_map_.find(leaf_mapping_feature_score.id(i));
    if (it == feature_id_weight_iv_map_.end()) {
      continue;
    }
    weight_iv_ptr = &(it->second);
    if (!weight_iv_ptr->exist) {
      continue;
    }
    weight_sum += leaf_mapping_feature_score.score(i) * weight_iv_ptr->weight;
    feature_score_weight_vector.emplace_back(
        leaf_mapping_feature_score.score(i), weight_iv_ptr);
    weight_sum_cross -= leaf_mapping_feature_score.score(i) *
                        leaf_mapping_feature_score.score(i) *
                        weight_iv_ptr->implicit_vector_sq;
  }
  std::vector<float> sum_vec;
  sum_vec.assign(iv_size_, 0.0f);
  for (const auto &iv_pair : feature_score_weight_vector) {
    for (int i = 0; i < iv_size_; i++) {
      sum_vec[i] += iv_pair.second->implicit_vector[i] * iv_pair.first;
    }
  }
  for (int i = 0; i < iv_size_; i++) {
    weight_sum_cross += (sum_vec[i] * sum_vec[i]);
  }
  weight_sum += weight_sum_cross * 0.5 + bias_weight_;
  *result = 1.0f / (1.0f + exp(-weight_sum));
  return Status::OK();
}

Status AlphaFmModel::Predict(
    const FeatureScoreVector &feature_score_vector,
    const FeatureScoreVector &xgboost_feature_score_vector,
    const FeatureScoreVector &leaf_mapping_feature_score_vector,
    std::vector<float> &fm_result) {
  fm_result.resize(feature_score_vector.feature_score_size());
  bool cancel = false;
#pragma omp parallel shared(feature_score_vector,                              \
                            xgboost_feature_score_vector,                      \
                            leaf_mapping_feature_score_vector, fm_result,      \
                            cancel)
  {
#pragma omp for
    for (int i = 0; i < feature_score_vector.feature_score_size(); i++) {
      float result;
      if (Predict(feature_score_vector.feature_score(i),
                  xgboost_feature_score_vector.feature_score(i),
                  leaf_mapping_feature_score_vector.feature_score(i), &result)
              .ok()) {
        fm_result[i] = result;
      } else {
        LOG(ERROR) << "Failed to call the Predict method.";
        cancel = true;
      }
    }
  }
  if (cancel) {
    return errors::Unknown("Failed to call the Predict method.");
  }
  return Status::OK();
}

} // namespace serving
} // namespace tensorflow
