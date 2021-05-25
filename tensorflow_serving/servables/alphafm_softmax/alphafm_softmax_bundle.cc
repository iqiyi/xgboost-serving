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

#include "tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_bundle.h"

#include "tensorflow_serving/servables/xgboost/xgboost_constants.h"

#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

AlphafmSoftmaxBundle::AlphafmSoftmaxBundle() {
  LOG(INFO) << "Call the constructor AlphafmSoftmaxBundle().";
}

AlphafmSoftmaxBundle::AlphafmSoftmaxBundle(std::string model_path) {
  LOG(INFO) << "Call the constructor AlphafmSoftmaxBundle(string)";
  Status status = LoadAlphafmSoftmaxModel(model_path);
  if (status.ok()) {
    LOG(INFO) << "Load the alphafm_softmax model successfully.";
  } else {
    LOG(ERROR) << status.error_message();
  }
}

Status AlphafmSoftmaxBundle::LoadAlphafmSoftmaxModel(std::string model_path) {
  // Load The Xgboost Model
  std::string xgboost_model_path = model_path + "/" + kXGBoostModelFileName;
  if (!Env::Default()->FileExists(xgboost_model_path).ok()) {
    return errors::Unknown("Xgboost Model Path is empty: " +
                           xgboost_model_path);
  }
  int status = XGBoosterCreate(NULL, 0, &xgbooster_);
  if (0 != status) {
    return errors::Unknown("Failed to call XGBoosterCreate: ",
                           XGBGetLastError());
  }
  status = XGBoosterLoadModel(xgbooster_, xgboost_model_path.c_str());
  if (0 != status) {
    return errors::Unknown("Failed to call XGBoosterLoadModel: ",
                           XGBGetLastError());
  }
  // Load The alphaFM_softmax Model
  std::string fm_model_path = model_path + "/" + kFmModelFileName;
  if (!Env::Default()->FileExists(fm_model_path).ok()) {
    return errors::Unknown("FM Model Path is empty: " + fm_model_path);
  }
  alphafm_softmax_model_.reset(new AlphafmSoftmaxModel());
  Status alphafm_softmax_status =
      alphafm_softmax_model_->LoadModel(fm_model_path);
  if (!alphafm_softmax_status.ok()) {
    LOG(ERROR) << "Failed to load the AlphafmSoftmaxModel model.";
    return errors::Unknown("Failed to load the AlphafmSoftmaxModel model.");
  }
  // Load The Feature Mapping
  std::string feature_mapping_path = model_path + "/" + kFeatureMappingFileName;
  if (!Env::Default()->FileExists(feature_mapping_path).ok()) {
    return errors::Unknown("Feature Mapping Path is empty: " +
                           feature_mapping_path);
  }
  feature_mapping_.reset(new FeatureMapping());
  Status feature_mapping_status =
      feature_mapping_->LoadModel(feature_mapping_path);
  if (!feature_mapping_status.ok()) {
    LOG(ERROR) << "Failed to load the FeatureMapping model.";
    return errors::Unknown("Failed to load the FeatureMapping model.");
  }
  return Status::OK();
}

Status AlphafmSoftmaxBundle::UnloadXgboostModel() {
  int status = XGBoosterFree(xgbooster_);
  xgbooster_ = nullptr;
  return (0 == status) ? Status::OK()
                       : errors::Unknown("Failed to unload the xgboost model: ",
                                         XGBGetLastError());
}

BoosterHandle AlphafmSoftmaxBundle::GetBoosterHandle() const {
  return xgbooster_;
}

FeatureMapping *AlphafmSoftmaxBundle::GetFeatureMappingHandle() const {
  return feature_mapping_.get();
}
AlphafmSoftmaxModel *
AlphafmSoftmaxBundle::GetAlphafmSoftmaxModelHandle() const {
  return alphafm_softmax_model_.get();
}

AlphafmSoftmaxBundle::~AlphafmSoftmaxBundle() {
  // Unload The Xgboost Model
  LOG(INFO) << "Call the destructor ~AlphafmSoftmaxBundle().";
  Status status = UnloadXgboostModel();
  if (status.ok()) {
    LOG(INFO) << "Unload the xgboost model successfully.";
  } else {
    LOG(ERROR) << status.error_message();
  }
  // Unload The alphaFM_softmax Model
  if (alphafm_softmax_model_) {
    alphafm_softmax_model_.reset(nullptr);
  }
  // Unload The Feature Mapping
  if (feature_mapping_) {
    feature_mapping_.reset(nullptr);
  }
}

} // namespace serving
} // namespace tensorflow
