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

#include "tensorflow_serving/servables/xgboost/xgboost_constants.h"

#include "tensorflow_serving/util/file_probing_env.h"

namespace tensorflow {
namespace serving {

XgboostBundle::XgboostBundle() {
  LOG(INFO) << "Call the constructor XgboostBundle().";
}

XgboostBundle::XgboostBundle(std::string model_path) {
  LOG(INFO) << "Call the constructor XgboostBundle(string)";
  Status status = LoadXgboostModel(model_path);
  if (status.ok()) {
    LOG(INFO) << "Load the xgboost model successfully.";
  } else {
    LOG(ERROR) << status.error_message();
  }
}

Status XgboostBundle::LoadXgboostModel(std::string model_path) {
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
  return Status::OK();
}

Status XgboostBundle::UnloadXgboostModel() {
  int status = XGBoosterFree(xgbooster_);
  xgbooster_ = nullptr;
  return (0 == status) ? Status::OK()
                       : errors::Unknown("Failed to unload the xgboost model: ",
                                         XGBGetLastError());
}

BoosterHandle XgboostBundle::GetBoosterHandle() const { return xgbooster_; }

XgboostBundle::~XgboostBundle() {
  // Unload The Xgboost Model
  LOG(INFO) << "Call the destructor ~XgboostBundle().";
  Status status = UnloadXgboostModel();
  if (status.ok()) {
    LOG(INFO) << "Unload the xgboost model successfully.";
  } else {
    LOG(ERROR) << status.error_message();
  }
}

} // namespace serving
} // namespace tensorflow
