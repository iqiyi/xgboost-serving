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

#ifndef TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_CONSTANTS_H_
#define TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_CONSTANTS_H_

#include <string>

namespace tensorflow {
namespace serving {
const std::string kXGBoostModelFileName = "deploy.model";
const std::string kFmModelFileName = "deploy.fm";
const std::string kFeatureMappingFileName = "deploy.leaf_mapping";
const std::string kXGBoostFeaturesName = "xgboost_features";
const std::string kFMFeaturesName = "fm_features";
} // namespace serving
} // namespace tensorflow

#endif // TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_CONSTANTS_H_
