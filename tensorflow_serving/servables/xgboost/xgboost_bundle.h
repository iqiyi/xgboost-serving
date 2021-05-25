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

#ifndef TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_BUNDLE_H_
#define TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_BUNDLE_H_

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "xgboost/c_api.h"
#include <memory>
#include <string>

namespace tensorflow {
namespace serving {
class XgboostBundle {
public:
  XgboostBundle();

  XgboostBundle(std::string model_path);

  // Load the xgboost model from the given path.
  Status LoadXgboostModel(std::string model_path);

  // Unload the xgboost model.
  Status UnloadXgboostModel();

  // Get the BoosterHandle.
  BoosterHandle GetBoosterHandle() const;

  ~XgboostBundle();

private:
  BoosterHandle xgbooster_; // Handle to the Booster.
};
}
}

#endif // TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_BUNDLE_H_
