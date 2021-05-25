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

#ifndef TENSORFLOW_SERVING_SERVABLES_ALPHAFM_ALPHAFM_BUNDLE_H_
#define TENSORFLOW_SERVING_SERVABLES_ALPHAFM_ALPHAFM_BUNDLE_H_

#include "tensorflow_serving/servables/alphafm/alphafm_model.h"
#include "tensorflow_serving/servables/alphafm/feature_mapping.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "xgboost/c_api.h"
#include <memory>
#include <string>

namespace tensorflow {
namespace serving {
class AlphafmBundle {
public:
  AlphafmBundle();

  AlphafmBundle(std::string model_path);

  // Load the model from the given path.
  Status LoadAllModel(std::string model_path);

  // Unload the xgboost model.
  Status UnloadXgboostModel();

  // Get the BoosterHandle.
  BoosterHandle GetBoosterHandle() const;

  // Get the FeatureMappingHandle.
  FeatureMapping *GetFeatureMappingHandle() const;

  // Get the FmModelHandle.
  AlphaFmModel *GetAlphafmModelHandle() const;

  ~AlphafmBundle();

private:
  BoosterHandle xgbooster_;                     // Handle to the Booster.
  std::unique_ptr<AlphaFmModel> alphafm_model_; // Handle to the AlphaFmModel.
  std::unique_ptr<FeatureMapping>
      feature_mapping_; // Handle to the FeatureMapping.
};
}
}

#endif // TENSORFLOW_SERVING_SERVABLES_ALPHAFM_ALPHAFM_BUNDLE_H_
