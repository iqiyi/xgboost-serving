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

#ifndef TENSORFLOW_SERVING_SERVABLES_ALPHAFM_SOFTMAX_ALPHAFM_SOFTMAX_BUNDLE_H_
#define TENSORFLOW_SERVING_SERVABLES_ALPHAFM_SOFTMAX_ALPHAFM_SOFTMAX_BUNDLE_H_

#include "tensorflow_serving/servables/alphafm/feature_mapping.h"
#include "tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_model.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "xgboost/c_api.h"
#include <memory>
#include <string>

namespace tensorflow {
namespace serving {
class AlphafmSoftmaxBundle {
public:
  AlphafmSoftmaxBundle();

  AlphafmSoftmaxBundle(std::string model_path);

  // Load the alphafm_softmax model from the given path.
  Status LoadAlphafmSoftmaxModel(std::string model_path);

  // Unload the alphafm_softmax model.
  Status UnloadXgboostModel();

  // Get the BoosterHandle.
  BoosterHandle GetBoosterHandle() const;

  // Get the FeatureMappingHandle.
  FeatureMapping *GetFeatureMappingHandle() const;

  // Get the AlphafmSoftmaxModelHandle.
  AlphafmSoftmaxModel *GetAlphafmSoftmaxModelHandle() const;

  ~AlphafmSoftmaxBundle();

private:
  BoosterHandle xgbooster_; // Handle to the Booster.
  std::unique_ptr<AlphafmSoftmaxModel> alphafm_softmax_model_;
  std::unique_ptr<FeatureMapping>
      feature_mapping_; // Handle to the FeatureMapping.
};
}
}

#endif // TENSORFLOW_SERVING_SERVABLES_ALPHAFM_SOFTMAX_ALPHAFM_SOFTMAX_BUNDLE_H_
