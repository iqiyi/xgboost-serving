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

#ifndef TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_SOURCE_ADAPTER_H_

#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/servables/xgboost/xgboost_bundle.h"
#include "tensorflow_serving/servables/xgboost/xgboost_source_adapter.pb.h"
#include "xgboost/c_api.h"

namespace tensorflow {
namespace serving {
class XgboostSourceAdapter final
    : public SimpleLoaderSourceAdapter<StoragePath, XgboostBundle> {
public:
  XgboostSourceAdapter();
  ~XgboostSourceAdapter() override;

private:
  friend class XgboostSourceAdapterCreator;
  TF_DISALLOW_COPY_AND_ASSIGN(XgboostSourceAdapter);
};
}
}

#endif // TENSORFLOW_SERVING_SERVABLES_XGBOOST_XGBOOST_SOURCE_ADAPTER_H_
