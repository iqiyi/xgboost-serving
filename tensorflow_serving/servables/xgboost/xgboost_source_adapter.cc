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

#include "xgboost_source_adapter.h"

#include <memory>

namespace tensorflow {
namespace serving {

XgboostSourceAdapter::XgboostSourceAdapter()
    : SimpleLoaderSourceAdapter<StoragePath, XgboostBundle>(
          [](const StoragePath &path, std::unique_ptr<XgboostBundle> * bundle) {
            bundle->reset(new XgboostBundle());
            return (*bundle)->LoadXgboostModel(path);
          },
          SimpleLoaderSourceAdapter<StoragePath,
                                    XgboostBundle>::EstimateNoResources()) {}
XgboostSourceAdapter::~XgboostSourceAdapter() { Detach(); }

// Register the source adapter.
class XgboostSourceAdapterCreator {
public:
  static Status
  Create(const XgboostSourceAdapterConfig &config,
         std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>
             *adapter) {
    adapter->reset(new XgboostSourceAdapter());
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(XgboostSourceAdapterCreator,
                                     XgboostSourceAdapterConfig);

} // namespace serving
} // namespace tensorflow
