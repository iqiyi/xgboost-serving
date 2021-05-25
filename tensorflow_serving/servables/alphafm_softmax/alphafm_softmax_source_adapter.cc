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

#include "alphafm_softmax_source_adapter.h"

#include <memory>

namespace tensorflow {
namespace serving {

AlphafmSoftmaxSourceAdapter::AlphafmSoftmaxSourceAdapter()
    : SimpleLoaderSourceAdapter<StoragePath, AlphafmSoftmaxBundle>(
          [](const StoragePath &path,
             std::unique_ptr<AlphafmSoftmaxBundle> * bundle) {
            bundle->reset(new AlphafmSoftmaxBundle());
            return (*bundle)->LoadAlphafmSoftmaxModel(path);
          },
          SimpleLoaderSourceAdapter<
              StoragePath, AlphafmSoftmaxBundle>::EstimateNoResources()) {}
AlphafmSoftmaxSourceAdapter::~AlphafmSoftmaxSourceAdapter() { Detach(); }

// Register the source adapter.
class AlphafmSoftmaxSourceAdapterCreator {
public:
  static Status
  Create(const AlphafmSoftmaxSourceAdapterConfig &config,
         std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>
             *adapter) {
    adapter->reset(new AlphafmSoftmaxSourceAdapter());
    return Status::OK();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(AlphafmSoftmaxSourceAdapterCreator,
                                     AlphafmSoftmaxSourceAdapterConfig);

} // namespace serving
} // namespace tensorflow
