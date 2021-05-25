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

#include "tensorflow_serving/servables/xgboost/xgboost_source_adapter.h"

#include <chrono>
#include <thread>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/sources/storage_path/file_system_storage_path_source.h"

namespace tensorflow {
namespace serving {
namespace {
TEST(XgboostSourceAdapterTest, Basic) {
  std::unique_ptr<FileSystemStoragePathSource> path_source;
  FileSystemStoragePathSourceConfig source_config;
  auto servable_0 = source_config.add_servables();
  servable_0->set_servable_name("test_0");
  servable_0->set_base_path("/tmp");
  FileSystemStoragePathSourceConfig::ServableVersionPolicy spolicy;
  (spolicy.mutable_latest())->set_num_versions(2);
  *servable_0->mutable_servable_version_policy() = spolicy;
  auto servable_1 = source_config.add_servables();
  servable_1->set_servable_name("test_1");
  servable_1->set_base_path("/tmp");
  source_config.set_file_system_poll_wait_seconds(1);
  FileSystemStoragePathSource::Create(source_config, &path_source);
  auto adapter =
      std::unique_ptr<XgboostSourceAdapter>(new XgboostSourceAdapter());
  ConnectSourceToTarget(path_source.get(), adapter.get());
  std::unique_ptr<AspiredVersionsManager> manager;
  auto policy =
      std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
  AspiredVersionsManager::Options manager_options;
  manager_options.manage_state_interval_micros = 1000000;
  manager_options.aspired_version_policy = std::move(policy);
  TF_ASSERT_OK(
      AspiredVersionsManager::Create(std::move(manager_options), &manager));
  ConnectSourceToTarget(adapter.get(), manager.get());
  Status status = path_source->PollFileSystemAndInvokeCallback();
  TF_ASSERT_OK(status);
}
} // namespace
} // namespace serving
} // namespace tensorflow
