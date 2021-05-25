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

#include "tensorflow_serving/servables/alphafm/feature_mapping.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow_serving/util/strings/numeric.h"
#include "tensorflow_serving/util/strings/split.h"

#include <fstream>
#include <vector>

namespace tensorflow {
namespace serving {

FeatureMapping::FeatureMapping() {
  LOG(INFO) << "Call the constructor FeatureMapping().";
}

FeatureMapping::FeatureMapping(std::string model_path) {
  LOG(INFO) << "Call the constructor FeatureMapping(model_path).";
  Status status = LoadModel(model_path);
  if (status.ok()) {
    LOG(INFO) << "Load the FeatureMapping Model successfully.";
  } else {
    LOG(ERROR) << "Failed to load the FeatureMapping Model.";
  }
}

FeatureMapping::~FeatureMapping() {
  LOG(INFO) << "Call the destructor ~FeatureMapping().";
  UnloadModel();
}

Status FeatureMapping::LoadModel(std::string model_path) {
  LOG(INFO) << "Call the method LoadModel(model_path).";
  std::ifstream infile(model_path, std::ios::in | std::ios::binary);
  if (!infile) {
    LOG(ERROR) << "Failed to open the file: " + model_path;
    return errors::Unknown("Failed to open the file: " + model_path);
  }
  string line;
  while (!infile.eof()) {
    std::getline(infile, line);
    if (line.empty()) {
      LOG(WARNING) << "Empty line";
      continue;
    }
    std::vector<string> split_vec;
    ::SplitStringUsing(line, " \t", &split_vec);
    if (split_vec.size() < 2) {
      LOG(WARNING) << "Feature mapping line parsing error: " << line;
      continue;
    }
    uint32_t tree_num;
    if (!::safe_strtoul(split_vec[0], &tree_num)) {
      LOG(WARNING) << "Feature mapping line parsing error: " << line;
      continue;
    }
    auto itr = split_vec.begin() + 1;
    for (; itr != split_vec.end(); ++itr) {
      std::vector<string> tree_leaf_pair_vec;
      ::SplitStringUsing(*itr, ":", &tree_leaf_pair_vec);
      if (tree_leaf_pair_vec.size() != 2) {
        LOG(WARNING) << "Feature mapping line parsing error: " << line;
        continue;
      }
      uint32_t leaf_index;
      int feature_id;
      if (!::safe_strtoul(tree_leaf_pair_vec[0], &leaf_index) ||
          !::safe_strtol(tree_leaf_pair_vec[1], &feature_id)) {
        LOG(WARNING) << "Feature mapping line parsing error: " << line;
        continue;
      }
      tree_leaf_feature_map_[tree_num].emplace(leaf_index, feature_id);
    }
  }
  return Status::OK();
}

Status FeatureMapping::GetFeatureId(uint32_t tree_num, uint32_t leaf_index,
                                    uint64_t *feature_id) {
  return GetFeatureIdInMap(tree_num, leaf_index, feature_id);
}

Status FeatureMapping::GetFeatureIdInMap(uint32_t tree_num, uint32_t leaf_index,
                                         uint64_t *feature_id) {
  auto tree_num_itr = tree_leaf_feature_map_.find(tree_num);
  if (tree_num_itr == tree_leaf_feature_map_.end()) {
    return errors::Unknown(
        "Failed to find the tree_num in the tree_leaf_feature_map.");
  }
  auto feature_id_itr = tree_num_itr->second.find(leaf_index);
  if (feature_id_itr == tree_num_itr->second.end()) {
    return errors::Unknown(
        "Failed to find the leaf_index in the tree_leaf_feature_map.");
  }
  *feature_id = feature_id_itr->second;
  return Status::OK();
}

void FeatureMapping::UnloadModel() {
  LOG(INFO) << "Call the method UnloadModel().";
  tree_leaf_feature_map_.clear();
}

} // namespace serving
} // namespace tensorflow
