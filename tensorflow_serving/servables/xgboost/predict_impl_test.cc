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

#include "tensorflow_serving/servables/xgboost/predict_impl.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/platform_config_util.h"
#include "tensorflow_serving/servables/xgboost/xgboost_bundle.h"
#include "tensorflow_serving/test_util/test_util.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {

class PredictImplTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    TF_ASSERT_OK(CreateServerCore(
        test_util::GetCWD() +
            "/tensorflow_serving/servables/xgboost/testdata/test_model/",
        &server_core_));
  }

  static void TearDownTestCase() { server_core_.reset(); }

  static Status CreateServerCore(const string &model_path,
                                 std::unique_ptr<ServerCore> *server_core) {
    ModelServerConfig config;
    auto model_config = config.mutable_model_config_list()->add_config();
    model_config->set_name("test");
    model_config->set_base_path(model_path);
    model_config->set_model_platform("xgboost");

    // For ServerCore Options, we leave servable_state_monitor_creator
    // unspecified so the default servable_state_monitor_creator will be used.
    ServerCore::Options options;
    options.model_server_config = config;
    options.platform_config_map =
        tensorflow::serving::CreateXgboostPlatformConfigMap();
    options.aspired_version_policy =
        std::unique_ptr<AspiredVersionPolicy>(new AvailabilityPreservingPolicy);
    // Reduce the number of initial load threads to be num_load_threads to avoid
    // timing out in tests.
    options.num_initial_load_threads = options.num_load_threads;
    return ServerCore::Create(std::move(options), server_core);
  }

  ServerCore *GetServerCore() { return server_core_.get(); }

private:
  static std::unique_ptr<ServerCore> server_core_;
};

std::unique_ptr<ServerCore> PredictImplTest::server_core_;

TEST_F(PredictImplTest, MissingOrEmptyModelSpec) {
  PredictRequest request;
  PredictResponse response;
  XgboostPredictor predictor;
  // Empty request is invalid.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            predictor.Predict(GetServerCore(), request, &response).code());

  ModelSpec *model_spec = request.mutable_model_spec();
  model_spec->clear_name();
  // Model name is not specified.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            predictor.Predict(GetServerCore(), request, &response).code());

  model_spec->set_name("tmp");
  // Model name is wrong, not found.
  EXPECT_EQ(tensorflow::error::NOT_FOUND,
            predictor.Predict(GetServerCore(), request, &response).code());
}

TEST_F(PredictImplTest, EmptyInputList) {
  PredictRequest request;
  PredictResponse response;
  XgboostPredictor predictor;
  ModelSpec *model_spec = request.mutable_model_spec();
  model_spec->set_name("test");
  model_spec->mutable_version()->set_value(1);
  auto status = predictor.Predict(GetServerCore(), request, &response);
  // The input is empty.
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT, status.code());
}

TEST_F(PredictImplTest, InputsDontMatchModelInputs) {
  PredictRequest request;
  PredictResponse response;
  XgboostPredictor predictor;
  ModelSpec *model_spec = request.mutable_model_spec();
  model_spec->set_name("test");
  google::protobuf::Map<tensorflow::string, FeatureScoreVector> &inputs =
      *request.mutable_inputs();

  // input features
  FeatureScoreVector input_feature_score_vector;
  auto input_feature_score = input_feature_score_vector.add_feature_score();
  input_feature_score->add_id(2);
  input_feature_score->add_score(1);
  input_feature_score->add_id(34);
  input_feature_score->add_score(1);
  input_feature_score->add_id(2000);
  input_feature_score->add_score(0.64667);
  input_feature_score->add_id(2206);
  input_feature_score->add_score(0.727273);
  inputs["input_features"] = input_feature_score_vector;
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            predictor.Predict(GetServerCore(), request, &response).code());
}

TEST_F(PredictImplTest, InputsHaveWrongSize) {
  PredictRequest request;
  PredictResponse response;
  XgboostPredictor predictor;
  ModelSpec *model_spec = request.mutable_model_spec();
  model_spec->set_name("test");
  google::protobuf::Map<tensorflow::string, FeatureScoreVector> &inputs =
      *request.mutable_inputs();

  // xgboost features
  FeatureScoreVector xgboost_feature_score_vector;
  auto xgboost_feature_score = xgboost_feature_score_vector.add_feature_score();
  xgboost_feature_score->add_id(2);
  xgboost_feature_score->add_score(1);
  xgboost_feature_score->add_id(34);
  xgboost_feature_score->add_score(1);
  xgboost_feature_score->add_id(2000);
  xgboost_feature_score->add_score(0.64667);
  xgboost_feature_score->add_id(2206);
  // xgboost_feature_score->add_score(0.727273);
  inputs["xgboost_features"] = xgboost_feature_score_vector;
  EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT,
            predictor.Predict(GetServerCore(), request, &response).code());
}

TEST_F(PredictImplTest, PredictionSuccess) {
  PredictRequest request;
  PredictResponse response;
  XgboostPredictor predictor;
  ModelSpec *model_spec = request.mutable_model_spec();
  model_spec->set_name("test");
  google::protobuf::Map<tensorflow::string, FeatureScoreVector> &inputs =
      *request.mutable_inputs();

  // xgboost features
  FeatureScoreVector xgboost_feature_score_vector;
  auto xgboost_feature_score = xgboost_feature_score_vector.add_feature_score();
  xgboost_feature_score->add_id(2);
  xgboost_feature_score->add_score(1);
  xgboost_feature_score->add_id(34);
  xgboost_feature_score->add_score(1);
  xgboost_feature_score->add_id(2000);
  xgboost_feature_score->add_score(0.64667);
  xgboost_feature_score->add_id(2206);
  xgboost_feature_score->add_score(0.727273);
  inputs["xgboost_features"] = xgboost_feature_score_vector;
  TF_EXPECT_OK(predictor.Predict(GetServerCore(), request, &response));
  TensorProto output_tensor_proto = response.outputs().at("index");
  std::vector<uint32_t> output_tensor_vec(
      output_tensor_proto.uint32_val().begin(),
      output_tensor_proto.uint32_val().end());
  std::vector<uint32_t> out_vec{
      54, 57, 42, 59, 42, 58, 39, 41, 52, 41, 53, 58, 47, 58, 54, 60,
      36, 57, 60, 50, 40, 41, 54, 57, 48, 43, 31, 58, 60, 31, 62, 47,
      47, 52, 58, 40, 44, 53, 54, 53, 42, 31, 40, 36, 56, 54, 61, 60,
      31, 51, 61, 48, 42, 59, 31, 51, 33, 62, 49, 60, 38, 52, 35, 44,
      40, 56, 32, 58, 61, 51, 58, 52, 46, 22, 33, 33, 31, 57, 33, 56,
      49, 39, 56, 52, 49, 61, 36, 49, 48, 39, 31, 44, 31, 46, 32, 33,
      56, 56, 40, 53, 51, 31, 31, 31, 62, 39, 39, 38, 35, 56};
  EXPECT_THAT(out_vec, output_tensor_vec);
}
} // namespace
} // namespace serving
} // namespace tensorflow
