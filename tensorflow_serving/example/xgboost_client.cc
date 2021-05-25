/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include <iostream>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;
using tensorflow::serving::FeatureScore;
using tensorflow::serving::FeatureScoreVector;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  tensorflow::string callPredict(const tensorflow::string& model_name) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);

    google::protobuf::Map<tensorflow::string, FeatureScoreVector>& inputs =
        *predictRequest.mutable_inputs();

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
    
    Status status;
    status = stub_->Predict(&context, predictRequest, &response);

    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is " << response.outputs_size() << std::endl;
      OutMap& map_outputs = *response.mutable_outputs();
      if(map_outputs.contains("index")) {
        tensorflow::TensorProto& result_tensor_proto = map_outputs.at("index");
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {
          std::cout << "the result tensor[" << 0
                    << "] is:" << std::endl
                    << tensor.SummarizeValue(tensor.dim_size(0)) << std::endl;
        } else {
          std::cout << "the result tensor[" << 0
                    << "] convert failed." << std::endl;
        }
      }
      return "Done.";
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
      return "gRPC failed.";
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char** argv) {
  tensorflow::string server_port = "localhost:8500";
  tensorflow::string model_name = "test";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port,
                       "the IP and port of the server"),
      tensorflow::Flag("model_name", &model_name, "name of model")};
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  ServingClient guide(grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
  std::cout<<guide.callPredict(model_name)<<std::endl;
  return 0;
}
