/* Copyright 2018 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/xgboost/predict_impl.h"
#include "tensorflow_serving/servables/alphafm/predict_impl.h"
#include "tensorflow_serving/servables/alphafm_softmax/predict_impl.h"

#include <bvar/bvar.h>

namespace tensorflow {
namespace serving {

class XgboostPredictionServiceImpl final : public PredictionService::Service {
public:
    explicit XgboostPredictionServiceImpl(ServerCore* server_core)
        : core_(server_core),
          predictor_(new XgboostPredictor()) {}

    ::grpc::Status Predict(::grpc::ServerContext* context,
                           const PredictRequest* request,
                           PredictResponse* response) override;

    ::grpc::Status PredictAlphafm(::grpc::ServerContext* context,
		                  const PredictRequest* request,
				  PredictResponse* response) override;

    ::grpc::Status PredictAlphafmSoftmax(::grpc::ServerContext* context,
                           const PredictRequest* request,
                           PredictResponse* response) override;

private:
    ServerCore* core_;
    std::unique_ptr<XgboostPredictor> predictor_;
    std::unique_ptr<AlphafmPredictor> alphafm_predictor_;
    std::unique_ptr<AlphafmSoftmaxPredictor> alphafm_softmax_predictor_;
    static bvar::LatencyRecorder latency_recorder;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
