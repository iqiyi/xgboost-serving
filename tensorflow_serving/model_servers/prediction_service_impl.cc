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

#include "tensorflow_serving/model_servers/prediction_service_impl.h"

#include "grpc/grpc.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"

#include <chrono>
#include <thread>

namespace tensorflow {
namespace serving {

bvar::LatencyRecorder XgboostPredictionServiceImpl::latency_recorder("xgboost_serving");

namespace {

int DeadlineToTimeoutMillis(const gpr_timespec deadline) {
  return gpr_time_to_millis(
      gpr_time_sub(gpr_convert_clock_type(deadline, GPR_CLOCK_MONOTONIC),
                   gpr_now(GPR_CLOCK_MONOTONIC)));
}

}  // namespace

::grpc::Status XgboostPredictionServiceImpl::Predict(
    ::grpc::ServerContext* context, const PredictRequest* request,
    PredictResponse* response) {
    auto start = std::chrono::system_clock::now();
    const ::grpc::Status status = ToGRPCStatus(predictor_->Predict(core_, *request, response));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    latency_recorder<<elapsed_seconds.count()*1000000;
    if (!status.ok()) {
        VLOG(1) << "Predict failed: " << status.error_message();
    }
    return status;
}

::grpc::Status XgboostPredictionServiceImpl::PredictAlphafm(
    ::grpc::ServerContext* context, const PredictRequest* request,
    PredictResponse* response) {
    auto start = std::chrono::system_clock::now();
    const ::grpc::Status status = ToGRPCStatus(alphafm_predictor_->Predict(core_, *request, response));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    latency_recorder<<elapsed_seconds.count()*1000000;
    if (!status.ok()) {
        VLOG(1) << "Predict failed: " << status.error_message();
    }
    return status;
}

::grpc::Status XgboostPredictionServiceImpl::PredictAlphafmSoftmax(
    ::grpc::ServerContext* context, const PredictRequest* request,
    PredictResponse* response) {
    auto start = std::chrono::system_clock::now();
    const ::grpc::Status status = ToGRPCStatus(alphafm_softmax_predictor_->Predict(core_, *request, response));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    latency_recorder<<elapsed_seconds.count()*1000000;
    if (!status.ok()) {
        VLOG(1) << "Predict failed: " << status.error_message();
    }
    return status;
}

}  // namespace serving
}  // namespace tensorflow
