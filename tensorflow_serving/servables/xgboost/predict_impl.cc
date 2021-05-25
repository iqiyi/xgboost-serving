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

#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/xgboost/xgboost_bundle.h"
#include "tensorflow_serving/servables/xgboost/xgboost_constants.h"

#include "tensorflow_serving/servables/xgboost/util.h"

#include <chrono>

namespace tensorflow {
namespace serving {

bvar::LatencyRecorder
    XgboostPredictor::xgboost_latency_recorder("xgboost_predict");

XgboostPredictor::XgboostPredictor() {}

Status XgboostPredictor::Predict(ServerCore *core,
                                 const PredictRequest &request,
                                 PredictResponse *response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  return PredictWithModelSpec(core, request.model_spec(), request, response);
}

Status XgboostPredictor::PredictWithModelSpec(ServerCore *core,
                                              const ModelSpec &model_spec,
                                              const PredictRequest &request,
                                              PredictResponse *response) {
  ServableHandle<XgboostBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  int32_t option_mask = 2;
  uint32_t ntree_limit = 0;
  if (!request.inputs().contains(kXGBoostFeaturesName)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "No xgboost_features input found");
  }
  int32_t batch_size =
      request.inputs().at(kXGBoostFeaturesName).feature_score_size();
  DMatrixHandle dinput;
  std::vector<bst_ulong> indptr_vector;
  std::vector<uint32_t> indices_vector;
  std::vector<float> data_vector;
  bst_ulong row_indptr = 0;
  indptr_vector.reserve(1 + batch_size);
  indptr_vector.push_back(row_indptr);
  for (int32_t i = 0; i < batch_size; i++) {
    if (request.inputs().at(kXGBoostFeaturesName).feature_score(i).id_size() !=
        request.inputs()
            .at(kXGBoostFeaturesName)
            .feature_score(i)
            .score_size()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "Sizes(xgboost_features) of id and "
                                "score must be the same");
    }
    row_indptr +=
        request.inputs().at(kXGBoostFeaturesName).feature_score(i).id_size();
    indptr_vector.push_back(row_indptr);
  }
  indices_vector.reserve(row_indptr);
  indices_vector.resize(row_indptr);
  data_vector.reserve(row_indptr);
  data_vector.resize(row_indptr);
  int32_t pre_size = 0;
  for (int32_t i = 0; i < batch_size; i++) {
    std::copy_n(
        request.inputs().at(kXGBoostFeaturesName).feature_score(i).id().begin(),
        request.inputs().at(kXGBoostFeaturesName).feature_score(i).id_size(),
        indices_vector.begin() + pre_size);
    std::copy_n(
        request.inputs()
            .at(kXGBoostFeaturesName)
            .feature_score(i)
            .score()
            .begin(),
        request.inputs().at(kXGBoostFeaturesName).feature_score(i).score_size(),
        data_vector.begin() + pre_size);
    pre_size +=
        request.inputs().at(kXGBoostFeaturesName).feature_score(i).id_size();
  }
  bst_ulong num_feature = 0;
  int32_t status =
      XGBoosterGetNumFeature(bundle->GetBoosterHandle(), &num_feature);
  if (0 != status) {
    LOG(ERROR) << "XGBoosterGetNumFeature failed: " << XGBGetLastError();
    return tensorflow::Status(
        tensorflow::error::UNKNOWN,
        "get num_feature with XGBoosterGetNumFeature failed");
  }
  status = XGDMatrixCreateFromCSREx(
      indptr_vector.data(), indices_vector.data(), data_vector.data(),
      /*nindptr*/ indptr_vector.size(), /*nelem*/ data_vector.size(),
      /*num_col*/ num_feature, &dinput);
  if (0 != status) {
    LOG(ERROR) << "XGDMatrixCreateFromCSR failed: " << XGBGetLastError();
    int32_t status_free = XGDMatrixFree(dinput);
    if (0 != status_free) {
      LOG(ERROR) << "call XGDMatrixFree to free dinput failed";
    }
    return tensorflow::Status(
        tensorflow::error::UNKNOWN,
        "create dmatrix with XGDMatrixCreateFromCSREx failed");
  }
  bst_ulong out_len = 0;
  const float *out_result = NULL;
  auto start = std::chrono::system_clock::now();
  status = XGBoosterPredict(bundle->GetBoosterHandle(), dinput, option_mask,
                            ntree_limit, /*training*/ 0, &out_len, &out_result);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  xgboost_latency_recorder << elapsed_seconds.count() * 1000000;
  if (0 != status) {
    LOG(ERROR) << "XGBoosterPredict failed: " << XGBGetLastError();
    int32_t status_free = XGDMatrixFree(dinput);
    if (0 != status_free) {
      LOG(ERROR) << "call XGDMatrixFree to free dinput failed";
    }
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "predict with XGBoosterPredict failed");
  }
  int32_t status_free = XGDMatrixFree(dinput);
  if (0 != status_free) {
    LOG(ERROR) << "call XGDMatrixFree to free dinput failed"
               << XGBGetLastError();
  }
  if (2 == option_mask) {
    std::vector<unsigned int> out_result_p(out_len);
    std::copy_n(out_result, out_len, out_result_p.data());
    Tensor output_tensor{DT_UINT32, {out_len}};
    std::copy_n(out_result_p.data(), output_tensor.flat<unsigned int>().size(),
                output_tensor.flat<unsigned int>().data());
    MakeModelSpec(request.model_spec().name(), /*signature_name=*/{},
                  bundle.id().version, response->mutable_model_spec());
    output_tensor.AsProtoField(&((*response->mutable_outputs())["index"]));
  } else {
    Tensor output_tensor{DT_FLOAT, {out_len}};
    std::copy_n(out_result, output_tensor.flat<float>().size(),
                output_tensor.flat<float>().data());
    MakeModelSpec(request.model_spec().name(), /*signature_name=*/{},
                  bundle.id().version, response->mutable_model_spec());
    if (1 == option_mask) {
      output_tensor.AsProtoField(&((*response->mutable_outputs())["margin"]));
    } else if (0 == option_mask) {
      output_tensor.AsProtoField(&((*response->mutable_outputs())["value"]));
    }
  }
  return Status::OK();
}

XgboostPredictor::~XgboostPredictor() {}
}
}
