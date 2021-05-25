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

#include "tensorflow_serving/servables/alphafm_softmax/predict_impl.h"

#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/servables/alphafm_softmax/alphafm_softmax_bundle.h"
#include "tensorflow_serving/servables/xgboost/xgboost_constants.h"

#include "tensorflow_serving/servables/xgboost/util.h"

#include <chrono>

namespace tensorflow {
namespace serving {

bvar::LatencyRecorder
    AlphafmSoftmaxPredictor::alphafm_softmax_xgboost_latency_recorder(
        "alphafm_softmax_xgboost_predict");
bvar::LatencyRecorder AlphafmSoftmaxPredictor::alphafm_softmax_latency_recorder(
    "alphafm_softmax_predict");

AlphafmSoftmaxPredictor::AlphafmSoftmaxPredictor() {}

Status AlphafmSoftmaxPredictor::Predict(ServerCore *core,
                                        const PredictRequest &request,
                                        PredictResponse *response) {
  if (!request.has_model_spec()) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Missing ModelSpec");
  }
  return PredictWithModelSpec(core, request.model_spec(), request, response);
}

Status AlphafmSoftmaxPredictor::PredictWithModelSpec(
    ServerCore *core, const ModelSpec &model_spec,
    const PredictRequest &request, PredictResponse *response) {
  ServableHandle<AlphafmSoftmaxBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(model_spec, &bundle));
  int32_t option_mask = 2;
  uint32_t ntree_limit = 0;
  if (!request.inputs().contains(kXGBoostFeaturesName)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "No xgboost_features input found");
  }
  if (!request.inputs().contains(kFMFeaturesName)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "No fm_features input found");
  }
  if (request.inputs().at(kXGBoostFeaturesName).feature_score_size() !=
      request.inputs().at(kFMFeaturesName).feature_score_size()) {
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "Sizes of xgboost_features and fm_features must be the same");
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
                .score_size() ||
        request.inputs().at(kFMFeaturesName).feature_score(i).id_size() !=
            request.inputs()
                .at(kFMFeaturesName)
                .feature_score(i)
                .score_size()) {
      return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                                "Sizes(xgboost_features/fm_features) of id and "
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
  alphafm_softmax_xgboost_latency_recorder << elapsed_seconds.count() * 1000000;
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
  std::vector<uint32_t> out_result_vector(out_len);
  std::copy_n(out_result, out_len, out_result_vector.data());
  uint32_t tree_count = out_result_vector.size() / batch_size;
  uint32_t index = 0;
  auto feature_mapping_handle = bundle->GetFeatureMappingHandle();
  FeatureScoreVector leaf_mapping_feature_score_vector;
  for (int32_t i = 0; i < batch_size; i++) {
    auto leaf_mapping_feature_score =
        leaf_mapping_feature_score_vector.add_feature_score();
    uint64_t feature_id;
    for (int32_t j = 0; j < tree_count; j++) {
      Status status = feature_mapping_handle->GetFeatureId(
          j, out_result_vector[index++], &feature_id);
      if (!status.ok()) {
        continue;
      }
      leaf_mapping_feature_score->add_id(feature_id);
      leaf_mapping_feature_score->add_score(1.0f);
    }
  }
  auto alphafm_softmax_model_handle = bundle->GetAlphafmSoftmaxModelHandle();
  std::vector<std::vector<float>> fm_result;
  start = std::chrono::system_clock::now();
  Status fm_status = alphafm_softmax_model_handle->Predict(
      request.inputs().at(kFMFeaturesName),
      request.inputs().at(kXGBoostFeaturesName),
      leaf_mapping_feature_score_vector, fm_result);
  end = std::chrono::system_clock::now();
  elapsed_seconds = end - start;
  alphafm_softmax_latency_recorder << elapsed_seconds.count() * 1000000;
  if (!fm_status.ok()) {
    LOG(ERROR) << "predict with alphaFM_softmax failed";
    return tensorflow::Status(tensorflow::error::UNKNOWN,
                              "predict with alphaFM_softmax failed");
  }
  Tensor output_tensor{
      DT_FLOAT,
      {fm_result.size(), alphafm_softmax_model_handle->GetClassNum()}};
  for (int i = 0; i < fm_result.size(); i++) {
    for (int j = 0; j < alphafm_softmax_model_handle->GetClassNum(); j++) {
      output_tensor.matrix<float>()(i, j) = fm_result[i][j];
    }
  }
  MakeModelSpec(request.model_spec().name(), /*signature_name=*/{},
                bundle.id().version, response->mutable_model_spec());
  output_tensor.AsProtoField(&((*response->mutable_outputs())["score"]));
  return Status::OK();
}

AlphafmSoftmaxPredictor::~AlphafmSoftmaxPredictor() {}
}
}
