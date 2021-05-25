# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server.

Typical usage example:

    xgboost_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys
import time
import grpc
import numpy
import random
from grpc._cython.cygrpc import CompressionAlgorithm
from grpc._cython.cygrpc import CompressionLevel

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from google.protobuf import text_format

import tensorflow as tf

tf.compat.v1.app.flags.DEFINE_integer('num_tests', 1, 'Number of tests')
tf.compat.v1.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
FLAGS = tf.compat.v1.app.flags.FLAGS


def do_inference(hostport, num_tests):
  """Tests PredictionService with requests.

  Args:
    hostport: Host:port address of the PredictionService.
    num_tests: Number of test images to use.

  Returns:
    void.
  """
  host, port = hostport.split(':')
  options = [
     ("grpc.default_compression_algorithm", CompressionAlgorithm.gzip),
     ("grpc.grpc.default_compression_level", CompressionLevel.high)
  ]
  channel = grpc.insecure_channel(hostport, options)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  for _ in range(num_tests):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    # request.model_spec.version.value = 1
    xgboost_feature_score_1 = predict_pb2.FeatureScore(
      id=[2, 34, 2000, 2206],
      score=[1, 1, 0.646667, 0.727273],
    )
    xgboost_features =  [xgboost_feature_score_1 for i in range(0, 1)]
    xgboost_feature_vector = predict_pb2.FeatureScoreVector(feature_score=xgboost_features)
    request.inputs['xgboost_features'].CopyFrom(xgboost_feature_vector)
    response = stub.Predict(request, 30.0)
    print(response)

def main(_):
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  do_inference(FLAGS.server, FLAGS.num_tests)


if __name__ == '__main__':
  tf.compat.v1.app.run()
