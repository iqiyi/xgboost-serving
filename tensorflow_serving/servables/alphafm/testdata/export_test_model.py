# Written by Hao Ziyu <haoziyu@qiyi.com>, May 2020.

#!/usr/bin/env python3.7

import xgboost as xgb
import scipy

import time

if __name__ == "__main__":
    dtrain = xgb.DMatrix("bazel-out/k8-py3-opt/bin/tensorflow_serving/servables/xgboost/testdata/export_test_model.runfiles/tf_serving/tensorflow_serving/servables/xgboost/testdata/agaricus.txt.train")
    dtest = xgb.DMatrix("bazel-out/k8-py3-opt/bin/tensorflow_serving/servables/xgboost/testdata/export_test_model.runfiles/tf_serving/tensorflow_serving/servables/xgboost/testdata/agaricus.txt.test")
    params = {"max_depth": 2, "eta": 1, "silent": 1, "objective": "binary:logistic"}
    num_round = 2
    bst = xgb.train(params, dtrain, num_round)
    bst.save_model("/tmp/deploy.model")
    indptr = [0, 127]
    indices = [i for i in range(0, 127)]
    data = [i+1 for i in range(0, 127)]
    dtest = scipy.sparse.csr_matrix((data, indices, indptr), shape=(1, 127))
    dtest = xgb.DMatrix(dtest)
    preds = bst.predict(dtest, False, 0, True)
    print(preds)

