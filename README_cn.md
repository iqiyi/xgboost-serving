# XGBoost Serving

XGBoost Serving 是 TensorFlow Serving 的 1 个 Fork 开发版本，增加了 [XGBoost](https://github.com/dmlc/xgboost)、[alphaFM](https://github.com/CastellanZhang/alphaFM) 和 [alphaFM_softmax](https://github.com/CastellanZhang/alphaFM_softmax) 等框架的支持。如果想了解更多关于 TensorFlow Serving 的资料，请切换到 `master` 分支或者访问 [TensorFlow Serving](https://github.com/tensorflow/serving)。

----
XGBoost Serving 是为生产环境而设计的支持 XGBoost 和 FM 模型的灵活、高性能的推理系统。其专注于模型训练结束后的在线推理部分，支持模型的完整生命周期管理，为客户端提供高性能的带版本访问接口，在爱奇艺内部具有广泛应用。

XGBoost Serving 主要具有以下特性：

-   支持多模型、多版本部署
-   支持 gRPC APIs
-   版本更新对客户端透明
-   支持金丝雀部署和 A/B 测试
-   高性能
-   支持 XGBoost 模型、XGBoost 和 FM 模型、XGBoost 和 alphaFM_softmax 模型部署
-   支持统计计算延时分布

## 文档

### 构建

构建 XGBoost Serving 最简单和直观的方式是使用 Docker 开发镜像。我们优先推荐使用 Docker 开发镜像进行构建，除非有 Docker 环境无法满足的需求。

*   [构建 XGBoost Serving](tensorflow_serving/docs/cn/building.md)
*   [在 Docker 中构建 XGBoost Serving](tensorflow_serving/docs/cn/building_with_docker.md)*(推荐)*

### 使用

#### 导出 XGBoost && FM 模型

在部署 XGBoost && FM 模型之前，需要导出 XGBoost 模型、leaf mapping 和 FM 模型。

请参考 [导出 XGBoost && FM 模型](tensorflow_serving/docs/cn/export_model.md) 以了解模型的规范以及如何导出 XGBoost && FM 模型。

#### 使用和配置

* [部署 XGBoost && FM 模型](tensorflow_serving/docs/cn/serving_basic.md)
* [配置 XGBoost Serving](tensorflow_serving/docs/cn/serving_config.md)

### 扩展

TensorFlow Serving 具有模块化的架构，提供了若干扩展点。由于 XGBoost Serving 派生自 TensorFlow Serving，所以也可以使用 TensorFlow Serving 提供的扩展点。

* [在 Docker 环境构建 Tensorflow Serving](tensorflow_serving/g3doc/building_with_docker.md)
* [了解 Tensorflow Serving 的架构](tensorflow_serving/g3doc/architecture.md)
* [Tensorflow Serving C++ API 参考](https://www.tensorflow.org/tfx/serving/api_docs/cc/)
* [创建新的 Servable](tensorflow_serving/g3doc/custom_servable.md)
* [创建新的 Source](tensorflow_serving/g3doc/custom_source.md)

## 贡献

**如果想向 XGBoost Serving 贡献代码或文档等，请参考：[贡献指南](CONTRIBUTING.md)。**

## 反馈和参与

* 请通过 [Github Issues](https://github.com/iqiyi/xgboost-serving/issues) 报告 bugs、提问及建议。
