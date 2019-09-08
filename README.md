# tensorflow bert for text classification

基于 bert run_classify 模板构建自定义文本分类模型，改动主要有

-  bert.my_data_loader

    - 针对数据构建 MentionProcessor 生成 tfrecord 文件

- model.run_classifier.py

    - 将 TPU estimator 接口替换为 GPU

    - 使用 train_and_evaluate 运行时评估

    - 在 metric_fn 中定义多类别 F1 评估指标


