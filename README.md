# Bi-LSTM for NER task

基于 `PyTorch` 框架实现的 NER 任务的分类，利用 LSTM 作为 Encoder，尝试了使用 Softmax, CRF 作为 Decoder 层来实现隐层和预测。默认下会保存所有 epochs 中的训练数据，同时利用严格的 f1-score 作为最优模型的评测

NER 标签使用的是 BIO 模式组织的


```bash
.
├── colorUtil.py    
├── crf.py          # CRF layer
├── dataProc.py
├── model.py        # 基本的模型框架
├── myUtils.py      # 多线程版本的严格 F1-score 实现
├── README.md
├── recode.md
└── train.py       
```

运行模型


```
python train.py
```