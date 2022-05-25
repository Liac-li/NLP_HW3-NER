# 实验记录

## Todo List

##### May 24th

- [x] DataLoader and vocab process

##### May 25th

- [ ] LSTM - CRF (Feature functions: $\varphi(y_i, y_{i+1}), \varphi(x_i, y_i)$)
- [ ] Config loss or F1-score of NER in evaluation 
- [ ] MESLoss with CrossEntropyLoss
- [ ] Try BERT model
- [ ] Solve the imbalance problem between classes
    -  SVM for classical models or Hinge loss for deep learning models. When I use standard cross entropy loss, its a nightmare to try and get the weights right [Reddit src](https://www.reddit.com/r/LanguageTechnology/comments/oun69p/comment/h73pmgv/?utm_source=share&utm_medium=web2x&context=3`)
    - (1) tinker with loss function, (2) tinker with learning rate, and (3) tinker with sampling usually gets me out of the "always predicts outside" pit [Reddit src](https://www.reddit.com/r/LanguageTechnology/comments/oun69p/comment/h768ebu/?utm_source=share&utm_medium=web2x&context=3)
- [ ] Figure out which framework is better, BIO labels or BIO + multi-classification on entities
- [ ] Can use Softmax as Decoder instead of CRF layer

## 理论部分

#### 简单想法

1. Begin, Inner, Outer 多分类问题 + 命名实体识别
    可能涉及**多任务学习的问题**
2. 直接把所有的 Tags 编码在一起统一的做识别

#### Math Formalization

**CRF**

$$
    v_t(j) = \max_{i=1}^{T}v_{t-1}(i) a_{ij} b_j(o_t) = \max_{i=1}^{T}v_{t-1}(i) score(X, Y)_{t}
$$

we set the score function as two types of potentials: emission and transition

$$
    \begin{aligned}
        \text{Score}(x, y) &= \sum_{i}\log \phi_{EMIT}(y_i \to x_i) + \log \phi_{TRAN}(y_{i-1} \to y_i)\\
        &= \sum_{i} h_i[y_i] + P_{y_i, y_{i-1}}\\
        & \text{Each parts use a Neural Network to estimate}
    \end{aligned}
$$

with $\mathbf{P} \in \mathbb{R}^{|T|\times|T|}$(tags to tags)

> Score function in General CRF is conclude a weight $Score(x, y) = \sum_{k} w_k \phi_k(x, y)$

> Difference with HMM: crf is data-based (feature function), and HMM is label-based (transition on hidden states)

**Loss Function**

> Supervised Training

- MESLoss(given_tags, viterbi_inferred_tags)
- [FBetaScore](https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/FBetaScore)
    $$
        F_\beta = (1 + \beta^2) * \frac{precision * recall}{(\beta^2\cdot precision) + recall}
    $$

## 实现

- Train Set 40,000, Test Set 4,700

#### bi-LSTM + CRF


## Reference 

- [PyTorch Adanced Tutorial: bi-lstm-crf](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
- [Github: Pytorch Implement of bi-lstm-crf](https://github.com/jidasheng/bi-lstm-crf.git) [MIT License]
