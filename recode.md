# 实验记录

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

with $\bm{P} \in \mathbb{R}^{|T|\times|T|}$(tags to tags)

> Score function in General CRF is conclude a weight $Score(x, y) = \sum_{k} w_k \phi_k(x, y)$

> Difference with HMM: crf is data-based (feature function), and HMM is label-based (transition on hidden states)

**Loss Function**

> Supervised Training

- MESLoss(given_tags, viterbi_inferred_tags)

## 实现

#### bi-LSTM + CRF


## Reference 

- [PyTorch Adanced Tutorial: bi-lstm-crf](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
- [Github: Pytorch Implement of bi-lstm-crf](https://github.com/jidasheng/bi-lstm-crf.git) [MIT License]
