from unicodedata import bidirectional
import mindspore
from mindspore import Parameter
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

import mindspore.numpy as np


class TextCNN(nn.Cell):
    """
        Architecture:
            Embedding -> Conv -> MaxPool -> Dropout -> FC
    """
    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size=config.vocab_size,
                                      embedding_size=config.embedding_size)
        # 特别注意如果用卷积，要指定pad_mode的不同模式 默认的pad_mode会通过pad使得卷积前后大小一致
        self.conv = nn.Conv2d(1,
                              config.core_channels, (4, config.embedding_size),
                              pad_mode='valid')

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Dense(config.core_channels, config.num_classes)

        self.maxpool1d = nn.MaxPool1d(kernel_size=29)

        self.squeeze_dims_1 = ops.Squeeze(
            axis=1)  #这样写是压缩axis=1上的维度，如果该维度不为1，报错
        self.squeeze_dims = ops.Squeeze()  #这样写是将tensor中所有维度为1的压缩掉
        # 注意看这里shape relu等使用方法，要先声明再用
        self.shape = ops.Shape()
        self.relu = nn.ReLU()
        self.expand_dims = ops.ExpandDims()
        self.concat = mindspore.ops.Concat(axis=1)

    # 卷积池化操作
    def conv_and_pool(self, x, conv, maxpool):
        x = self.squeeze_dims(self.relu(conv(x)))
        x = self.squeeze_dims(maxpool(x))
        return x

    #等于forward
    def construct(self, x):
        embedding_x = self.embedding(x)
        out = self.expand_dims(embedding_x, 1)
        out = self.conv_and_pool(out, self.conv, self.maxpool1d)
        out = self.fc(self.dropout(out))
        return out


class TextRNN(nn.Cell):
    """
        x => rnn_block => dropout => fc
    """
    def __init__(self,
                 vocab_size,
                 n_class,
                 n_hidden,
                 num_out,
                 num_direction=1,
                 layer_num=1,
                 dropout=0.5):
        super(TextRNN, self).__init__()
        self.hide_size = n_hidden
        self.n_dirct = num_direction
        self.n_layer = layer_num
        self.dropout = dropout

        bi_direct = True if self.n_dirct == 2 else False

        self.embedding = nn.Embedding(vocab_size, n_class)
        self.rnn = nn.LSTM(input_size=n_class,
                           hidden_size=n_hidden,
                           batch_first=True,
                           bidirectional=bi_direct)
        # self.prev_h = Tensor(np.zeros((1, n_hidden)).astype(np.float32))
        self.fc = nn.Dense(n_hidden * self.n_dirct, 500, activation=nn.ReLU())
        self.fc2 = nn.Dense(500, num_out, activation=nn.LogSigmoid())

        self.dp = nn.Dropout(keep_prob=self.dropout)
        self.slice = ops.Slice()
        self.squeeze_1 = ops.Squeeze(1)
        # self.permuter = ops.Transpose()
        self.softmax = nn.Softmax()
        # self.squeeze = ops.Squeeze()
        # self.argmax = ops.Argmax()
        self.gatea = nn.Dense(n_hidden * self.n_dirct,
                              32,
                              activation=nn.Tanh())

        # self.conv = nn.Conv2d(32, 16, (3, 5), pad_mode='valid')
        self.maxpool = nn.MaxPool1d(kernel_size=5)
        # self.convfc1 = nn.Dense(7136, 4096, activation=nn.ReLU())
        # self.convfc2 = nn.Dense(4096, 10, activation=nn.Sigmoid())
        
        self.conv_seq = nn.Conv1d(self.hide_size * self.n_dirct, self.hide_size, 5, pad_mode='valid', weight_init='normal')
        self.conv_seqfc = nn.Dense(self.hide_size * self.n_dirct * 28, 10, activation=nn.ReLU())


    def selfattenLayer(self, inputs):
        """
            input: (batch, seq_len, hidden_size)
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        hidden_size = inputs.shape[2]

        tanh_op = nn.Tanh()
        permute_op = ops.Transpose()
        sum_op = ops.ReduceSum()
        bmm_op = ops.BatchMatMul()
        softmax_op = ops.Softmax()

        score = bmm_op(inputs, permute_op(inputs, (0, 2, 1)))
        score = tanh_op(score)
        attn = softmax_op(score.view(-1,
                                     seq_len)).view(batch_size, -1, seq_len)
        output = bmm_op(attn, inputs)
        output = sum_op(output, 1)
        output_norm = nn.Norm(axis=1, keep_dims=True)(output)
        return output / output_norm

    def general_attn(self, input):
        bmm_op = ops.BatchMatMul()
        sum_op = ops.ReduceSum()

        attn = self.gatea(input)  # (batch, seq_len, seq_len)
        output = bmm_op(attn, input)  # (batch, seq_len, hidden_size)
        output = sum_op(output, 1)
        out = output / nn.Norm(axis=1, keep_dims=True)(output)
        return out

    def convAndpool(self, x):
        """
        Input: 
            x : (batch_size, seq_len, embedding_size)
        """
        expenddim_op = ops.ExpandDims()
        squeeze_dims = ops.Squeeze()

        x = expenddim_op(x, -1)
        # print(x.shape)

        x = x.view(x.shape[0], x.shape[1], 30, -1)
        out = nn.ReLU()(self.conv(x))
        out = out.view(out.shape[0], out.shape[1], -1)
        # out = ops.Squeeze(2)(out)
        out = self.maxpool(out).view(out.shape[0], -1)
        # out = squeeze_dims(out)
        # print(out.shape, type(out))
        out = self.dp(out)
        out = self.convfc1(out)
        out = self.dp(out)
        out = self.convfc2(out)
        return out

    def convOnSeq(self, x):
        permute_op = ops.Transpose()

        x_t = permute_op(x, (0, 2, 1))
        cnn_out = self.conv_seq(x_t)
        cnn_out = self.maxpool(x_t)
        cnn_out = self.dp(cnn_out)
        cnn_out = permute_op(cnn_out, (0, 2, 1)).view(x.shape[0], -1)
        fc_out = self.conv_seqfc(cnn_out)
        return fc_out


    def construct(self, x):
        """
            Input:
                x: (batch, time_step, input_size)
                seqLen: (batch)
        """
        # print(type(x), x)
        # print(set_color(f"{x.shape}", COLOR.RED))
        # print(seq_len)

        # print(seq_length.shape)
        # print(x.shape)
        x = self.embedding(x)
        # batch_size = x.shape[0]
        # seq_len = x.shape[1]

        # h0 = Tensor(np.ones([self.n_dirct*1, batch_size, self.hide_size]).astype(np.float32))
        # c0 = Tensor(np.ones([self.n_dirct*1, batch_size, self.hide_size]).astype(np.float32))

        rnn_out, _ = self.rnn(
            x)  # (batch, time_step, num_directions*hidden_size)

        # last_hdState = self.slice(
        #     rnn_out, (0, seq_len - 1, 0),
        #     (-1, -1, -1))  # (batch, 1, num_directions*hidden_size)
        # last_hdState = rnn_out[:, -1, : ]

        # attn_out = self.selfattenLayer(rnn_out) # (batch_size, ...)
        # attn_out = self.general_attn(rnn_out)
        # conv_out = self.convAndpool(rnn_out)
        conv_sep = self.convOnSeq(rnn_out)
        return conv_sep

        # dropout = self.dp(conv_out)
        # out = self.fc(dropout)
        # out = self.fc2(out)
        # # print(out.shape)

        # return out


class TextRCNN(nn.Cell):
    def __init__(self, config, n_direct=2):
        super(TextRCNN, self).__init__()
        
        self.n_direct = n_direct
        self.hidden_size = 300
        self.num_layers = 2
        self.batch_size = config.batch_size
        
        bidirect = True if n_direct == 2 else False

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn_block = nn.LSTM(
            config.embedding_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=bidirect,
            batch_first=True,
            dropout=config.dropout
        )
        
        self.maxpool_1d = nn.MaxPool1d(kernel_size=config.pad_size)
        self.fc = nn.Dense(self.hidden_size * n_direct + config.embedding_size,
                config.num_classes)
        self.relu = nn.ReLU()
        
    def construct(self, x):
        concat_op = ops.Concat(axis=2) 
        permute_op = ops.Transpose()

        embed = self.embedding(x)
        rnn_out, _ = self.rnn_block(embed)
        rnn_out = concat_op((embed, rnn_out))
        out = self.relu(rnn_out)
        out = permute_op(out, (0, 2, 1))
        out = self.maxpool_1d(out).squeeze()
        out = self.fc(out)
        
        return out

