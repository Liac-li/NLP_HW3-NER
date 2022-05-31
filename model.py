# from tkinter import W
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from itertools import zip_longest

from crf import CRF

class BiRNN_NER(nn.Module):
    def __init__(self, args, target_size, class_weight=None):
        super(BiRNN_NER, self).__init__()
        self.args = args
        self.target_size = target_size
        self.class_weight = class_weight if class_weight is not None else torch.ones(target_size)

        self.embedding = nn.Embedding(args.vocab_size, args.embed_size)
        self.multiHeadAttn = nn.MultiheadAttention(args.hidden_size * 2, 4, batch_first=True)
        self.rnn_layer = nn.LSTM(args.embed_size, args.hidden_size, args.rnn_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)

        self.decoder = nn.Linear(args.hidden_size * 2, target_size)
        self.crf = CRF(args.hidden_size * 2, target_size)
        self.loss_func = nn.CrossEntropyLoss(weight=self.class_weight, reduction='sum')
        
    def __encode(self, x, mask):
        embeds = self.embedding(x)
        
        # print(mask.min(), mask.max())
        packed_seq = pack_padded_sequence(embeds, mask.cpu().int(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn_layer(packed_seq)
        rnn_out, seq_len_unpacked = pad_packed_sequence(packed_out, batch_first=True) 
        attn_out, _ = self.multiHeadAttn(rnn_out, rnn_out, rnn_out)
        # out = self.dropout(attn_out)
        out = attn_out

        block_mask = torch.zeros((out.size(0), out.size(1))).to(x.device)
        for id, seq_len in enumerate(seq_len_unpacked):
            block_mask[id, :seq_len] = 1

        return out, block_mask

       
    def forward(self, x, mask):

        out, block_mask = self.__encode(x, mask)
        _, out = self.crf(out, block_mask)
        # print(out)
        # out = F.log_softmax(out, dim=2)
        out = torch.tensor(list(zip_longest(*out, fillvalue=0))).to(x.device)
        out = torch.permute(out, (1, 0))
        # print(out.size())
        return out
    
    def loss(self, x, tags, mask):
        # TODO: handle mask
        
        # out = self.forward(x, mask)
        # tags = tags[:,:out.size(1)] 
        # block_mask = torch.zeros((out.size(0), out.size(1))).to(x.device)
        # for id, seq_len in enumerate(mask):
        #     block_mask[id, :seq_len] = 1
        out, block_mask = self.__encode(x, mask)
        tags = tags[:,:out.size(1)]

        loss = self.crf.loss(out, tags, block_mask)
        # out = torch.permute(out, (0, 2, 1)) # (batch_size, target_size, seq_len)
        # loss = self.loss_func(out, tags) # (batch_size, seq_len)
        return loss

