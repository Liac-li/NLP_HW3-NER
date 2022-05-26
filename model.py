import torch
import torch.nn as nn
import torch.nn.functional as F

class BiRNN_NER(nn.Module):
    def __init__(self, args, target_size, class_weight=None):
        super(BiRNN_NER, self).__init__()
        self.args = args
        self.target_size = target_size
        self.class_weight = class_weight if class_weight is not None else torch.ones(target_size)

        self.embedding = nn.Embedding(args.vocab_size, args.embed_size)
        self.encoder = nn.LSTM(args.embed_size, args.hidden_size, args.rnn_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(args.hidden_size * 2, target_size)
        
        self.loss_func = nn.CrossEntropyLoss(weight=self.class_weight, reduction='sum')
        
    def forward(self, x, mask):

        embeds = self.embedding(x)
        rnn_out, _ = self.encoder(embeds)
        out = self.decoder(rnn_out)
        out = F.log_softmax(out, dim=2)
        return out
    
    def loss(self, x, tags, mask):
        # TODO: handle mask
        
        # vec_tags = F.one_hot(tags, num_classes=self.target_size).to(tags.device)
        # vec_tags = torch.permute(vec_tags, (0, 2, 1))
        
        out = self.forward(x, mask)
        out = torch.permute(out, (0, 2, 1)) # (batch_size, target_size, seq_len)
        # print(out.size(), vec_tags.size())
        loss = self.loss_func(out, tags) # (batch_size, seq_len)
        return loss

