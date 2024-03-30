import torch
import torch.nn as nn
from models.SegRNN import SegRNN

class DiscriminatorWithSegRNN(nn.Module):
    '''
    Discriminator Class
    Values:
      in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
      hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self,cfg, mean, std):
        super(DiscriminatorWithSegRNN, self).__init__()
        self.hidden_dim = cfg.hid_d
        self.mean = mean
        self.std = std
        self.in_dim = cfg.l+cfg.pred
        self.seg_len = cfg.seg_len

        self.configs = {'seq_len': self.in_dim,
                   'pred_len': self.hidden_dim,
                   'enc_in': 1,
                   'd_model': 512,
                   'dropout': 0,
                   'rnn_type': 'gru',
                   'dec_way': 'pmf',
                   'seg_len': self.seg_len,
                   'channel_id': 0,
                   'revin': 1}
        self.segRNN = SegRNN(self.configs)

        for param in self.segRNN.parameters():
            param.requires_grad = True

        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_chan, h_0, c_0):
        '''
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        '''
        x = in_chan
        x = (x - self.mean) / self.std

        x = x.permute(1, 2, 0)
        h_n = self.segRNN(x)
        h_n = h_n.permute(2, 0, 1)
        out = self.linear(h_n)
        out = self.sigmoid(out)
        return out