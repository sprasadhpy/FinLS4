import torch
import torch.nn as nn

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
      hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, cfg, mean, std):
        super(Discriminator, self).__init__()
        self.hidden_dim = cfg.hid_d
        self.mean = mean
        self.std = std
        self.lstm = nn.LSTM(input_size=cfg.l+cfg.pred, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
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
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.linear(h_n)
        out = self.sigmoid(out)
        return out