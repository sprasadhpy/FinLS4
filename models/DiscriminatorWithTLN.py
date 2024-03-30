import torch
import torch.nn as nn
from models.TLN import F_SVD_Net, FFT_Conv_Net, Conv_SVD_Net, FT_matrix_Net

class DiscriminatorWithTLN(nn.Module):
    '''
        Discriminator Class
        Values:
          in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
          hidden_dim: the inner dimension, a scalar
        '''

    def __init__(self,  cfg, mean, std):
        super(DiscriminatorWithTLN, self).__init__()
        self.hidden_dim = cfg.hid_d
        self.mean = mean
        self.std = std
        self.input_dim = cfg.l + cfg.pred
        if cfg.model == "ForGAN-FT-Matrix":
            self.tln = FT_matrix_Net(self.input_dim, self.hidden_dim, 1)
        elif cfg.model == "ForGAN-F-SVD":
            self.tln = F_SVD_Net(self.input_dim, self.hidden_dim, 1)
        elif cfg.model == "ForGAN-FFT-Conv":
            self.tln = FFT_Conv_Net(self.input_dim, self.hidden_dim, 1)
        elif cfg.model == "ForGAN-Conv-SVD":
            self.tln = Conv_SVD_Net(self.input_dim, self.hidden_dim, 1)


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
        x = self.tln(x)
        x = x.permute(2, 0, 1)
        out = self.linear(x)
        out = self.sigmoid(out)
        return out