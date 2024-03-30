import torch
import torch.nn as nn
from Basisformer import Basisformer

class DiscriminatorWithBasisformer(nn.Module):
    '''
        Discriminator Class
        Values:
          in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
          hidden_dim: the inner dimension, a scalar
        '''

    def __init__(self, in_dim, hidden_dim, mean, std, batch_size, seg_len):
        super(DiscriminatorWithBasisformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.mean = mean
        self.std = std
        self.basis = Basisformer(seq_len=in_dim, pred_len=hidden_dim, d_model=512, heads=8, basis_nums=8, block_nums=4,
                                 bottle=4, map_bottleneck=4, device='cuda', tau=0.1)
        # nn.init.xavier_normal_(self.basis.weight_ih_l0)
        # nn.init.xavier_normal_(self.basis.weight_hh_l0)
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


        # x = x.permute(1, 2, 0)
        h_n = self.basis(x)
        # h_n = h_n.permute(2, 0, 1)
        out = self.linear(h_n)
        out = self.sigmoid(out)
        return out