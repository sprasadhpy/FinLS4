import torch
import torch.nn as nn
from models.SegRNN import SegRNN
from Utils.combine_vectors import combine_vectors

class GeneratorWithSegRNN(nn.Module):
    '''
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self,cfg, mean, std):
        super(GeneratorWithSegRNN, self).__init__()
        self.input_dim = cfg.z_dim + cfg.l
        self.cond_dim = cfg.l
        self.hidden_dim = cfg.hid_g
        self.output_dim = cfg.pred
        self.noise_dim = cfg.z_dim
        self.mean = mean
        self.std = std
        self.seg_len = cfg.seg_len

        # Add the modules

        self.configs = {'seq_len': self.cond_dim,
                        'pred_len': self.hidden_dim,
                        'enc_in': 1,
                        'd_model': 512,
                        'dropout': 0,
                        'rnn_type': 'gru',
                        'dec_way': 'pmf',
                        'seg_len': self.seg_len,
                        'channel_id': 0,
                        'revin': 1
                        }
        self.segRNN = SegRNN(self.configs)

        self.lstm = nn.LSTM(input_size=self.cond_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear1 = nn.Linear(in_features=self.hidden_dim + self.noise_dim,
                                 out_features=self.hidden_dim + self.noise_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(in_features=self.hidden_dim + self.noise_dim, out_features=self.output_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.ReLU()

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, noise, condition, h_0, c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''


        condition = (condition - self.mean) / self.std
        condition = condition.permute(1, 2, 0)
        h_n = self.segRNN(condition)
        h_n = h_n.permute(2, 0, 1)
        out = combine_vectors(noise.to(torch.float), h_n.to(torch.float), dim=-1)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = out * self.std + self.mean
        return out