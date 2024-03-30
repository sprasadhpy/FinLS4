import torch
import torch.nn as nn
from Utils.combine_vectors import combine_vectors
from models.TLN import F_SVD_Net, FFT_Conv_Net, Conv_SVD_Net, FT_matrix_Net




# LSTM ForGAN generator
class GeneratorWithTLN(nn.Module):
    '''
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, cfg, mean, std):
        super(GeneratorWithTLN, self).__init__()
        self.input_dim = cfg.z_dim + cfg.l
        self.cond_dim = cfg.l
        self.hidden_dim = cfg.hid_g
        self.output_dim = cfg.pred
        self.noise_dim = cfg.z_dim
        # predicting a single value, so the output dimension is 1
        self.mean = mean
        self.std = std
        # Add the modules

        # print(cfg.model, 'Inside GeneratorWithTLN')

        if cfg.model == "ForGAN-FT-Matrix":
            self.tln = FT_matrix_Net(self.cond_dim, self.hidden_dim, 1)
        elif cfg.model == "ForGAN-F-SVD":
            self.tln = F_SVD_Net(self.cond_dim, self.hidden_dim, 1)
        elif cfg.model == "ForGAN-FFT-Conv":
            self.tln = FFT_Conv_Net(self.cond_dim, self.hidden_dim, 1)
        elif cfg.model == "ForGAN-Conv-SVD":
            self.tln = Conv_SVD_Net(self.cond_dim, self.hidden_dim, 1)


        self.linear1 = nn.Linear(in_features=self.hidden_dim + self.noise_dim,
                                 out_features=self.hidden_dim + self.noise_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(in_features=self.hidden_dim + self.noise_dim, out_features=self.output_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.ReLU()

    def forward(self, noise, condition, h_0, c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''
        # x = combine_vectors(noise.to(torch.float),condition.to(torch.float),2)
        condition = (condition - self.mean) / self.std
        condition = condition.permute(1, 2, 0)
        out = self.tln(condition)
        out = out.permute(2, 0, 1)

        out = combine_vectors(noise.to(torch.float), out.to(torch.float), dim=-1)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        out = out * self.std + self.mean
        return out

