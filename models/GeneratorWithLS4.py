import torch
import torch.nn as nn
#from models.SegRNN import SegRNN
from Utils.combine_vectors import combine_vectors
from models.HiddenStateSpaceLS4 import HiddenStateSpaceLS4
from models.HiddenStateSpaceLS4 import get_hidden_states

class GeneratorWithLS4(nn.Module):
    '''
    Generator Class
    Values:
        noise_dim: the dimension of the noise, a scalar
        cond_dim: the dimension of the condition, a scalar
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, cfg, mean, std, ls4_encoder_model):
        super(GeneratorWithLS4, self).__init__()
        self.input_dim = cfg.z_dim + cfg.l
        self.cond_dim = cfg.l #TODO :: make config changes in ls4
        self.hidden_dim = cfg.hid_g
        self.output_dim = cfg.pred
        self.noise_dim = cfg.z_dim
        self.mean = mean
        self.std = std
        self.seg_len = cfg.seg_len

        self.device = cfg.device
        
        self.ls4_encoder_model = ls4_encoder_model.model#nn.LSTM(input_size=self.cond_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
        #nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        #nn.init.xavier_normal_(self.lstm.weight_hh_l0)
        self.linear1 = nn.Linear(in_features=self.hidden_dim + self.noise_dim,
                                 out_features=self.hidden_dim + self.noise_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(in_features=self.hidden_dim + self.noise_dim, out_features=self.output_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.ReLU()

        for param in self.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def encode(self, data):

        self.ls4_encoder_model.eval()
        
        if isinstance(self.ls4_encoder_model, torch.nn.DataParallel) or isinstance(self.ls4_encoder_model, torch.optim.swa_utils.AveragedModel):
            self.ls4_encoder_model.module.setup_rnn()
        else:
            self.ls4_encoder_model.setup_rnn()

        #data shape : [1, 64, 20]
        data = data.permute(1,2,0)
        mask = data[:,-1,:].unsqueeze(1)
        #if forgan_part_type == 'generator':
        #data = data[:, :-1]
        #data = data.unsqueeze(2)
        mask = mask.to(self.device)
        data = data.to(self.device)
        #print(data.shape, mask.shape)

        #decode = lambda x: x
        #decode_func = decode # more details here https://github.com/alexzhou907/ls4/blob/2b5cb76f89f6b935bbd697771433f3c74126ffca/datasets/__init__.py#L46
        
        # Perform the encoding operation
        z, z_mean, z_std = self.ls4_encoder_model.encode(data, mask)
        #z is a seq of hidden state
        #we take last one
        z = z[:,-1,:].unsqueeze(2)
        #print(z.shape)
        #should be (batch X hidd_dim X 1)
        return z#, z_mean, z_std
        
    def forward(self, noise, condition, h_0, c_0):
        '''
        Function for completing a forward pass of the generator:adding the noise and the condition separately
        '''

        #h_states = get_hidden_states(model, device, -1, testloader, decode_func, log_dir=log_dir, savenpy=True, return_preds=True)
        #no norm in ls4 model, handled by encoder module itself
        #condition = (condition - self.mean) / self.std
        h_n = self.encode(condition)
        #condition = condition.permute(1, 2, 0)
        #h_n = self.segRNN(condition)
        #print(h_n.shape)
        h_n = h_n.permute(2, 0, 1)
        #print(h_n.shape)
        out = combine_vectors(noise.to(torch.float), h_n.to(torch.float), dim=-1)
        #print(out.shape)
        out = self.linear1(out)
        #print(out.shape)
        out = self.activation(out)
        #print(out.shape)
        out = self.linear2(out)
        #print(out.shape)
        '''
        torch.Size([64, 5, 1])
        torch.Size([1, 64, 5])
        torch.Size([1, 64, 25])
        torch.Size([1, 64, 25])
        torch.Size([1, 64, 25])
        torch.Size([1, 64, 1])
        '''
        out = out * self.std + self.mean
        return out