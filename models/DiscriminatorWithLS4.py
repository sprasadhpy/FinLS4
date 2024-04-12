import torch
import torch.nn as nn
from models.SegRNN import SegRNN

class DiscriminatorWithLS4(nn.Module):
    '''
    Discriminator Class
    Values:
      in_dim: the input dimension (noise dim + conditin dim + forecast dim for the condition for this dataset), a scalar
      hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self,cfg, mean, std, ls4_encoder_model):
        super(DiscriminatorWithLS4, self).__init__()
        self.hidden_dim = cfg.hid_d
        self.mean = mean
        self.std = std
        self.in_dim = cfg.l+cfg.pred
        self.seg_len = cfg.seg_len
        self.device = cfg.device

        '''
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
        '''

        #encode the seq using ls4
        self.ls4_encoder_model = ls4_encoder_model.model
        
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=1)
        nn.init.xavier_normal_(self.linear.weight)
        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def encode(self, data):

        self.ls4_encoder_model.eval()
        
        if isinstance(self.ls4_encoder_model, torch.nn.DataParallel) or isinstance(self.ls4_encoder_model, torch.optim.swa_utils.AveragedModel):
            self.ls4_encoder_model.module.setup_rnn()
        else:
            self.ls4_encoder_model.setup_rnn()

        #print('data',data.shape) #data torch.Size([64, 21, 1])
        mask = data[:,-1,:].unsqueeze(1)
        #print('mask',mask.shape) #mask torch.Size([64, 1, 1])

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
        return z#, z_mean, z_std

    def forward(self, in_chan, h_0, c_0):
        '''
        in_chan: concatenated condition with real or fake
        h_0 and c_0: for the LSTM
        '''
        x = in_chan
        #x = (x - self.mean) / self.std
        
        x = x.permute(1, 2, 0)
        #h_n = self.segRNN(x)
        h_n = self.encode(x)
        #print(h_n.shape)# torch.Size([64, 5, 1])
        h_n = h_n.permute(2, 0, 1)
        #print(h_n.shape)# torch.Size([1, 64, 5])
        out = self.linear(h_n)
        #print(out.shape)# torch.Size([1, 64, 1])
        out = self.sigmoid(out)
        #print(out.shape)# torch.Size([1, 64, 1])
        return out