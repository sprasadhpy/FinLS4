import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class SegRNN(nn.Module):
    def __init__(self, configs):
        super(SegRNN, self).__init__()

        # get parameters
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.d_model = configs['d_model']
        self.dropout = configs['dropout']

        self.rnn_type = configs['rnn_type']
        self.dec_way = configs['dec_way']
        self.seg_len = configs['seg_len']
        self.channel_id = configs['channel_id']
        self.revin = configs['revin']

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way in ['rmf', 'pmf']

        self.seg_num_x = self.seq_len//self.seg_len

        # build model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        if self.dec_way == "rmf":
            self.seg_num_y = self.pred_len // self.seg_len
            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        elif self.dec_way == "pmf":
            self.seg_num_y = self.pred_len // self.seg_len

            if self.channel_id:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
                self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
            else:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)


    def forward(self, x):

        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        if self.revin:
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1) # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.view(-1, self.seg_num_x, self.seg_len))

        # encoding
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x) # bc,n,d  1,bc,d

        # decoding
        if self.dec_way == "rmf":
            y = []
            for i in range(self.seg_num_y):
                yy = self.predict(hn)    # 1,bc,l
                yy = yy.permute(1,0,2)   # bc,1,l
                y.append(yy)
                yy = self.valueEmbedding(yy)
                if self.rnn_type == "lstm":
                    _, (hn, cn) = self.rnn(yy, (hn, cn))
                else:
                    _, hn = self.rnn(yy, hn)
            y = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len) # b,c,s
        elif self.dec_way == "pmf":
            if self.channel_id:
                # m,d//2 -> 1,m,d//2 -> c,m,d//2
                # c,d//2 -> c,1,d//2 -> c,m,d//2
                # c,m,d -> cm,1,d -> bcm, 1, d
                pos_emb = torch.cat([
                    self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
                ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
            else:
                # m,d -> bcm,d -> bcm, 1, d
                pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

            # pos_emb: m,d -> bcm,d ->  bcm,1,d
            # hn, cn: 1,bc,d -> 1,bc,md -> 1,bcm,d
            if self.rnn_type == "lstm":
                _, (hy, cy) = self.rnn(pos_emb,
                                       (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                        cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
            else:
                _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
            # 1,bcm,d -> 1,bcm,w -> b,c,s
            y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        if self.revin:
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            y = y.permute(0, 2, 1) + seq_last

        return y