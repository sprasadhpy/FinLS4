import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import torch
import torch.nn.functional as F


class Basisformer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, heads, basis_nums, block_nums, bottle, map_bottleneck, device, tau):
        super().__init__()
        self.d_model = d_model
        self.k = heads
        self.N = basis_nums
        self.coefnet = Coefnet(blocks=block_nums, d_model=d_model, heads=heads)

        self.pred_len = pred_len
        self.seq_len = seq_len

        self.MLP_x = MLP_bottle(seq_len, heads * int(seq_len / heads), int(seq_len / bottle))
        self.MLP_y = MLP_bottle(pred_len, heads * int(pred_len / heads), int(pred_len / bottle))
        self.MLP_sx = MLP_bottle(heads * int(seq_len / heads), seq_len, int(seq_len / bottle))
        self.MLP_sy = MLP_bottle(heads * int(pred_len / heads), pred_len, int(pred_len / bottle))

        self.project1 = wn(nn.Linear(seq_len, d_model))
        self.project2 = wn(nn.Linear(seq_len, d_model))
        self.project3 = wn(nn.Linear(pred_len, d_model))
        self.project4 = wn(nn.Linear(pred_len, d_model))
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss(reduction='none')

        self.device = device

        # smooth array
        arr = torch.zeros((seq_len + pred_len - 2, seq_len + pred_len))
        for i in range(seq_len + pred_len - 2):
            arr[i, i] = -1
            arr[i, i + 1] = 2
            arr[i, i + 2] = -1
        self.smooth_arr = arr.to(device)
        self.map_MLP = MLP_bottle(1, self.N * (self.seq_len + self.pred_len), map_bottleneck, bias=True)
        self.tau = tau
        self.epsilon = 1E-5

    def forward(self, x, mark, y=None, train=True, y_mark=None):
        mean_x = x.mean(dim=1, keepdim=True)
        std_x = x.std(dim=1, keepdim=True)
        feature = (x - mean_x) / (std_x + self.epsilon)
        B, L, C = feature.shape
        feature = feature.permute(0, 2, 1)
        feature = self.project1(feature)  # (B,C,d)

        m = self.map_MLP(mark[:, 0].unsqueeze(1)).reshape(B, self.seq_len + self.pred_len, self.N)
        m = m / torch.sqrt(torch.sum(m ** 2, dim=1, keepdim=True) + self.epsilon)

        raw_m1 = m[:, :self.seq_len].permute(0, 2, 1)  # (B,L,N)
        raw_m2 = m[:, self.seq_len:].permute(0, 2, 1)  # (B,L',N)
        m1 = self.project2(raw_m1)  # (B,N,d)

        score, attn_x1, attn_x2 = self.coefnet(m1, feature)  # (B,k,C,N)

        base = self.MLP_y(raw_m2).reshape(B, self.N, self.k, -1).permute(0, 2, 1, 3)  # (B,k,N,L/k)
        out = torch.matmul(score, base).permute(0, 2, 1, 3).reshape(B, C, -1)  # (B,C,k * (L/k))
        out = self.MLP_sy(out).reshape(B, C, -1).permute(0, 2, 1)  # （BC,L）

        output = out * (std_x + self.epsilon) + mean_x

        # loss
        if train:
            l_smooth = torch.einsum('xl,bln->xbn', self.smooth_arr, m)
            l_smooth = abs(l_smooth).mean()
            # l_smooth = self.criterion1(l_smooth,torch.zeros_like(l_smooth))

            # #back
            mean_y = y.mean(dim=1, keepdim=True)
            std_y = y.std(dim=1, keepdim=True)
            feature_y_raw = (y - mean_y) / (std_y + self.epsilon)

            feature_y = feature_y_raw.permute(0, 2, 1)
            feature_y = self.project3(feature_y)  # (BC,d)
            m2 = self.project4(raw_m2)  # (N,d)

            score_y, attn_y1, attn_y2 = self.coefnet(m2, feature_y)  # (B,k,C,N)
            logit_q = score.permute(0, 2, 3, 1)  # (B,C,N,k)
            logit_k = score_y.permute(0, 2, 3, 1)  # (B,C,N,k)

            # l_pos = torch.bmm(logit_q.view(-1,1,self.k), logit_k.view(-1,self.k,1)).reshape(-1,1)  #(B*C*N,1,1)
            l_neg = torch.bmm(logit_q.reshape(-1, self.N, self.k),
                              logit_k.reshape(-1, self.N, self.k).permute(0, 2, 1)).reshape(-1, self.N)  # (B,C*N,N)

            labels = torch.arange(0, self.N, 1, dtype=torch.long).unsqueeze(0).repeat(B * C, 1).reshape(-1)

            labels = labels.to(self.device)

            cross_entropy_loss = nn.CrossEntropyLoss()
            l_entropy = cross_entropy_loss(l_neg / self.tau, labels)

            return output, l_entropy, l_smooth, attn_x1, attn_x2, attn_y1, attn_y2
        else:
            # #back
            mean_y = y.mean(dim=1, keepdim=True)
            std_y = y.std(dim=1, keepdim=True)
            feature_y_raw = (y - mean_y) / (std_y + self.epsilon)

            feature_y = feature_y_raw.permute(0, 2, 1)
            feature_y = self.project3(feature_y)  # (BC,d)
            m2 = self.project4(raw_m2)  # (N,d)

            score_y, attn_y1, attn_y2 = self.coefnet(m2, feature_y)  # (B,k,C,N)
            return output, m, attn_x1, attn_x2, attn_y1, attn_y2


class MLP(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, output_len)),
            nn.ReLU(),
            wn(nn.Linear(output_len, output_len))
        )

        self.linear2 = nn.Sequential(
            wn(nn.Linear(output_len, output_len)),
            nn.ReLU(),
            wn(nn.Linear(output_len, output_len))
        )

        self.skip = wn(nn.Linear(input_len, output_len))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x) + self.skip(x))
        x = self.linear2(x)

        return x


class MLP_bottle(nn.Module):
    def __init__(self, input_len, output_len, bottleneck, bias=True):
        super().__init__()
        self.linear1 = nn.Sequential(
            wn(nn.Linear(input_len, bottleneck, bias=bias)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck, bottleneck, bias=bias))
        )

        self.linear2 = nn.Sequential(
            wn(nn.Linear(bottleneck, bottleneck)),
            nn.ReLU(),
            wn(nn.Linear(bottleneck, output_len))
        )

        self.skip = wn(nn.Linear(input_len, bottleneck, bias=bias))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear1(x) + self.skip(x))
        x = self.linear2(x)

        return x


class Coefnet(nn.Module):
    def __init__(self, blocks, d_model, heads, norm_layer=None, projection=None):
        super().__init__()
        layers = [BCAB(d_model, heads) for i in range(blocks)]
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        # heads = heads if blocks > 0 else 1
        self.last_layer = last_layer(d_model, heads)

    def forward(self, basis, series):
        attns1 = []
        attns2 = []
        for layer in self.layers:
            basis, series, basis_attn, series_attn = layer(basis, series)  # basis(B,N,d)  series(B,C,d)
            attns1.append(basis_attn)
            attns2.append(series_attn)

        coef = self.last_layer(series, basis)  # (B,k,C,N)

        return coef, attns1, attns2


class BCAB(nn.Module):
    def __init__(self, d_model, heads=8, index=0, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_basis = channel_AutoCorrelationLayer(d_model, heads, dropout=dropout)
        self.conv1_basis = wn(nn.Linear(d_model, d_ff))
        self.conv2_basis = wn(nn.Linear(d_ff, d_model))

        self.dropout_basis = nn.Dropout(dropout)
        self.activation_basis = F.relu if activation == "relu" else F.gelu

        self.cross_attention_ts = channel_AutoCorrelationLayer(d_model, heads, dropout=dropout)
        self.conv1_ts = wn(nn.Linear(d_model, d_ff))
        self.conv2_ts = wn(nn.Linear(d_ff, d_model))

        self.dropout_ts = nn.Dropout(dropout)
        self.activation_ts = F.relu if activation == "relu" else F.gelu
        self.layer_norm11 = nn.LayerNorm(d_model)
        self.layer_norm12 = nn.LayerNorm(d_model)
        self.layer_norm21 = nn.LayerNorm(d_model)
        self.layer_norm22 = nn.LayerNorm(d_model)

    def forward(self, basis, series):
        basis_raw = basis
        series_raw = series
        basis_add, basis_attn = self.cross_attention_basis(
            basis_raw, series_raw, series_raw,
        )
        basis_out = basis_raw + self.dropout_basis(basis_add)
        basis_out = self.layer_norm11(basis_out)

        y_basis = basis_out
        y_basis = self.dropout_basis(self.activation_basis(self.conv1_basis(y_basis)))
        y_basis = self.dropout_basis(self.conv2_basis(y_basis))
        basis_out = basis_out + y_basis

        basis_out = self.layer_norm12(basis_out)

        series_add, series_attn = self.cross_attention_ts(
            series_raw, basis_raw, basis_raw
        )
        series_out = series_raw + self.dropout_ts(series_add)

        series_out = self.layer_norm21(series_out)

        y_ts = series_out
        y_ts = self.dropout_ts(self.activation_ts(self.conv1_ts(y_ts)))
        y_ts = self.dropout_ts(self.conv2_ts(y_ts))
        series_out = series_out + y_ts
        series_out = series_raw

        series_out = self.layer_norm22(series_out)

        return basis_out, series_out, basis_attn, series_attn


class channel_AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, n_heads, mask=False, d_keys=None,
                 d_values=None, dropout=0):
        super().__init__()

        self.mask = mask

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.value_projection = wn(nn.Linear(d_model, d_values * n_heads))
        self.out_projection = wn(nn.Linear(d_values * n_heads, d_model))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        num = len(queries.shape)
        if num == 2:
            L, _ = queries.shape
            S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(L, H, -1).permute(1, 0, 2)
            keys = self.key_projection(keys).view(S, H, -1).permute(1, 0, 2)
            values = self.value_projection(values).view(S, H, -1).permute(1, 0, 2)
            # queries = queries.view(L, H, -1).permute(1,0,2)
            # keys = keys.view(S, H, -1).permute(1,0,2)
            # values = values.view(S, H, -1).permute(1,0,2)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, values)  # (H,L,D)

            out = out.permute(1, 0, 2).reshape(L, -1)
        else:
            B, L, _ = queries.shape
            B, S, _ = keys.shape
            H = self.n_heads

            queries = self.query_projection(queries).view(B, L, H, -1).permute(0, 2, 1, 3)
            keys = self.key_projection(keys).view(B, S, H, -1).permute(0, 2, 1, 3)
            values = self.value_projection(values).view(B, S, H, -1).permute(0, 2, 1, 3)

            dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)

            attn = self.dropout(attn)

            out = torch.matmul(attn, values)  # (H,L,D)

            out = out.permute(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_projection(out), attn


class last_layer(nn.Module):
    def __init__(self, d_model, n_heads, mask=False, d_keys=None,
                 d_values=None, dropout=0):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.key_projection = wn(nn.Linear(d_model, d_keys * n_heads))
        self.n_heads = n_heads
        self.scale = d_keys ** -0.5

    def forward(self, queries, keys):
        B, L, _ = queries.shape
        B, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1).permute(0, 2, 1, 3)
        keys = self.key_projection(keys).view(B, S, H, -1).permute(0, 2, 1, 3)

        dots = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale  # (B,H,L,S)

        return dots

