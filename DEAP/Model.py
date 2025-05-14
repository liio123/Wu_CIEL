import math

import torch.nn.functional as F
import torch.nn as nn
import torch
# from autograd import grad
from layers import GraphConvolution,Linear
from utils import *
from sympy import series, sin, cos, exp, ln
import sympy

class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        self.dp = nn.Dropout(dropout)
        self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                # print(result.shape)
                # print(x.shape)
                # print(adj[i].shape)
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        #result = self.dp(result)
        return result

class Attentionadj(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attentionadj, self).__init__()

        self.project = nn.Sequential(
            #nn.BatchNorm2d(3),
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

# 位置编码
# class PositionalEncoding(nn.Module):
#     def __init__(self,d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         """x:[seq_len, batch_size, d_model]"""
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

# # 时间特征提取模块
# class TimeExtraction(nn.Module):
#     def __init__(self, input_dim=5, nhead=2, d_model=32, num_classes=3, dropout=0.5):
#         super(TimeExtraction, self).__init__()
#         self.input_dim = input_dim
#         self.nhead = nhead
#         self.d_model = d_model
#         self.num_classes = num_classes
#         self.dropout = nn.Dropout(p=dropout)
#
#         self.positionalEncoding = PositionalEncoding(d_model=d_model)
#
#         # TransformerEncoderLayer with self-attention
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, dim_feedforward=1024, nhead=nhead, dropout=dropout
#         )
#     # Classification layer
#     # self.pred_layer = nn.Sequential(
#     #     nn.Linear(d_model, d_model),
#     #     nn.Dropout(dropout),
#     #     nn.ReLU(),
#     #     nn.Linear(d_model, num_classes)
#     # )
#
#     def forward(self, x):
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = x.permute(2, 0, 1)
#         # Positional Encoding layer
#         out = self.positionalEncoding(x)
#         # Transformer Encoder layer layer
#         out = self.encoder_layer(out)
#         out = out.permute(1, 2, 0)
#         return out

def sin_taylor_optimized(x: torch.Tensor, n_terms: int = 5) -> torch.Tensor:
    """
    优化后的泰勒展开计算 sin(x)
    """
    result = torch.zeros_like(x)
    term = x.clone()  # 初始项: x^1 / 1!
    for n in range(1, n_terms + 1):
        result += term
        # 递推计算下一项: (-1) * x^2 / ((2n)(2n+1)) * previous_term
        term = -term * x ** 2 / (2 * n * (2 * n + 1))
    return result

def exp_taylor_optimized(x: torch.Tensor, n_terms: int = 10) -> torch.Tensor:
    """
    优化后的泰勒展开计算 exp(x)
    """
    result = torch.ones_like(x)
    term = torch.ones_like(x)  # 初始项: x^0 / 0! = 1

    for n in range(1, n_terms):
        term = term * x / n  # 递推计算下一项: x^n / n! = (x^{n-1} / (n-1)!) * (x / n)
        result += term

    return result

def ln_taylor_optimized(x: torch.Tensor, n_terms: int = 10) -> torch.Tensor:
    """
    优化后的泰勒展开计算 ln(1 + x)
    """
    # if torch.any(torch.abs(x) >= 1):
    #     raise ValueError("泰勒展开要求 |x| < 1")

    result = torch.zeros_like(x)
    term = x.clone()  # 初始项: x^1 / 1

    for n in range(1, n_terms + 1):
        result += term
        term = -term * x * n / (n + 1)  # 递推计算下一项: (-1) * x * (n / (n+1)) * previous_term

    return result

class PINN_VMC(nn.Module):
    def __init__(self, hidden_dim, output_dim, xdim,kadj,num_out, dropout):
        super(PINN_VMC, self).__init__()
        # self.spconv = SeparableConv2d(32, 32)
        self.BN1 = nn.BatchNorm1d(5)
        self.GCN = Chebynet(xdim, kadj, num_out, dropout)
        self.attentionadj = Attentionadj(32)
        self.A = nn.Parameter(torch.FloatTensor(32, 32).cuda())
        self.A = nn.init.kaiming_normal(self.A, )
        # self.temporal = TimeExtraction()

        self.fc1 = nn.Linear(32*16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, fadj):
        # x = x.clone().detach().requires_grad_(True)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        # x_time = self.temporal(x)

        # x = self.spconv(x)
        # fadj = fadj.permute(0, 3, 1, 2)
        fadj, att = self.attentionadj(fadj)
        # fadj = torch.sign(fadj) * torch.maximum(abs(fadj) - 1 / 2 * 0.5 *0.2, torch.zeros_like(fadj))
        fadj = normalize_A(fadj, symmetry=False, gaowei=True)
        sadj = normalize_A(self.A, symmetry=False, gaowei=False)

        sin_fadj = sin_taylor_optimized(fadj, n_terms=5)  # 5阶泰勒展开近似
        exp_fadj = exp_taylor_optimized(fadj, n_terms=5)
        ln_fadj = ln_taylor_optimized(fadj, n_terms=5)
        sin_sadj = sin_taylor_optimized(sadj, n_terms=5)  # 5阶泰勒展开近似
        exp_sadj = exp_taylor_optimized(sadj, n_terms=5)
        ln_sadj = ln_taylor_optimized(sadj, n_terms=5)

        fadj = 0.3 * sin_fadj + 0.3 * exp_fadj + 0.4 * ln_fadj
        sadj = 0.3 * sin_sadj + 0.3 * exp_sadj + 0.4 * ln_sadj

        x_fadj = self.GCN(x, fadj)
        x_sadj = self.GCN(x, sadj)

        feature = (x_fadj + x_sadj) / 2
        output = feature.reshape(-1, 32*16)

        # output = torch.cat([feature.reshape(-1, 32*16), x_time.reshape(-1, 32*5)], dim=1)

        output = self.activation(self.fc1(output))
        output = self.activation(self.fc2(output))
        output = self.fc3(output)
        return output
