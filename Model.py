import torch.nn.functional as F
import torch.nn as nn
import torch
# from autograd import grad
from layers import GraphConvolution,Linear
from utils import *


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
    def __init__(self, in_size, hidden_size=62):
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
        self.BN1 = nn.BatchNorm1d(5)
        self.GCN = Chebynet(xdim, kadj, num_out, dropout)
        self.attentionadj = Attentionadj(62)
        self.A = nn.Parameter(torch.FloatTensor(62, 62).cuda())
        self.A = nn.init.kaiming_normal(self.A, )

        self.fc1 = nn.Linear(62*16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, fadj):
        # x = x.clone().detach().requires_grad_(True)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        # x_time = self.temporal(x)

        # x = self.spconv(x)
        fadj = fadj.permute(0, 3, 1, 2)
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
        output = feature.reshape(-1, 62 * 16)

        # output = torch.cat([feature.reshape(-1, 32*16), x_time.reshape(-1, 32*5)], dim=1)

        output = self.activation(self.fc1(output))
        output = self.activation(self.fc2(output))
        output = self.fc3(output)
        return output
