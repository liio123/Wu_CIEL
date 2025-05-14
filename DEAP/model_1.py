import torch
from torch import nn
import torch.nn.functional as F
from layers import GraphConvolution,Linear
from utils import *
import warnings
warnings.filterwarnings('ignore')

channel_num = 32


def corrcoef(x):
    """传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵"""
    f = (x.shape[0] - 1) / x.shape[0]  # 方差调整系数
    x_reducemean = x - torch.mean(x, axis=0)
    numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
    var_ = x.var(axis=0).reshape(x.shape[1], 1)
    denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
    corrcoef = numerator / denominator
    return corrcoef

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积：groups=in_channels=out_channels
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1,
                                        padding=1, groups=in_channels, bias=bias)
        # 逐点卷积：普通1x1卷积
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x_raw = x.to(torch.float32)   # ([128, 16, 5, 8])
        # x_raw = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.depthwise_conv(x)  # torch.Size([128, 16, 5])
        # print(x.shape)
        x = self.pointwise_conv(x)  # torch.Size([128, 16, 5])
        x = self.bn(x)
        x = F.relu(x)

        return x

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


class Model(nn.Module):
    def __init__(self, xdim, kadj, num_out, dropout):
        super(Model, self).__init__()

        self.spconv = SeparableConv2d(channel_num, channel_num)
        self.GCN = Chebynet(xdim, kadj, num_out, dropout)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.FloatTensor(4).cuda())

        self.fc62 = nn.Sequential(
            # nn.BatchNorm1d(248),
            nn.Linear(channel_num * 16, channel_num),
            nn.BatchNorm1d(channel_num),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )

        self.fcclass = nn.Linear(channel_num, 4)
        self.rule = nn.Linear(channel_num, 4)


    def forward(self, de, adj):
        # 可分离卷积
        de = self.spconv(de).permute(0, 2, 1)    # torch.Size([128, 16, 5])
        # print(de.shape)
        pcc_list = []
        for i in range(de.shape[0]):
            # 基于频带和通道计算相关性系数
            pcc = corrcoef(de[i])   # torch.Size([16, 16])
            pcc_list.append(pcc)
        cor = torch.stack(pcc_list)
        # print(cor.shape)
        # 图卷积神经网络
        g = self.GCN(de.permute(0, 2, 1), cor)
        # print(g.shape)
        output1 = self.fc62(g.reshape(g.shape[0], -1))
        output = self.fcclass(output1)

        return output
