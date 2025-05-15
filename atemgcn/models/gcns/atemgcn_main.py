import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.graph import Graph
from ..builder import BACKBONES
from .utils import MSGCN, MSTCN, MW_MSG3DBlock

class ATEM(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0), groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones((dim)), requires_grad=True)

        self.drop_path = nn.Identity()  # Optional: replace with DropPath if needed

    def forward(self, x):
        shortcut = x  # x: [N, C, T, V]

        x = self.dwconv(x)  # depthwise conv on (T, V)
        x = x.permute(0, 2, 3, 1)  # [N, T, V, C] for LayerNorm

        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to [N, C, T, V]

        return shortcut + self.drop_path(x)


@BACKBONES.register_module()
class MSG3D(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 num_gcn_scales=13,
                 num_g3d_scales=6,
                 num_person=2,
                 tcn_dropout=0):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        # Note that A is a 2D tensor
        A = torch.tensor(self.graph.A[0], dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_point = A.shape[-1]
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(self.num_point * in_channels * num_person)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.gcn3d1 = MW_MSG3DBlock(3, c1, A, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MSGCN(num_gcn_scales, 3, c1, A),
            ATEM(c1, c1),
            ATEM(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = ATEM(c1, c1)

        self.gcn3d2 = MW_MSG3DBlock(c1, c2, A, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MSGCN(num_gcn_scales, c1, c1, A),
            ATEM(c1, c2, stride=2),
            ATEM(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = ATEM(c2, c2)

        self.gcn3d3 = MW_MSG3DBlock(c2, c3, A, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MSGCN(num_gcn_scales, c2, c2, A),
            ATEM(c2, c3),
            ATEM(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = ATEM(c3, c3)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous().reshape(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.reshape(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        # N * M, C, T, V
        return x.reshape((N, M) + x.shape[1:])

    def init_weights(self):
        pass
