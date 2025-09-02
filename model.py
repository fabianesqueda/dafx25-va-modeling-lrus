import math
import torch
import torch.nn as nn

# pip install mambapy
# Source: https://github.com/alxndrTL/mamba.py
from mambapy.pscan import pscan

class TanhApprox(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(1 + x**2)


class Model(nn.Module):
    def __init__(  self
                 , input_channels=1
                 , output_channels=1
                 , N=8
                 , H=8
                 , depth=4):
        super().__init__()

        self.N = N
        self.H = H

        self.input_layer = nn.Linear(input_channels, self.H, bias=False)

        self.model_blocks = nn.ModuleList( [ModelBlock(  N=self.N
                                                       , H=self.H)
                                                    for _ in range(depth)] )

        self.output_layer = nn.Linear(self.H, output_channels, bias=False)


    def forward(self, x):

        # Map input to hidden dimension H
        x = self.input_layer(x)

        # Iterate over model blocks. Each block consists of LRU + NL w/skip.
        for lru_block in self.model_blocks:
            x = lru_block(x)

        # Map output back to original dimension
        return self.output_layer(x)


# ---------------------------------------------------------------------------------- #

class ModelBlock(nn.Module):
    def __init__(  self
                 , N
                 , H):
        super().__init__()

        self.lru = LRUBlock(  N=N
                            , H=H)

        self.nonlinear_block = nn.Sequential(  TanhApprox()
                                             , nn.Linear(H, H))

    def forward(self, x):

        skip = x.clone()
        y = self.lru(x.unsqueeze(-2)).squeeze(-2)
        y = self.nonlinear_block(y)

        return y + skip


# ----------------------- #

class LRUBlock(nn.Module):
    def __init__(self, N, H):
        super().__init__()

        r_min = 0.8
        r_max = 1.0

        # Nu log
        u = torch.rand(1, 1, 1, N)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u * (r_max**2 - r_min**2) + r_min**2)))
        self.nu_log._no_weight_decay = True

        # Gamma log
        A = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones(A.shape) - torch.abs(A) ** 2)))

        # B
        self.B = nn.Parameter(torch.randn(1, 1, N, H) / math.sqrt(2.0 * H))
        self.B._no_weight_decay = True

        # C
        self.C = nn.Parameter(torch.randn(1, 1, H, N) / math.sqrt(N))

        # D
        self.D = nn.Parameter(torch.randn(1, 1, H, 1)) 

    def scan(self, A, x):
        Ax = pscan(A, x)
        out = torch.zeros_like(Ax)
        out[:, 1:, ...] = Ax[:, :-1, ...]
        return out

    def forward(self, u):

        A = torch.exp(-torch.exp(self.nu_log))

        # Expand A across batch/time dims
        A_expanded = A.expand(u.shape[0], u.shape[1], -1, -1)

        # Ax + Bu
        Bu = torch.exp(self.gamma_log) * torch.matmul(self.B, u.transpose(-2, -1)).transpose(-2, -1)
        x = self.scan(A_expanded, Bu).transpose(-2, -1)

        # Cx + Du
        y = torch.matmul(self.C, x)
        y = y + self.D * u.transpose(-2, -1)

        return y.transpose(-2, -1)

# ----------EOF---------- #
