import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn

from .aft2d import AFT2DBlock

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=16,
            stride=16,
            padding=0
        )
        
        self.norm = nn.LayerNorm(64)
        
        self.downsample = nn.ModuleList([
            AFT2DBlock(in_dim=64, hidden_dim=128, out_dim=16, r=1, expansion_factor=2, dropout=0.1),
            AFT2DBlock(in_dim=16, hidden_dim=32, out_dim=4, r=4, expansion_factor=2, dropout=0.1)
        ])
        
        
    def forward(self, x):
        """
        参数:
            x: 输入图像 (B, C, H, W)，已归一化到[0,1]
        返回:
            out: (B, H', W', 4)
        """
        
        x = self.embed(x)  # (B, emb_dim, H/patch_size, W/patch_size)
        
        x = x.permute(0, 2, 3, 1)  # (B, H/patch_size, W/patch_size, emb_dim)

        x = self.norm(x)

        for block in self.downsample:
            x = block(x)
        
        return x
    