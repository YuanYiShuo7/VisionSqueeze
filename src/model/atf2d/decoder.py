import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn

from .aft2d import AFT2DBlock

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 对称的AFT2DBlocks
        self.upsample = nn.ModuleList([
            AFT2DBlock(in_dim=4, hidden_dim=32, out_dim=16, r=4, expansion_factor=2, dropout=0.1),
            AFT2DBlock(in_dim=16, hidden_dim=128, out_dim=64, r=1, expansion_factor=2, dropout=0.1)
        ])
        
        self.norm = nn.LayerNorm(64)
        
        # 直接使用转置卷积重建
        self.reconstruct = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=16,
            stride=16,
            padding=0,
            output_padding=0
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        参数:
            x: 编码特征 (B, H', W', 4)
        返回:
            out: 重建图像 (B, 3, H, W)
        """
        # 特征上采样
        for block in self.upsample:
            x = block(x)
        
        # 归一化
        x = self.norm(x)
        
        # 转换格式
        x = x.permute(0, 3, 1, 2)
        
        # 上采样到原始分辨率
        x = self.reconstruct(x)
        
        # 激活函数
        x = self.sigmoid(x)
        
        return x