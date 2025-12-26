import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256], latent_dim=16):
        super().__init__()
        
        # 逐步下采样，压缩到很小的潜在表示
        layers = []
        in_dim = in_channels
        
        # 下采样阶段
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim
        
        # 最终编码层 - 压缩到很小的潜在空间
        layers.append(
            nn.Conv2d(in_dim, latent_dim, kernel_size=4, stride=2, padding=1)
        )

        self.encoder = nn.Sequential(*layers)
        
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            latent: 潜在表示 
        """

        latent = self.encoder(x)
        return latent