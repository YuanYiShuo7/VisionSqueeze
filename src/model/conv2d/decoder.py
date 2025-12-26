import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dims=[256, 128, 64, 64, 32], out_channels=3):
        """
        与Encoder完全对称的Decoder
        
        Args:
            latent_dim: 潜在维度，必须与Encoder的latent_dim匹配
            hidden_dims: 隐藏层维度列表，必须与Encoder的hidden_dims顺序相反
            out_channels: 输出通道数（RGB图像为3）
        """
        super().__init__()
        
        # 反转hidden_dims的顺序以实现对称
        hidden_dims_reversed = hidden_dims[::-1]
        
        layers = []
        in_dim = latent_dim
        
        # 第一层上采样（对应Encoder的最后下采样层）
        layers.append(
            nn.ConvTranspose2d(in_dim, hidden_dims_reversed[0], 
                              kernel_size=4, stride=2, padding=1, output_padding=0)
        )
        layers.append(nn.BatchNorm2d(hidden_dims_reversed[0]))
        layers.append(nn.ReLU(inplace=True))
        in_dim = hidden_dims_reversed[0]
        
        # 中间上采样层（对应Encoder的中间下采样层）
        for i in range(len(hidden_dims_reversed)):
            # 如果是最后一层，输出到out_channels
            if i == len(hidden_dims_reversed) - 1:
                # 最后一层使用卷积而不是转置卷积来匹配尺寸
                layers.append(
                    nn.ConvTranspose2d(in_dim, out_channels, 
                                      kernel_size=3, stride=2, padding=1, output_padding=1)
                )
            else:
                # 中间层
                layers.append(
                    nn.ConvTranspose2d(in_dim, hidden_dims_reversed[i+1], 
                                      kernel_size=3, stride=2, padding=1, output_padding=1)
                )
                layers.append(nn.BatchNorm2d(hidden_dims_reversed[i+1]))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dims_reversed[i+1]
        
        # 添加Sigmoid激活函数确保输出在[0,1]范围
        layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 潜在表示 [B, latent_dim, H', W']
            
        Returns:
            reconstructed: 重建图像 [B, out_channels, H, W]
        """
        return self.decoder(x)