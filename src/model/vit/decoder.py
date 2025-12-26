import torch
import torch.nn as nn
import math

class PatchUpsample(nn.Module):
    """将特征图上采样并重塑回图像"""
    def __init__(self, embed_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        # 计算补丁数量
        self.num_patches = (img_size // patch_size) ** 2
        
        # 位置编码（与编码器匹配）
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim) * 0.02)
        
        # CLS token（仅为了结构对称）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 展平为序列 [B, C, H, W] -> [B, C, H*W]
        x = x.flatten(2)
        
        # 转置 [B, C, H*W] -> [B, H*W, C]
        x = x.transpose(1, 2)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]
        
        return x

class ViTDecoderBlock(nn.Module):
    """ViT解码器块（结构与编码器类似但可能更简单）"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP层
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        # 自注意力
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output
        
        # MLP
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dims=[256, 128, 64, 64, 32], out_channels=3,
                 patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4.0, img_size=224):
        """
        基于ViT的解码器
        
        Args:
            latent_dim: 潜在维度，必须与Encoder的latent_dim匹配
            hidden_dims: 保留参数以保持接口一致（ViT不使用）
            out_channels: 输出通道数
            patch_size: 补丁大小（必须与编码器匹配）
            embed_dim: 嵌入维度（必须与编码器匹配）
            depth: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            img_size: 输出图像尺寸
        """
        super().__init__()
        
        # 从潜在空间到嵌入空间的投影
        self.latent_to_embed = nn.ConvTranspose2d(latent_dim, embed_dim, 
                                                 kernel_size=1, stride=1)
        
        # 补丁上采样
        self.patch_upsample = PatchUpsample(embed_dim, patch_size, img_size)
        
        # Transformer解码器（可以使用更浅的层）
        self.blocks = nn.ModuleList([
            ViTDecoderBlock(embed_dim, num_heads, mlp_ratio) 
            for _ in range(depth)
        ])
        
        # 最终归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 图像重建层
        self.reconstruct = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * out_channels),
            nn.Tanh()
        )
        
        # 保存参数
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_channels = out_channels
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 潜在表示 [B, latent_dim, H', W']
            
        Returns:
            reconstructed: 重建图像 [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # 从潜在空间投影到嵌入空间
        x = self.latent_to_embed(x)  # [B, embed_dim, H', W']
        
        # 通过补丁上采样转为序列
        x = self.patch_upsample(x)  # [B, num_patches+1, embed_dim]
        
        # 通过Transformer解码器
        for block in self.blocks:
            x = block(x)
        
        # 最终归一化
        x = self.norm(x)
        
        # 移除CLS token
        x = x[:, 1:, :]  # [B, num_patches, embed_dim]
        
        # 重建图像补丁
        x = self.reconstruct(x)  # [B, num_patches, patch_size*patch_size*out_channels]
        
        # 重塑为图像
        num_patches = (self.img_size // self.patch_size) ** 2
        h_patches = w_patches = int(math.sqrt(num_patches))
        
        # 重塑为 [B, h_patches, w_patches, patch_size, patch_size, out_channels]
        x = x.reshape(B, h_patches, w_patches, 
                     self.patch_size, self.patch_size, self.out_channels)
        
        # 重新排列维度 [B, out_channels, H, W]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.out_channels, 
                     h_patches * self.patch_size,
                     w_patches * self.patch_size)
        
        # 使用Sigmoid确保输出在[0,1]范围
        x = torch.sigmoid(x)
        
        return x