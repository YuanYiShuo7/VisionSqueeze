import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """将图像分割为补丁并嵌入"""
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 使用卷积实现补丁嵌入
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + (224 // patch_size) ** 2, embed_dim) * 0.02)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x):
        B, C, H, W = x.shape
        # 确保输入尺寸是patch_size的整数倍
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        
        # 补丁投影 [B, C, H, W] -> [B, embed_dim, H/patch, W/patch]
        x = self.proj(x)
        
        # 展平补丁 [B, embed_dim, H/patch, W/patch] -> [B, embed_dim, num_patches]
        x = x.flatten(2)
        
        # 转置得到 [B, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]
        
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 投影输出
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP层
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256], latent_dim=16,
                 patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                 mlp_ratio=4.0, img_size=224):
        """
        基于ViT的编码器
        
        Args:
            in_channels: 输入通道数
            hidden_dims: 保留参数以保持接口一致（ViT不使用）
            latent_dim: 潜在维度
            patch_size: 补丁大小
            embed_dim: 嵌入维度
            depth: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            img_size: 输入图像尺寸
        """
        super().__init__()
        
        # 补丁嵌入
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) 
            for _ in range(depth)
        ])
        
        # 最终归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 潜在空间投影
        num_patches = (img_size // patch_size) ** 2
        self.latent_proj = nn.Conv2d(embed_dim, latent_dim, kernel_size=1)
        
        # 保存参数用于解码器
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            latent: 潜在表示 [B, latent_dim, H', W']
        """
        B, C, H, W = x.shape
        
        # 通过ViT编码器
        x = self.patch_embed(x)  # [B, num_patches+1, embed_dim]
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 最终归一化
        x = self.norm(x)
        
        # 移除CLS token并重塑为空间格式
        x = x[:, 1:, :]  # 移除CLS token [B, num_patches, embed_dim]
        
        # 重塑为2D特征图 [B, embed_dim, h, w]
        h = w = int(math.sqrt(x.size(1)))
        x = x.transpose(1, 2).reshape(B, self.embed_dim, h, w)
        
        # 投影到潜在空间
        latent = self.latent_proj(x)  # [B, latent_dim, h, w]
        
        return latent