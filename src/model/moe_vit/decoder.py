import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math

class ViTPatchDecoder(nn.Module):
    """与ViTPatchEncoder对称的解码器"""
    def __init__(self, latent_channels, output_channels=64, patch_size=16, embed_dim=256):
        super().__init__()
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 潜在空间到嵌入空间的投影
        hw = patch_size // 4
        self.proj_from_latent = nn.Sequential(
            nn.Linear(latent_channels * hw * hw, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Tanh()
        )
        
        # Transformer解码器
        self.transformer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # 重建图像
        self.reconstruct_patch = nn.Sequential(
            nn.Linear(embed_dim, output_channels * patch_size * patch_size),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: [N, latent_channels, patch_size//4, patch_size//4]
        输出: [N, output_channels, patch_size, patch_size]
        """
        N = x.shape[0]
        
        # 展平潜在表示
        x_flat = x.flatten(1)  # [N, latent_channels * (patch_size//4)^2]
        
        # 投影到嵌入空间
        patch_emb = self.proj_from_latent(x_flat)  # [N, embed_dim]
        patch_emb = patch_emb.unsqueeze(1)  # [N, 1, embed_dim]
        
        # Transformer解码
        attn_output, _ = self.transformer[1](patch_emb, patch_emb, patch_emb)
        patch_emb = patch_emb + attn_output
        patch_emb = self.transformer[2](patch_emb)
        
        # 通过MLP
        patch_emb = patch_emb + self.transformer[3:](patch_emb)
        patch_emb = patch_emb.squeeze(1)  # [N, embed_dim]
        
        # 重建图像补丁
        patch_flat = self.reconstruct_patch(patch_emb)  # [N, output_channels * patch_size^2]
        
        # 重塑为2D格式
        patch = patch_flat.view(N, self.output_channels, self.patch_size, self.patch_size)
        
        return patch


class MoEPatchDecoder(nn.Module):
    """与MoEViTEncoder对称的解码器"""
    def __init__(self, 
                 latent_channels_list: List[int], 
                 patch_size: int = 16,
                 output_channels: int = 64,
                 embed_dim: int = 256):
        super().__init__()
        self.num_experts = len(latent_channels_list)
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.embed_dim = embed_dim
        
        # 解码器专家（与编码器一一对应）
        self.decoders = nn.ModuleList([
            ViTPatchDecoder(c, output_channels, patch_size, embed_dim) 
            for c in latent_channels_list
        ])
        
        # 重建头：将特征转换为RGB图像
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(output_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def decode_patch(self, latent: torch.Tensor, expert_id: int) -> torch.Tensor:
        latent = latent.unsqueeze(0)
        decoder = self.decoders[expert_id]
        patch_features = decoder(latent)
        return patch_features.squeeze(0)
    
    def decode_patches_to_features(self, latents: List[Dict]) -> torch.Tensor:
        if not latents:
            return torch.zeros(1, self.output_channels, 0, 0)
        
        device = latents[0]['latent'].device
        
        xs = [p['position'][0] for p in latents]
        ys = [p['position'][1] for p in latents]
        max_x = max(xs)
        max_y = max(ys)
        image_w = (max_x + 1) * self.patch_size
        image_h = (max_y + 1) * self.patch_size
        
        feature_buffer = torch.zeros(1, self.output_channels, image_h, image_w, device=device)
        count_buffer = torch.zeros(1, 1, image_h, image_w, device=device)
        
        expert_groups = {i: {'latents': [], 'positions': [], 'indices': []} 
                        for i in range(self.num_experts)}
        
        for idx, patch_data in enumerate(latents):
            expert_id = patch_data['expert_id']
            expert_groups[expert_id]['latents'].append(patch_data['latent'])
            expert_groups[expert_id]['positions'].append(patch_data['position'])
            expert_groups[expert_id]['indices'].append(idx)
        
        all_patches = [None] * len(latents)
        
        for expert_id in range(self.num_experts):
            if not expert_groups[expert_id]['latents']:
                continue
                
            expert_latents = expert_groups[expert_id]['latents']
            positions = expert_groups[expert_id]['positions']
            indices = expert_groups[expert_id]['indices']
            
            latents_batch = torch.stack(expert_latents)
            decoder = self.decoders[expert_id]
            patches_batch = decoder(latents_batch)
            
            for i, idx in enumerate(indices):
                all_patches[idx] = {
                    'patch_features': patches_batch[i],
                    'position': positions[i]
                }
        
        for patch_data in all_patches:
            patch_features = patch_data['patch_features']
            x, y = patch_data['position']
            
            x_start = x * self.patch_size
            y_start = y * self.patch_size
            x_end = x_start + self.patch_size
            y_end = y_start + self.patch_size
            
            feature_buffer[0, :, y_start:y_end, x_start:x_end] += patch_features
            count_buffer[0, 0, y_start:y_end, x_start:x_end] += 1
        
        feature_map = feature_buffer / torch.clamp(count_buffer, min=1)
        
        return feature_map
    
    def forward(self, latents: List[Dict]) -> torch.Tensor:
        feature_map = self.decode_patches_to_features(latents)
        reconstructed_image = self.reconstruction_head(feature_map)
        
        return reconstructed_image