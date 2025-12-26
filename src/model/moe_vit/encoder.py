import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math
import random

class ViTPatchEncoder(nn.Module):
    def __init__(self, latent_channels, mid_channels=64, is_shared_expert=False, 
                 shared_channels_indices=None, patch_size=16, embed_dim=256):
        """
        Args:
            latent_channels: 目标输出通道数
            mid_channels: 中间层通道数
            is_shared_expert: 是否为共享专家
            shared_channels_indices: 共享通道的索引列表
            patch_size: 补丁大小
            embed_dim: Transformer嵌入维度
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.is_shared_expert = is_shared_expert
        self.shared_channels_indices = shared_channels_indices
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # ViT编码器结构
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Transformer编码器
        self.transformer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True),
            nn.LayerNorm(embed_dim),
        )
        
        # 投影到潜在空间
        self.proj_to_latent = nn.Sequential(
            nn.Linear(embed_dim, mid_channels * patch_size * patch_size),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels * patch_size * patch_size, latent_channels * (patch_size//4) * (patch_size//4)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: [N, 3, patch_size, patch_size]
        输出: [N, latent_channels, patch_size//4, patch_size//4]
        """
        N = x.shape[0]
        
        # 补丁嵌入
        patch_emb = self.patch_embed(x)  # [N, embed_dim, 1, 1]
        patch_emb = patch_emb.flatten(1)  # [N, embed_dim]
        
        # Transformer处理
        patch_emb = patch_emb.unsqueeze(1)  # [N, 1, embed_dim]
        attn_output, _ = self.transformer[6](patch_emb, patch_emb, patch_emb)
        patch_emb = patch_emb + attn_output
        patch_emb = self.transformer[7](patch_emb)
        patch_emb = patch_emb.squeeze(1)  # [N, embed_dim]
        
        # 投影到潜在空间
        latent_flat = self.proj_to_latent(patch_emb)  # [N, latent_channels * (patch_size//4)^2]
        
        # 重塑为2D格式
        hw = self.patch_size // 4
        latent = latent_flat.view(N, self.latent_channels, hw, hw)
        
        return latent


class ViTPatchRouter(nn.Module):
    def __init__(
        self,
        in_dim,
        num_experts,
        initial_noise_std: float = 1,
        min_noise_std: float = 0.5,
        warmup_steps: int = 10000,
    ):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_experts)
        )
        self.initial_noise_std = initial_noise_std
        self.min_noise_std = min_noise_std
        self.warmup_steps = warmup_steps
        self.register_buffer('step', torch.tensor(0))
    
    def forward(self, patch_feat):
        logits = self.router(patch_feat)
        
        if self.training:
            progress = min(self.step.item() / self.warmup_steps, 1.0)
            current_noise_std = self.initial_noise_std * (1 - progress) + self.min_noise_std * progress
            
            noise = torch.randn_like(logits) * current_noise_std
            logits = logits + noise
            
            self.step += 1
        
        probs = F.softmax(logits, dim=-1)
        
        if self.training:
            tau = max(1.0, 5.0 * (1 - progress))
            expert_id = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            expert_id = torch.argmax(expert_id, dim=-1)
        else:
            expert_id = torch.argmax(probs, dim=-1)
        
        return probs, expert_id


class ViTSharedExpertManager:
    """ViT版本的共享专家管理器"""
    def __init__(self, max_total_channels: int, shared_channels_fraction: float = 0.2):
        self.max_total_channels = max_total_channels
        self.shared_channels_fraction = shared_channels_fraction
        self.shared_channels = None
        self.shared_channel_indices = None
        
    def init_shared_channels(self, total_expert_channels: int):
        num_shared_channels = int(total_expert_channels * self.shared_channels_fraction)
        num_shared_channels = max(1, num_shared_channels)
        
        self.shared_channels = torch.randn(num_shared_channels, requires_grad=True)
        
        self.shared_channel_indices = random.sample(
            range(total_expert_channels), 
            num_shared_channels
        )
        self.shared_channel_indices.sort()
        
        return self.shared_channels, self.shared_channel_indices
    
    def replace_with_shared_channels(self, expert_output: torch.Tensor, 
                                   expert_idx: int, 
                                   replacement_rate: float = 0.3) -> torch.Tensor:
        if self.shared_channels is None:
            return expert_output
            
        B, C, H, W = expert_output.shape
        
        num_replace = max(1, int(C * replacement_rate))
        replace_indices = random.sample(range(C), num_replace)
        
        for i in range(B):
            for ch_idx in replace_indices:
                shared_val = random.choice(self.shared_channels)
                expert_output[i, ch_idx, :, :] = shared_val
        
        return expert_output


class MoEPatchEncoder(nn.Module):
    def __init__(self, latent_channels_list: List[int], 
                 patch_size: int = 16,
                 use_shared_expert: bool = False,
                 shared_expert_fraction: float = 0.2,
                 channel_replacement_rate: float = 0.3,
                 embed_dim: int = 256):
        """
        基于ViT的MoE编码器
        
        Args:
            latent_channels_list: 每个专家的输出通道数列表
            patch_size: patch大小
            use_shared_expert: 是否使用共享专家机制
            shared_expert_fraction: 共享通道占总通道的比例
            channel_replacement_rate: 通道替换的比例
            embed_dim: Transformer嵌入维度
        """
        super().__init__()
        self.num_experts = len(latent_channels_list)
        self.patch_size = patch_size
        self.patch_feat_dim = patch_size * patch_size * 3
        self.use_shared_expert = use_shared_expert
        self.channel_replacement_rate = channel_replacement_rate
        self.embed_dim = embed_dim
        
        total_expert_channels = sum(latent_channels_list)
        
        self.router = ViTPatchRouter(
            in_dim=self.patch_feat_dim,
            num_experts=self.num_experts
        )
        
        if self.use_shared_expert:
            self.shared_manager = ViTSharedExpertManager(
                max_total_channels=total_expert_channels * 2,
                shared_channels_fraction=shared_expert_fraction
            )
            
            self.shared_channels, self.shared_channel_indices = \
                self.shared_manager.init_shared_channels(total_expert_channels)
            
            self.register_parameter('shared_channels_param', 
                                  nn.Parameter(self.shared_channels))
            
            self.register_buffer('shared_channel_indices_tensor',
                               torch.tensor(self.shared_channel_indices, dtype=torch.long))
        
        self.encoders = nn.ModuleList()
        for c in latent_channels_list:
            if self.use_shared_expert:
                expert_shared_indices = []
                for idx in self.shared_channel_indices:
                    expert_boundary = 0
                    for expert_idx, expert_c in enumerate(latent_channels_list):
                        if idx < expert_boundary + expert_c:
                            local_idx = idx - expert_boundary
                            expert_shared_indices.append(local_idx)
                            break
                        expert_boundary += expert_c
                
                encoder = ViTPatchEncoder(
                    latent_channels=c,
                    mid_channels=64,
                    is_shared_expert=False,
                    shared_channels_indices=expert_shared_indices if expert_shared_indices else None,
                    patch_size=patch_size,
                    embed_dim=embed_dim
                )
            else:
                encoder = ViTPatchEncoder(
                    latent_channels=c,
                    mid_channels=64,
                    patch_size=patch_size,
                    embed_dim=embed_dim
                )
            self.encoders.append(encoder)
    
    def apply_shared_channels(self, expert_output: torch.Tensor, 
                            expert_idx: int) -> torch.Tensor:
        if not self.use_shared_expert or not self.training:
            return expert_output
        
        B, C, H, W = expert_output.shape
        
        expert_shared_indices = []
        expert_boundary_start = 0
        for i in range(expert_idx):
            expert_boundary_start += self.encoders[i].latent_channels
        
        for shared_idx in self.shared_channel_indices_tensor:
            if expert_boundary_start <= shared_idx < expert_boundary_start + C:
                local_idx = shared_idx - expert_boundary_start
                expert_shared_indices.append(local_idx.item())
        
        if not expert_shared_indices:
            return expert_output
        
        replacement_mask = torch.zeros(B, C, H, W, device=expert_output.device)
        
        for batch_idx in range(B):
            for ch_idx in expert_shared_indices:
                num_pixels = H * W
                num_replace = max(1, int(num_pixels * self.channel_replacement_rate))
                replace_indices = random.sample(range(num_pixels), num_replace)
                
                for idx in replace_indices:
                    h_idx = idx // W
                    w_idx = idx % W
                    replacement_mask[batch_idx, ch_idx, h_idx, w_idx] = 1.0
        
        if hasattr(self, 'shared_channels_param'):
            shared_values = torch.stack([
                random.choice(self.shared_channels_param) 
                for _ in range(B * C * H * W)
            ]).view(B, C, H, W)
        else:
            shared_values = torch.randn_like(expert_output)
        
        expert_output = expert_output * (1 - replacement_mask) + shared_values * replacement_mask
        
        return expert_output
    
    def extract_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        assert B == 1, "只支持单张图像输入"
        
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        N_patches = patches.shape[1] * patches.shape[2]
        patches = patches.view(N_patches, 3, self.patch_size, self.patch_size)
        
        positions = []
        for y in range(grid_h):
            for x in range(grid_w):
                positions.append((x, y))
        
        return patches, positions
    
    def forward(self, image: torch.Tensor) -> List[Dict]:
        patches, positions = self.extract_patches(image)
        N = len(patches)
        
        if image.dim() == 3:
            H = image.shape[1]
            W = image.shape[2]
        else:
            H = image.shape[2]
            W = image.shape[3]
        grid_w = W // self.patch_size
        
        router_probs, expert_ids = self.router(patches.view(N, -1))
        
        expert_indices = {i: [] for i in range(self.num_experts)}
        
        for idx, expert_id in enumerate(expert_ids):
            expert_indices[int(expert_id.item())].append(idx)
        
        results = []
        for expert_id in range(self.num_experts):
            if expert_indices[expert_id]:
                indices = torch.tensor(expert_indices[expert_id], device=patches.device, dtype=torch.long)
                expert_patches = patches[indices]
                
                latents = self.encoders[expert_id](expert_patches)
                
                if self.use_shared_expert:
                    latents = self.apply_shared_channels(latents, expert_id)
                    used_shared = True
                else:
                    used_shared = False
                
                for j, patch_idx in enumerate(indices.tolist()):
                    result = {
                        'position': positions[patch_idx],
                        'latent': latents[j],
                        'expert_id': expert_id,
                        'router_probs': router_probs[patch_idx],
                        'used_shared': used_shared
                    }
                    results.append(result)
        
        results = sorted(results, key=lambda x: x['position'][1] * grid_w + x['position'][0])
        
        return results
    
    def get_shared_channel_info(self) -> Dict:
        if not self.use_shared_expert:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "num_shared_channels": len(self.shared_channels_param) if hasattr(self, 'shared_channels_param') else 0,
            "shared_channel_indices": self.shared_channel_indices_tensor.tolist() if hasattr(self, 'shared_channel_indices_tensor') else [],
            "replacement_rate": self.channel_replacement_rate
        }