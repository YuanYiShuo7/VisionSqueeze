import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import random

class PatchEncoder(nn.Module):
    def __init__(self, latent_channels, mid_channels=64, is_shared_expert=False, shared_channels_indices=None):
        """
        Args:
            latent_channels: 目标输出通道数
            mid_channels: 中间层通道数
            is_shared_expert: 是否为共享专家
            shared_channels_indices: 共享通道的索引列表，用于在共享专家中标识哪些通道是共享的
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.is_shared_expert = is_shared_expert
        self.shared_channels_indices = shared_channels_indices
        
        # 网络结构保持不变
        self.net = nn.Sequential(
            # 第一层：stride=2，降为1/2
            nn.Conv2d(3, mid_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 第二层：stride=2，再降为1/2（总共1/4）
            nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 最后一层：1x1卷积调整通道数
            nn.Conv2d(mid_channels, latent_channels, 1)
        )

    def forward(self, x):
        return self.net(x)  # [N, C, H/4, W/4]


class PatchRouter(nn.Module):
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
        self.register_buffer('step', torch.tensor(0))  # 记录训练步数
    
    def forward(self, patch_feat):
        logits = self.router(patch_feat)
        
        if self.training:
            # 计算当前噪声强度（线性衰减）
            progress = min(self.step.item() / self.warmup_steps, 1.0)
            current_noise_std = self.initial_noise_std * (1 - progress) + self.min_noise_std * progress
            
            noise = torch.randn_like(logits) * current_noise_std
            logits = logits + noise
            
            self.step += 1
        
        probs = F.softmax(logits, dim=-1)
        
        if self.training:
            # 使用带温度参数的gumbel_softmax，温度也可以衰减
            tau = max(1.0, 5.0 * (1 - progress))  # 温度从5.0衰减到1.0
            expert_id = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            expert_id = torch.argmax(expert_id, dim=-1)
        else:
            expert_id = torch.argmax(probs, dim=-1)
        
        return probs, expert_id


class SharedExpertManager:
    """共享专家管理器"""
    def __init__(self, max_total_channels: int, shared_channels_fraction: float = 0.2):
        """
        Args:
            max_total_channels: 所有专家通道数的总和上限
            shared_channels_fraction: 共享通道占总通道的比例
        """
        self.max_total_channels = max_total_channels
        self.shared_channels_fraction = shared_channels_fraction
        self.shared_channels = None
        self.shared_channel_indices = None
        
    def init_shared_channels(self, total_expert_channels: int):
        """初始化共享通道"""
        # 计算共享通道数
        num_shared_channels = int(total_expert_channels * self.shared_channels_fraction)
        num_shared_channels = max(1, num_shared_channels)  # 至少1个共享通道
        
        # 创建共享通道（随机初始化）
        self.shared_channels = torch.randn(num_shared_channels, requires_grad=True)
        
        # 随机选择共享通道的位置
        self.shared_channel_indices = random.sample(
            range(total_expert_channels), 
            num_shared_channels
        )
        self.shared_channel_indices.sort()
        
        return self.shared_channels, self.shared_channel_indices
    
    def replace_with_shared_channels(self, expert_output: torch.Tensor, 
                                   expert_idx: int, 
                                   replacement_rate: float = 0.3) -> torch.Tensor:
        """
        用共享通道替换专家输出中的部分通道
        
        Args:
            expert_output: 专家输出 [B, C, H, W]
            expert_idx: 专家索引
            replacement_rate: 通道替换比例
            
        Returns:
            替换后的输出
        """
        if self.shared_channels is None:
            return expert_output
            
        B, C, H, W = expert_output.shape
        
        # 计算需要替换的通道数
        num_replace = max(1, int(C * replacement_rate))
        
        # 随机选择要替换的通道索引
        replace_indices = random.sample(range(C), num_replace)
        
        # 为每个batch样本替换通道
        for i in range(B):
            for ch_idx in replace_indices:
                # 从共享通道中随机选择一个值来替换
                shared_val = random.choice(self.shared_channels)
                expert_output[i, ch_idx, :, :] = shared_val
        
        return expert_output


class MoEPatchEncoder(nn.Module):
    def __init__(self, latent_channels_list: List[int], 
                 patch_size: int = 16,
                 use_shared_expert: bool = False,
                 shared_expert_fraction: float = 0.2,
                 channel_replacement_rate: float = 0.3):
        """
        带有共享专家机制的MoE编码器
        
        Args:
            latent_channels_list: 每个专家的输出通道数列表
            patch_size: patch大小
            use_shared_expert: 是否使用共享专家机制
            shared_expert_fraction: 共享通道占总通道的比例
            channel_replacement_rate: 通道替换的比例
        """
        super().__init__()
        self.num_experts = len(latent_channels_list)
        self.patch_size = patch_size
        self.patch_feat_dim = patch_size * patch_size * 3
        self.use_shared_expert = use_shared_expert
        self.channel_replacement_rate = channel_replacement_rate
        
        # 计算总的专家通道数
        total_expert_channels = sum(latent_channels_list)
        
        # 创建路由器（专家数量不变）
        self.router = PatchRouter(
            in_dim=self.patch_feat_dim,
            num_experts=self.num_experts
        )
        
        # 如果有共享专家机制
        if self.use_shared_expert:
            # 创建共享专家管理器
            self.shared_manager = SharedExpertManager(
                max_total_channels=total_expert_channels * 2,  # 预留一些空间
                shared_channels_fraction=shared_expert_fraction
            )
            
            # 初始化共享通道
            self.shared_channels, self.shared_channel_indices = \
                self.shared_manager.init_shared_channels(total_expert_channels)
            
            # 注册共享通道为可学习参数
            self.register_parameter('shared_channels_param', 
                                  nn.Parameter(self.shared_channels))
            
            # 保存共享通道索引
            self.register_buffer('shared_channel_indices_tensor',
                               torch.tensor(self.shared_channel_indices, dtype=torch.long))
        
        # 创建专用专家（保持原有的通道数）
        self.encoders = nn.ModuleList()
        for c in latent_channels_list:
            if self.use_shared_expert:
                # 计算每个专家的共享通道索引
                expert_shared_indices = []
                for idx in self.shared_channel_indices:
                    # 确定这个共享通道属于哪个专家
                    expert_boundary = 0
                    for expert_idx, expert_c in enumerate(latent_channels_list):
                        if idx < expert_boundary + expert_c:
                            # 这个共享通道属于当前专家
                            local_idx = idx - expert_boundary
                            expert_shared_indices.append(local_idx)
                            break
                        expert_boundary += expert_c
                
                encoder = PatchEncoder(
                    latent_channels=c,
                    mid_channels=64,
                    is_shared_expert=False,
                    shared_channels_indices=expert_shared_indices if expert_shared_indices else None
                )
            else:
                encoder = PatchEncoder(c)
            self.encoders.append(encoder)
    
    def apply_shared_channels(self, expert_output: torch.Tensor, 
                            expert_idx: int) -> torch.Tensor:
        """
        应用共享通道替换
        
        Args:
            expert_output: 专家输出 [B, C, H, W]
            expert_idx: 专家索引
            
        Returns:
            替换后的输出
        """
        if not self.use_shared_expert or not self.training:
            return expert_output
        
        B, C, H, W = expert_output.shape
        
        # 确定当前专家负责的共享通道
        expert_shared_indices = []
        expert_boundary_start = 0
        for i in range(expert_idx):
            expert_boundary_start += self.encoders[i].latent_channels
        
        # 遍历所有共享通道索引，找到属于当前专家的
        for shared_idx in self.shared_channel_indices_tensor:
            if expert_boundary_start <= shared_idx < expert_boundary_start + C:
                local_idx = shared_idx - expert_boundary_start
                expert_shared_indices.append(local_idx.item())
        
        if not expert_shared_indices:
            return expert_output
        
        # 创建替换掩码
        replacement_mask = torch.zeros(B, C, H, W, device=expert_output.device)
        
        # 随机选择要替换的batch和空间位置
        for batch_idx in range(B):
            # 对每个共享通道，随机选择部分空间位置进行替换
            for ch_idx in expert_shared_indices:
                # 随机选择替换的比例
                num_pixels = H * W
                num_replace = max(1, int(num_pixels * self.channel_replacement_rate))
                
                # 随机选择要替换的像素位置
                replace_indices = random.sample(range(num_pixels), num_replace)
                
                # 将选中的位置替换为共享通道的值
                for idx in replace_indices:
                    h_idx = idx // W
                    w_idx = idx % W
                    replacement_mask[batch_idx, ch_idx, h_idx, w_idx] = 1.0
        
        # 获取当前批次的共享通道值（随机采样）
        if hasattr(self, 'shared_channels_param'):
            # 从共享通道参数中随机选择值
            shared_values = torch.stack([
                random.choice(self.shared_channels_param) 
                for _ in range(B * C * H * W)
            ]).view(B, C, H, W)
        else:
            # 随机生成共享值
            shared_values = torch.randn_like(expert_output)
        
        # 应用替换
        expert_output = expert_output * (1 - replacement_mask) + shared_values * replacement_mask
        
        return expert_output
    
    def extract_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """与原始版本相同"""
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, H, W]
        
        B, C, H, W = image.shape
        assert B == 1, "只支持单张图像输入"
        
        # 计算grid宽高（动态）
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        
        # 使用unfold提取patch
        patches = image.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        N_patches = patches.shape[1] * patches.shape[2]
        patches = patches.view(N_patches, 3, self.patch_size, self.patch_size)
        
        # 生成位置索引
        positions = []
        for y in range(grid_h):
            for x in range(grid_w):
                positions.append((x, y))
        
        return patches, positions
    
    def forward(self, image: torch.Tensor) -> List[Dict]:
        """
        前向传播，包含共享专家机制
        
        Args:
            image: 单张图像 [1, 3, H, W] 或 [3, H, W]
            
        Returns:
            results: List[Dict]，每个字典包含:
                - position: (x, y) patch位置坐标
                - latent: [C, H', W'] patch的潜在表示
                - expert_id: int 分配的专家ID
                - router_probs: [num_experts] 路由概率
                - used_shared: bool 是否使用了共享通道
        """
        # 1. 提取patch和位置信息
        patches, positions = self.extract_patches(image)
        N = len(patches)
        
        if image.dim() == 3:
            H = image.shape[1]
            W = image.shape[2]
        else:
            H = image.shape[2]
            W = image.shape[3]
        grid_w = W // self.patch_size
        
        # 2. 路由决策
        router_probs, expert_ids = self.router(patches.view(N, -1))
        
        # 3. 分组处理每个专家
        expert_indices = {i: [] for i in range(self.num_experts)}
        
        for idx, expert_id in enumerate(expert_ids):
            expert_indices[int(expert_id.item())].append(idx)
        
        # 4. 并行处理每个专家的所有patch
        results = []
        for expert_id in range(self.num_experts):
            if expert_indices[expert_id]:
                indices = torch.tensor(expert_indices[expert_id], device=patches.device, dtype=torch.long)
                expert_patches = patches[indices]
                
                # 批量处理该专家的所有patch
                latents = self.encoders[expert_id](expert_patches)  # [K, C, H', W']
                
                # 应用共享通道替换（如果启用）
                if self.use_shared_expert:
                    latents = self.apply_shared_channels(latents, expert_id)
                    used_shared = True
                else:
                    used_shared = False
                
                # 为每个patch创建结果
                for j, patch_idx in enumerate(indices.tolist()):
                    result = {
                        'position': positions[patch_idx],
                        'latent': latents[j],  # [C, H', W']
                        'expert_id': expert_id,
                        'router_probs': router_probs[patch_idx],  # [num_experts]
                        'used_shared': used_shared
                    }
                    results.append(result)
        
        # 按原始顺序排序
        results = sorted(results, key=lambda x: x['position'][1] * grid_w + x['position'][0])
        
        return results
    
    def get_shared_channel_info(self) -> Dict:
        """获取共享通道信息"""
        if not self.use_shared_expert:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "num_shared_channels": len(self.shared_channels_param) if hasattr(self, 'shared_channels_param') else 0,
            "shared_channel_indices": self.shared_channel_indices_tensor.tolist() if hasattr(self, 'shared_channel_indices_tensor') else [],
            "replacement_rate": self.channel_replacement_rate
        }