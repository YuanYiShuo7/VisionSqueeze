import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math

class PatchDecoder(nn.Module):
    """与PatchEncoder对称的解码器，支持动态输出通道"""
    def __init__(self, latent_channels, output_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, output_channels, 1),
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.net(x)
        return x

class MoEPatchDecoder(nn.Module):
    """与MoEPatchEncoder对称的解码器，支持动态图像尺寸和动态输出通道"""
    def __init__(self, 
                 latent_channels_list: List[int], 
                 patch_size: int = 16,
                 output_channels: int = 64):  # 添加输出通道参数
        super().__init__()
        self.num_experts = len(latent_channels_list)
        self.patch_size = patch_size
        self.output_channels = output_channels
        
        # 解码器专家（与编码器一一对应），传入输出通道数
        self.decoders = nn.ModuleList([
            PatchDecoder(c, output_channels) for c in latent_channels_list
        ])
        
        # 重建头：将特征转换为RGB图像
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(output_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
        
    def decode_patch(self, latent: torch.Tensor, expert_id: int) -> torch.Tensor:
        """
        解码单个patch为特征
        
        Args:
            latent: [C, H', W'] patch的潜在表示
            expert_id: 专家ID
            
        Returns:
            patch_features: [output_channels, patch_size, patch_size] patch特征
        """
        # 添加batch维度
        latent = latent.unsqueeze(0)  # [1, C, H', W']
        
        # 选择对应的解码器
        decoder = self.decoders[expert_id]
        
        # 解码为特征
        patch_features = decoder(latent)  # [1, output_channels, patch_size, patch_size]
        
        return patch_features.squeeze(0)  # [output_channels, patch_size, patch_size]
    
    def decode_patches_to_features(self, latents: List[Dict]) -> torch.Tensor:
        """
        批量解码所有patch并拼接成特征图
        
        Args:
            latents: List[Dict]，每个字典包含:
                - latent: [C, H', W'] patch的潜在表示
                - expert_id: int 分配的专家ID
                - position: (x, y) patch位置坐标
                
        Returns:
            feature_map: [1, output_channels, H, W] 完整特征图
        """
        if not latents:
            return torch.zeros(1, self.output_channels, 0, 0)
        
        device = latents[0]['latent'].device
        
        # 计算需要的图像尺寸（根据最大位置动态计算）
        xs = [p['position'][0] for p in latents]
        ys = [p['position'][1] for p in latents]
        max_x = max(xs)
        max_y = max(ys)
        image_w = (max_x + 1) * self.patch_size
        image_h = (max_y + 1) * self.patch_size
        
        # 初始化特征图缓冲区，使用output_channels
        feature_buffer = torch.zeros(1, self.output_channels, image_h, image_w, device=device)
        count_buffer = torch.zeros(1, 1, image_h, image_w, device=device)
        
        # 按专家分组以进行批量处理
        expert_groups = {i: {'latents': [], 'positions': [], 'indices': []} 
                        for i in range(self.num_experts)}
        
        # 分组
        for idx, patch_data in enumerate(latents):
            expert_id = patch_data['expert_id']
            expert_groups[expert_id]['latents'].append(patch_data['latent'])
            expert_groups[expert_id]['positions'].append(patch_data['position'])
            expert_groups[expert_id]['indices'].append(idx)
        
        # 逐个专家进行批量解码
        all_patches = [None] * len(latents)
        
        for expert_id in range(self.num_experts):
            if not expert_groups[expert_id]['latents']:
                continue
                
            # 获取该专家的所有数据
            expert_latents = expert_groups[expert_id]['latents']
            positions = expert_groups[expert_id]['positions']
            indices = expert_groups[expert_id]['indices']
            
            # 批量堆叠潜在表示
            latents_batch = torch.stack(expert_latents)  # [K, C, H', W']
            
            # 批量解码为特征
            decoder = self.decoders[expert_id]
            patches_batch = decoder(latents_batch)  # [K, output_channels, patch_size, patch_size]
            
            # 将解码后的patch特征存回对应位置
            for i, idx in enumerate(indices):
                all_patches[idx] = {
                    'patch_features': patches_batch[i],
                    'position': positions[i]
                }
        
        # 将所有patch特征拼接到特征图中
        for patch_data in all_patches:
            patch_features = patch_data['patch_features']  # [output_channels, patch_size, patch_size]
            x, y = patch_data['position']
            
            # 计算像素坐标
            x_start = x * self.patch_size
            y_start = y * self.patch_size
            x_end = x_start + self.patch_size
            y_end = y_start + self.patch_size
            
            # 将patch特征放入特征图缓冲区
            feature_buffer[0, :, y_start:y_end, x_start:x_end] += patch_features
            count_buffer[0, 0, y_start:y_end, x_start:x_end] += 1
        
        # 平均处理重叠区域
        feature_map = feature_buffer / torch.clamp(count_buffer, min=1)
        
        return feature_map
    
    def forward(self, latents: List[Dict]) -> torch.Tensor:
        """
        主前向传播函数：解码→特征图→重建图像
        
        Args:
            latents: List[Dict]，编码器的输出格式
            
        Returns:
            reconstructed_image: [1, 3, H, W] 重建的完整图像
        """
        # 1. 解码所有patch为特征图
        feature_map = self.decode_patches_to_features(latents)  # [1, output_channels, H, W]
        
        # 2. 通过重建头生成最终图像
        reconstructed_image = self.reconstruction_head(feature_map)  # [1, 3, H, W]
        
        return reconstructed_image