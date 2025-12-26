import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from pathlib import Path
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time
from datetime import datetime
import math
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.moe_conv2d.encoder import MoEPatchEncoder
from model.moe_conv2d.decoder import MoEPatchDecoder

class ImageDataset(Dataset):
    """图像数据集"""
    def __init__(self, image_dir, transform=None, target_divisor=64, max_samples=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.jpeg"))
        
        if not self.image_paths:
            raise ValueError(f"在 {image_dir} 中没有找到图像文件")
        
        # 限制数据集大小
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            
        print(f"找到 {len(self.image_paths)} 张图像")
        
        self.transform = transform
        self.target_divisor = target_divisor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """返回单张图像，不带batch维度"""
        try:
            # 加载图像
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # 转换为tensor并归一化到[0,1]
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            # 获取原始尺寸
            C, H, W = image.shape
            
            # 计算需要padding的大小
            pad_h = (self.target_divisor - H % self.target_divisor) % self.target_divisor
            pad_w = (self.target_divisor - W % self.target_divisor) % self.target_divisor
            
            # 初始化padding信息字典
            padding_info = {
                'original_shape': (H, W),
                'padded': False,
                'needs_unpad': False,
                'unpad_top': 0,
                'unpad_bottom': H,
                'unpad_left': 0,
                'unpad_right': W
            }
            
            if pad_h > 0 or pad_w > 0:
                # 需要padding
                padding = (pad_w // 2, pad_w - pad_w // 2, 
                          pad_h // 2, pad_h - pad_h // 2)
                
                # 使用反射padding
                image = F.pad(image.unsqueeze(0), padding, mode='reflect')[0]  # 去掉batch维度
                
                padding_info.update({
                    'padded': True,
                    'needs_unpad': True,
                    'unpad_top': pad_h // 2,
                    'unpad_bottom': H + pad_h // 2,  # padding后的位置
                    'unpad_left': pad_w // 2,
                    'unpad_right': W + pad_w // 2,
                    'padding': padding
                })
            
            return {
                'image': image,  # (C, H, W) 或 (C, H_pad, W_pad)
                'padding_info': padding_info,
                'image_path': str(img_path)
            }
        except Exception as e:
            print(f"加载图像 {self.image_paths[idx]} 时出错: {e}")
            # 返回一个空的占位符
            return {
                'image': torch.zeros((3, 256, 256), dtype=torch.float32),
                'padding_info': {
                    'original_shape': (256, 256),
                    'padded': False,
                    'needs_unpad': False,
                    'unpad_top': 0,
                    'unpad_bottom': 256,
                    'unpad_left': 0,
                    'unpad_right': 256
                },
                'image_path': 'error'
            }

def collate_fn(batch):
    """自定义collate函数，处理padding_info"""
    images = [item['image'] for item in batch]
    padding_infos = [item['padding_info'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    images_tensor = torch.stack(images, dim=0)
    
    return {
        'image': images_tensor,
        'padding_info': padding_infos,
        'image_path': image_paths
    }

class CombinedLoss(nn.Module):
    """组合损失函数：重建损失 + 负载均衡损失 + 压缩率损失"""
    def __init__(self, recon_weight=1.0, load_balance_weight=0.1, 
                 compression_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.load_balance_weight = load_balance_weight
        self.compression_weight = compression_weight  # 新增：压缩率损失权重
        
    def forward(self, pred, target, padding_infos, router_probs=None, 
                expert_ids=None, num_experts=None, latents_info=None):
        """
        计算总损失
        Args:
            pred: 重建图像 [B, 3, H, W]
            target: 原始图像 [B, 3, H, W]
            padding_infos: padding信息列表
            router_probs: 路由概率 [N_total, num_experts]
            expert_ids: 专家ID [N_total]
            num_experts: 专家数量
            latents_info: 包含潜在表示信息的列表，每个元素是一个字典，包含：
                - latent: [C, H', W'] patch的潜在表示
                - expert_id: int 分配的专家ID
                - position: (x, y) patch位置
        """
        recon_loss = self.compute_recon_loss(pred, target, padding_infos)
        total_loss = self.recon_weight * recon_loss
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'total_loss': total_loss.item()
        }
        
        # 负载均衡损失
        if router_probs is not None and expert_ids is not None and num_experts is not None:
            load_balance_loss = self.compute_load_balancing_loss(
                router_probs, expert_ids, num_experts
            )
            total_loss += self.load_balance_weight * load_balance_loss
            loss_dict['load_balance_loss'] = load_balance_loss.item()
            loss_dict['total_loss'] = total_loss.item()
        
        # 压缩率损失（新增）
        if latents_info is not None and self.compression_weight > 0:
            compression_loss = self.compute_compression_loss(
                target, padding_infos, latents_info
            )
            total_loss += self.compression_weight * compression_loss
            loss_dict['compression_loss'] = compression_loss.item()
            loss_dict['total_loss'] = total_loss.item()
        
        self.loss_dict = loss_dict
        return total_loss
    
    @staticmethod
    def compute_load_balancing_loss(router_probs, expert_ids, num_experts):
        N = router_probs.shape[0]
        expert_mask = F.one_hot(expert_ids, num_classes=num_experts).float()
        router_prob_sum = router_probs.sum(dim=0)
        expert_mask_sum = expert_mask.sum(dim=0)
        router_prob_mean = router_prob_sum / N
        expert_mask_mean = expert_mask_sum / N
        load_balancing_loss = torch.sum(router_prob_mean * expert_mask_mean) * num_experts
        return load_balancing_loss
    
    @staticmethod
    def compute_recon_loss(pred, target, padding_infos):
        """计算重建损失，考虑padding区域和锐度损失"""
        batch_size = pred.shape[0]
        total_recon_loss = 0
        
        for i in range(batch_size):
            pred_img = pred[i]
            target_img = target[i]
            info = padding_infos[i]
            
            # 提取有效区域（去padding）
            if info['needs_unpad']:
                top = info['unpad_top']
                bottom = info['unpad_bottom']
                left = info['unpad_left']
                right = info['unpad_right']
                
                H, W = pred_img.shape[1], pred_img.shape[2]
                bottom = min(bottom, H)
                right = min(right, W)
                
                if bottom > top and right > left:
                    pred_unpadded = pred_img[:, top:bottom, left:right]
                    target_unpadded = target_img[:, top:bottom, left:right]
                else:
                    pred_unpadded = pred_img
                    target_unpadded = target_img
            else:
                pred_unpadded = pred_img
                target_unpadded = target_img
            
            recon_loss = F.mse_loss(pred_unpadded, target_unpadded)
            total_recon_loss += recon_loss
            
        avg_recon_loss = total_recon_loss / batch_size if batch_size > 0 else 0
        
        return avg_recon_loss
    
    
@staticmethod
def compute_compression_loss(target, padding_infos, latents_info):
    """
    计算压缩率损失：鼓励更高的压缩率（支持梯度传播）
    
    实现思路：
    1. 使用可微分的方式估计潜在表示的大小
    2. 通过sigmoid逼近原始图像的"有效区域"大小
    3. 计算压缩比损失
    """
    device = target.device
    batch_size = target.shape[0]
    
    # 存储每张图像的损失
    compression_losses = []
    
    # 由于批次中只有一张图像，直接处理第一张
    for i in range(batch_size):
        # 获取当前图像
        target_img = target[i]  # [3, H, W]
        
        # 计算原始图像有效区域的大小（通过sigmoid逼近）
        # 使用target本身作为mask（值为0-1），乘以一个大常数后通过sigmoid
        mask = torch.sigmoid(target_img * 255)  # 放大后通过sigmoid，接近0或1
        original_size_estimate = mask.sum() * 24.0  # 假设每个像素24位 (3通道×8位)
        
        # 计算压缩后的大小
        compressed_size_estimate = torch.tensor(0.0, device=device)
        
        # 处理所有patch的潜在表示
        for patch_info in latents_info:
            latent = patch_info.get('latent')
            if latent is not None:
                # 使用可微分的方式估计潜在表示的大小
                # 将latent的值通过sigmoid转换为接近0或1，然后求和
                latent_mask = torch.sigmoid(latent * 1000)  # 放大使sigmoid接近二值化
                compressed_size_estimate += latent_mask.sum() * 32.0  # 假设32位浮点数
        
        # 避免除以零
        eps = 1e-8
        compression_ratio = original_size_estimate / (compressed_size_estimate + eps)
        
        # 损失 = -log(压缩率 + 1) 使得梯度更平滑
        compression_loss = -torch.log(compression_ratio + 1.0)
        compression_losses.append(compression_loss)
    
    # 返回平均损失
    if compression_losses:
        return torch.stack(compression_losses).mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
class Trainer:
    """MoE AutoEncoder训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 设置实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{timestamp}"
        
        # 创建保存目录
        self.save_dir = Path(config['save_dir']) / self.experiment_id
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.sample_dir = self.save_dir / "samples"
        self.stats_dir = self.save_dir / "stats"
        
        for dir_path in [self.save_dir, self.checkpoint_dir, self.sample_dir, self.stats_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"实验ID: {self.experiment_id}")
        print(f"保存目录: {self.save_dir}")
        
        # 模型配置
        self.latent_channels_list = config.get('latent_channels_list', [32, 64, 96])
        self.patch_size = config.get('patch_size', 16)
        self.num_experts = len(self.latent_channels_list)
        
        self.encoder = MoEPatchEncoder(
            latent_channels_list=self.latent_channels_list,
            patch_size=self.patch_size,
        ).to(self.device)
        
        self.decoder = MoEPatchDecoder(
            latent_channels_list=self.latent_channels_list,
            patch_size=self.patch_size,
        ).to(self.device)
        
        # 初始化优化器
        params = list(self.encoder.parameters()) + \
                list(self.decoder.parameters())
        
        self.optimizer = optim.AdamW(
            params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_lr']
        )
        
        # 损失函数
        self.criterion = CombinedLoss(
            recon_weight=config.get('recon_weight', 1.0),
            load_balance_weight=config.get('load_balance_weight', 0.1),
            compression_weight=config.get('compression_weight', 0.1),  # 新增
        )
                
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        
        # 专家统计信息（新增）
        self.expert_stats = {
            'total_patches': 0,
            'expert_counts': [0] * self.num_experts,
            'expert_usage_per_epoch': [],  # 记录每个epoch的专家使用情况
            'compression_ratios': []  # 记录压缩率
        }
        
        self.encoder.train()
        self.decoder.train()
        
        # 打印模型信息
        self.print_model_info()
    
    def print_model_info(self):
        """打印模型信息"""
        print("\n" + "="*50)
        print("模型配置信息:")
        print(f"  专家数量: {self.num_experts}")
        print(f"  潜在通道列表: {self.latent_channels_list}")
        print(f"  Patch大小: {self.patch_size}")
        print(f"  图像大小: dynamic (由输入图像决定)")
        print(f"  网格大小: dynamic (由输入图像和patch_size决定)")
        print(f"  每个图像的Patch数量: dynamic")
        
        print("\n模型参数数量:")
        print(f"  MoE编码器: {self.count_parameters(self.encoder):,}")
        print(f"  MoE解码器: {self.count_parameters(self.decoder):,}")
        total_params = (self.count_parameters(self.encoder) +
                       self.count_parameters(self.decoder))
        print(f"  总参数: {total_params:,}")
        print("="*50 + "\n")
    
    def count_parameters(self, model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def process_single_image(self, image):
        """
        处理单张图像
        Args:
            image: [1, 3, H, W]
        Returns:
            reconstructed: 重建图像
            router_probs: 路由概率
            expert_ids: 专家ID
            latents: 原始latents信息
        """
        
        # 编码
        latents = self.encoder(image)
        
        # 解码
        reconstructed = self.decoder(latents)
        
        # 提取路由信息
        router_probs = torch.stack([item['router_probs'] for item in latents])
        expert_ids = torch.tensor([item['expert_id'] for item in latents], device=image.device)
        
        return reconstructed, router_probs, expert_ids, latents

    def update_expert_stats(self, expert_ids, latents_info, target, padding_info):
        """
        更新专家统计信息（新增）
        Args:
            expert_ids: 当前批次的专家ID列表
            latents_info: 当前批次的潜在表示信息
            target: 目标图像
            padding_info: padding信息
        """
        # 统计专家使用次数
        for expert_id in expert_ids.cpu().numpy():
            self.expert_stats['expert_counts'][expert_id] += 1
            self.expert_stats['total_patches'] += 1
        
        # 计算压缩率
        if len(latents_info) > 0:
            # 计算原始图像的有效尺寸（去除padding）
            if padding_info['needs_unpad']:
                top = padding_info['unpad_top']
                bottom = padding_info['unpad_bottom']
                left = padding_info['unpad_left']
                right = padding_info['unpad_right']
                
                H, W = target.shape[1], target.shape[2]
                bottom = min(bottom, H)
                right = min(right, W)
                
                if bottom > top and right > left:
                    effective_H = bottom - top
                    effective_W = right - left
                else:
                    effective_H, effective_W = padding_info['original_shape']
            else:
                effective_H, effective_W = padding_info['original_shape']
            
            # 计算原始数据大小（比特数）
            original_bits = 3.0 * effective_H * effective_W * 8.0
            
            # 计算压缩后数据大小
            compressed_bits = 0.0
            for patch_info in latents_info:
                latent = patch_info.get('latent')
                if latent is not None:
                    C, H_latent, W_latent = latent.shape
                    compressed_bits += C * H_latent * W_latent * 32.0
            
            if compressed_bits > 0:
                compression_ratio = original_bits / compressed_bits
                self.expert_stats['compression_ratios'].append(compression_ratio)
    
    def print_expert_stats(self, epoch=None):
        """打印专家统计信息（新增）"""
        if self.expert_stats['total_patches'] == 0:
            return
        
        print("\n" + "="*50)
        print("专家使用情况统计:")
        if epoch is not None:
            print(f"Epoch {epoch} 统计结果:")
        
        total_patches = self.expert_stats['total_patches']
        print(f"总patch数: {total_patches}")
        print("\n各专家使用次数和比例:")
        
        for expert_id in range(self.num_experts):
            count = self.expert_stats['expert_counts'][expert_id]
            percentage = count / total_patches * 100 if total_patches > 0 else 0
            latent_channels = self.latent_channels_list[expert_id]
            print(f"  专家 {expert_id} (通道数: {latent_channels}): "
                  f"{count}次 ({percentage:.1f}%)")
        
        # 计算负载均衡指标
        avg_count = total_patches / self.num_experts if self.num_experts > 0 else 0
        imbalance = sum(abs(count - avg_count) for count in self.expert_stats['expert_counts']) / (2 * total_patches) if total_patches > 0 else 0
        print(f"\n负载不均衡度: {imbalance:.3f} (0表示完全均衡)")
        
        # 计算平均压缩率
        if self.expert_stats['compression_ratios']:
            avg_compression = np.mean(self.expert_stats['compression_ratios'])
            print(f"平均压缩率: {avg_compression:.2f}x")
        
        print("="*50)
    
    def save_expert_stats(self, epoch):
        """保存专家统计信息到文件（新增）"""
        stats_file = self.stats_dir / f'expert_stats_epoch_{epoch}.json'
        
        stats_data = {
            'epoch': epoch,
            'total_patches': self.expert_stats['total_patches'],
            'expert_counts': self.expert_stats['expert_counts'],
            'latent_channels_list': self.latent_channels_list,
            'num_experts': self.num_experts
        }
        
        # 计算百分比
        if stats_data['total_patches'] > 0:
            stats_data['expert_percentages'] = [
                count / stats_data['total_patches'] * 100 
                for count in stats_data['expert_counts']
            ]
        
        # 保存到文件
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        # 记录到epoch统计中
        self.expert_stats['expert_usage_per_epoch'].append({
            'epoch': epoch,
            'expert_counts': self.expert_stats['expert_counts'].copy(),
            'total_patches': self.expert_stats['total_patches']
        })
    
    def reset_expert_stats(self):
        """重置专家统计信息（新增）"""
        self.expert_stats['total_patches'] = 0
        self.expert_stats['expert_counts'] = [0] * self.num_experts
        self.expert_stats['compression_ratios'] = []
    
    def plot_expert_usage_history(self):
        """绘制专家使用历史图（新增）"""
        if not self.expert_stats['expert_usage_per_epoch']:
            return
        
        epochs = [stats['epoch'] for stats in self.expert_stats['expert_usage_per_epoch']]
        
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        ax1 = plt.subplot(2, 1, 1)
        
        # 绘制每个专家的使用百分比
        for expert_id in range(self.num_experts):
            percentages = []
            for stats in self.expert_stats['expert_usage_per_epoch']:
                if stats['total_patches'] > 0:
                    percentage = stats['expert_counts'][expert_id] / stats['total_patches'] * 100
                else:
                    percentage = 0
                percentages.append(percentage)
            
            ax1.plot(epochs, percentages, 'o-', linewidth=2, 
                    label=f'Expert {expert_id} (C={self.latent_channels_list[expert_id]})')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Expert Usage (%)')
        ax1.set_title('Expert Usage Percentage Over Training')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 绘制总patch数
        ax2 = plt.subplot(2, 1, 2)
        total_patches = [stats['total_patches'] for stats in self.expert_stats['expert_usage_per_epoch']]
        ax2.plot(epochs, total_patches, 's-', linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Total Patches Processed')
        ax2.set_title('Total Patches Processed Over Training')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        history_path = self.save_dir / 'expert_usage_history.png'
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"专家使用历史图已保存到: {history_path}")
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.encoder.train()
        self.decoder.train()
        
        # 重置专家统计信息
        self.reset_expert_stats()
        
        total_loss = 0
        total_recon_loss = 0
        total_load_balance_loss = 0
        total_compression_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            images = batch['image'].to(self.device)
            padding_infos = batch['padding_info']
            
            # 跳过空批次
            if images.shape[0] == 0:
                continue
            
            # 处理每张图像（由于MoE处理方式，需要逐张处理）
            batch_reconstructed = []
            batch_router_probs = []
            batch_expert_ids = []
            batch_latents_info = []

            for i in range(images.shape[0]):
                single_image = images[i:i+1]  # [1, 3, H, W]
                
                # 处理单张图像
                reconstructed, router_probs, expert_ids, latents_info = self.process_single_image(single_image)
                batch_reconstructed.append(reconstructed)
                batch_router_probs.append(router_probs)
                batch_expert_ids.append(expert_ids)
                batch_latents_info.extend(latents_info)
                
                # 更新专家统计信息（新增）
                self.update_expert_stats(expert_ids, latents_info, images[i], padding_infos[i])

            # 拼接结果
            reconstructed = torch.cat(batch_reconstructed, dim=0)  # [B, 3, H, W]
            router_probs = torch.cat(batch_router_probs, dim=0)    # [N_total, num_experts]
            expert_ids = torch.cat(batch_expert_ids, dim=0)        # [N_total]
            
            # 计算总损失
            loss = self.criterion(
                reconstructed, images, padding_infos,
                router_probs=router_probs,
                expert_ids=expert_ids,
                num_experts=len(self.latent_channels_list),
                latents_info=batch_latents_info
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + 
                    list(self.decoder.parameters()), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            total_recon_loss += self.criterion.loss_dict.get('recon_loss', 0)
            total_load_balance_loss += self.criterion.loss_dict.get('load_balance_loss', 0)
            total_compression_loss += self.criterion.loss_dict.get('compression_loss', 0)
            num_batches += 1
            
            # 更新进度条
            postfix = {
                'loss': loss.item(),
                'recon': self.criterion.loss_dict.get('recon_loss', 0),
                'balance': self.criterion.loss_dict.get('load_balance_loss', 0),
                'compression': self.criterion.loss_dict.get('compression_loss', 0),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            pbar.set_postfix(postfix)
            
            # 保存示例图像
            if batch_idx == 0 and epoch % self.config['save_image_every'] == 0:
                self.save_sample_images(
                    images[0], 
                    reconstructed[0], 
                    padding_infos[0], 
                    epoch, 
                    batch_idx
                )
        
        # 计算平均损失
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        # 打印并保存专家统计信息
        self.print_expert_stats(epoch)
        self.save_expert_stats(epoch)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': total_recon_loss / max(num_batches, 1),
            'load_balance_loss': total_load_balance_loss / max(num_batches, 1),
            'compression_loss': total_compression_loss / max(num_batches, 1)
        }
    
    def save_sample_images(self, original, reconstructed, padding_info, epoch, batch_idx):
        """保存原始图像和重建图像的对比"""
        sample_save_path = self.sample_dir / f'sample_epoch_{epoch}_batch_{batch_idx}.png'
        
        original = original.detach().cpu()
        reconstructed = reconstructed.detach().cpu()
        
        # 移除padding（如果存在）
        if padding_info['needs_unpad']:
            top = padding_info['unpad_top']
            bottom = padding_info['unpad_bottom']
            left = padding_info['unpad_left']
            right = padding_info['unpad_right']
            
            H, W = original.shape[1], original.shape[2]
            bottom = min(bottom, H)
            right = min(right, W)
            
            if bottom > top and right > left:
                original = original[:, top:bottom, left:right]
                reconstructed = reconstructed[:, top:bottom, left:right]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 确保在[0,1]范围内
        original = torch.clamp(original, 0, 1)
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        # 转换为numpy
        original_np = original.permute(1, 2, 0).numpy()
        reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
        
        axes[0].imshow(original_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed_np)
        axes[1].set_title(f'Reconstructed (Epoch {epoch})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(sample_save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'config': self.config,

            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            
            # 保存专家统计信息（新增）
            'expert_stats': self.expert_stats,
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"保存最佳模型，损失: {loss:.6f}")
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint['train_losses']
            
            # 加载专家统计信息（新增）
            if 'expert_stats' in checkpoint:
                self.expert_stats = checkpoint['expert_stats']
            
            print(f"从 {checkpoint_path} 加载检查点，epoch={self.current_epoch}, loss={checkpoint['loss']:.6f}")
            return True
        else:
            print(f"检查点文件不存在: {checkpoint_path}")
            return False
    
    def plot_training_history(self):
        """绘制训练历史"""
        if len(self.train_losses) > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(self.train_losses, 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Total Loss')
            plt.title(f'Total Training Loss\nBest: {self.best_loss:.6f}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            # 绘制最近100个epoch的损失
            recent_losses = self.train_losses[-100:] if len(self.train_losses) > 100 else self.train_losses
            plt.plot(recent_losses, 'r-', linewidth=2)
            plt.xlabel('Epoch (recent)')
            plt.ylabel('Total Loss')
            plt.title(f'Recent Total Loss\nCurrent: {self.train_losses[-1]:.6f}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            # 绘制专家配置
            expert_labels = [f'E{i}(C={c})' for i, c in enumerate(self.latent_channels_list)]
            plt.bar(expert_labels, self.latent_channels_list)
            plt.xlabel('Expert')
            plt.ylabel('Latent Channels')
            plt.title('Expert Configurations')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            # 绘制当前专家使用情况
            if self.expert_stats['total_patches'] > 0:
                expert_percentages = [
                    count / self.expert_stats['total_patches'] * 100 
                    for count in self.expert_stats['expert_counts']
                ]
                plt.bar(expert_labels, expert_percentages)
                plt.xlabel('Expert')
                plt.ylabel('Usage (%)')
                plt.title(f'Current Expert Usage\nTotal Patches: {self.expert_stats["total_patches"]:,}')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            history_path = self.save_dir / 'training_history.png'
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"训练历史图已保存到: {history_path}")
            
            # 绘制专家使用历史图
            self.plot_expert_usage_history()
    
    def train(self, train_loader):
        """训练主循环"""
        print(f"开始训练MoE AutoEncoder，共 {self.config['num_epochs']} 个epoch...")
        print(f"设备: {self.device}")
        print(f"批量大小: {train_loader.batch_size}")
        print(f"数据集大小: {len(train_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch + 1, self.config['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # 训练一个epoch
            epoch_result = self.train_epoch(train_loader, epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印epoch总结
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}:")
            print(f"  Total Loss: {epoch_result['total_loss']:.6f}")
            print(f"  Recon Loss: {epoch_result['recon_loss']:.6f}")
            print(f"  Load Balance Loss: {epoch_result['load_balance_loss']:.6f}")
            print(f"  Compression Loss: {epoch_result['compression_loss']:.6f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存检查点
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, epoch_result['total_loss'], is_best=False)
            
            # 保存最佳模型
            if epoch_result['total_loss'] < self.best_loss:
                self.best_loss = epoch_result['total_loss']
                self.save_checkpoint(epoch, epoch_result['total_loss'], is_best=True)
            
            # 绘制训练历史
            if epoch % self.config['save_image_every'] == 0:
                self.plot_training_history()
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"总训练时间: {training_time:.2f}秒")
        print(f"最佳损失: {self.best_loss:.6f}")
        print(f"最终损失: {self.train_losses[-1] if self.train_losses else 0:.6f}")
        
        # 保存最终模型
        self.save_checkpoint(self.current_epoch, self.train_losses[-1] if self.train_losses else 0)
        self.plot_training_history()
        
        # 最终专家使用情况分析
        print("\n最终专家使用情况分析:")
        self.print_expert_stats()
        
        return self.train_losses

def main():
    model_type = 'moe_conv2d'

    config = {
        'model_type': model_type,
        
        # 数据配置
        'image_dir': './dataset/train/',
        'batch_size': 1,  # MoE模型需要较小batch size
        'num_workers': 0,
        
        # 模型配置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'latent_channels_list': [4, 6, 8, 10],  # 4个专家，不同通道数
        'patch_size': 64,
        
        # 训练配置
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'min_lr': 1e-5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # 损失函数配置
        'recon_weight': 1000,
        'load_balance_weight': 1.8,
        'compression_weight': 1,
        
        # 保存配置
        'save_dir': f'./checkpoints/{model_type}',
        'save_every': 50,
        'save_image_every': 5,
        
        # 数据集配置
        'max_samples': 80,
    }
    
    transform = None
    
    # 创建数据集
    dataset = ImageDataset(
        image_dir=config['image_dir'],
        transform=transform,
        target_divisor=64,
        max_samples=config['max_samples']
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['device'] == 'cuda',
        collate_fn=collate_fn
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 可选: 从检查点恢复训练
    resume_from_checkpoint = None  # 设置为检查点路径以恢复训练
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        if trainer.load_checkpoint(resume_from_checkpoint):
            print(f"从检查点恢复训练: {resume_from_checkpoint}")
    
    # 开始训练
    trainer.train(dataloader)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()