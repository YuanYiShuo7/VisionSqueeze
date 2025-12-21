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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.atf2d.encoder import Encoder
from model.atf2d.decoder import Decoder

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
                'image': torch.zeros((3, 64, 64), dtype=torch.float32),
                'padding_info': {
                    'original_shape': (64, 64),
                    'padded': False,
                    'needs_unpad': False,
                    'unpad_top': 0,
                    'unpad_bottom': 64,
                    'unpad_left': 0,
                    'unpad_right': 64
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

class EnhancedMSELoss(nn.Module):
    """增强的MSE损失，避免值太小"""
    def __init__(self, scale_factor=1000):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        return mse * self.scale_factor

class CombinedImageLoss(nn.Module):
    """组合图像损失函数"""
    def __init__(self, mse_weight=1.0, l1_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        
    def forward(self, pred, target):
        # MSE损失（像素级精度）
        mse_loss = F.mse_loss(pred, target) * 1000  # 放大
        
        # L1损失（更鲁棒）
        l1_loss = F.l1_loss(pred, target)
        
        # 总损失
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        # 记录各分量
        self.loss_dict = {
            'mse_loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss

class AutoEncoderTrainer:
    """完整的AutoEncoder训练器（同时训练Encoder和Decoder）"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 设置实验ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{timestamp}_{config.get('experiment_name', 'autoencoder')}"
        
        # 创建保存目录
        self.save_dir = Path(config['save_dir']) / self.experiment_id
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.sample_dir = self.save_dir / "samples"
        
        for dir_path in [self.save_dir, self.checkpoint_dir, self.sample_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"实验ID: {self.experiment_id}")
        print(f"保存目录: {self.save_dir}")
        
        # 初始化模型
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        
        # 初始化优化器
        encoder_params = list(self.encoder.parameters())
        decoder_params = list(self.decoder.parameters())
        
        if config.get('freeze_encoder', False):
            # 冻结Encoder，只训练Decoder
            print("冻结Encoder，只训练Decoder")
            for param in encoder_params:
                param.requires_grad = False
            trainable_params = decoder_params
        else:
            # 同时训练Encoder和Decoder
            print("同时训练Encoder和Decoder")
            trainable_params = encoder_params + decoder_params
        
        self.optimizer = optim.AdamW(
            trainable_params,
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
        if config.get('use_combined_loss', True):
            self.criterion = CombinedImageLoss(
                mse_weight=config.get('mse_weight', 1.0),
                l1_weight=config.get('l1_weight', 0.5)
            )
        else:
            self.criterion = EnhancedMSELoss(scale_factor=10)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        
        # 设置模型为训练模式
        self.encoder.train()
        self.decoder.train()
        
        print(f"初始化完成: Encoder参数={self.count_parameters(self.encoder):,}, "
              f"Decoder参数={self.count_parameters(self.decoder):,}")
    
    def count_parameters(self, model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def compute_loss(self, pred, target, padding_infos):
        """计算损失，考虑padding区域"""
        batch_size = pred.shape[0]
        total_loss = 0
        
        for i in range(batch_size):
            pred_img = pred[i]
            target_img = target[i]
            info = padding_infos[i]
            
            if info['needs_unpad']:
                # 提取非padding区域
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
                    loss = self.criterion(pred_unpadded.unsqueeze(0), target_unpadded.unsqueeze(0))
                else:
                    loss = self.criterion(pred_img.unsqueeze(0), target_img.unsqueeze(0))
            else:
                loss = self.criterion(pred_img.unsqueeze(0), target_img.unsqueeze(0))
            
            total_loss += loss
        
        return total_loss / batch_size if batch_size > 0 else 0
    
    def save_sample_images(self, original, reconstructed, padding_info, epoch, save_path):
        """保存原始图像和重建图像的对比"""
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        total_mse_loss = 0
        total_l1_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            images = batch['image'].to(self.device)
            padding_infos = batch['padding_info']
            
            # 跳过空批次
            if images.shape[0] == 0:
                continue
            
            # 前向传播
            features = self.encoder(images)
            reconstructed = self.decoder(features)
            
            # 计算损失
            loss = self.compute_loss(reconstructed, images, padding_infos)
            
            # 记录各分量损失
            if hasattr(self.criterion, 'loss_dict'):
                total_mse_loss += self.criterion.loss_dict.get('mse_loss', 0)
                total_l1_loss += self.criterion.loss_dict.get('l1_loss', 0)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            postfix = {
                'loss': loss.item(),
                'avg_loss': total_loss / max(num_batches, 1),
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            if hasattr(self.criterion, 'loss_dict'):
                postfix['mse'] = self.criterion.loss_dict.get('mse_loss', 0)
                postfix['l1'] = self.criterion.loss_dict.get('l1_loss', 0)
            
            pbar.set_postfix(postfix)
            
            # 保存示例图像
            if batch_idx == 0 and epoch % self.config['save_image_every'] == 0:
                sample_save_path = self.sample_dir / f'sample_epoch_{epoch}.png'
                self.save_sample_images(
                    images[0], 
                    reconstructed[0], 
                    padding_infos[0], 
                    epoch, 
                    sample_save_path
                )
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'mse_loss': total_mse_loss / max(num_batches, 1) if hasattr(self.criterion, 'loss_dict') else 0,
            'l1_loss': total_l1_loss / max(num_batches, 1) if hasattr(self.criterion, 'loss_dict') else 0
        }
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        if is_best:
            best_encoder_path = self.checkpoint_dir / 'best_encoder.pth'
            torch.save({'encoder_state_dict': self.encoder.state_dict()}, best_encoder_path)
            
            best_decoder_path = self.checkpoint_dir / 'best_decoder.pth'
            torch.save({'decoder_state_dict': self.decoder.state_dict()}, best_decoder_path)
            
            print(f"保存最佳模型，损失: {loss:.6f}")

        else:
            encoder_path = self.checkpoint_dir / f'encoder_epoch_{epoch}.pth'
            torch.save({
                'encoder_state_dict': self.encoder.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': self.config
            }, encoder_path)
            
            # 单独保存Decoder权重
            decoder_path = self.checkpoint_dir / f'decoder_epoch_{epoch}.pth'
            torch.save({
                'decoder_state_dict': self.decoder.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': self.config
            }, decoder_path)
            print(f"单独保存Encoder权重: {encoder_path}")
            print(f"单独保存Decoder权重: {decoder_path}")
    
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
            self.train_losses = checkpoint.get('train_losses', [])
            
            print(f"从 {checkpoint_path} 加载检查点，epoch={self.current_epoch}, loss={checkpoint['loss']:.6f}")
            return True
        else:
            print(f"检查点文件不存在: {checkpoint_path}")
            return False
    
    def plot_training_history(self):
        """绘制训练历史"""
        if len(self.train_losses) > 0:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss\nBest: {self.best_loss:.6f}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            # 绘制最近100个epoch的损失
            recent_losses = self.train_losses[-100:] if len(self.train_losses) > 100 else self.train_losses
            plt.plot(recent_losses, 'r-', linewidth=2)
            plt.xlabel('Epoch (recent)')
            plt.ylabel('Loss')
            plt.title(f'Recent Training Loss\nCurrent: {self.train_losses[-1]:.6f}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            history_path = self.save_dir / 'training_history.png'
            plt.savefig(history_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"训练历史图已保存到: {history_path}")
    
    def train(self, train_loader):
        """训练主循环"""
        print(f"开始训练AutoEncoder，共 {self.config['num_epochs']} 个epoch...")
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
            print(f"Epoch {epoch}/{self.config['num_epochs']}, "
                  f"Loss: {epoch_result['loss']:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if hasattr(self.criterion, 'loss_dict'):
                print(f"  MSE Loss: {epoch_result['mse_loss']:.6f}, "
                      f"L1 Loss: {epoch_result['l1_loss']:.6f}")
            
            # 保存检查点
            if epoch % self.config['save_every'] == 0:
                self.save_checkpoint(epoch, epoch_result['loss'], is_best=False)
            
            # 保存最佳模型
            if epoch_result['loss'] < self.best_loss:
                self.best_loss = epoch_result['loss']
                self.save_checkpoint(epoch, epoch_result['loss'], is_best=True)
            
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
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  Encoder参数量: {self.count_parameters(self.encoder):,}")
        print(f"  Decoder参数量: {self.count_parameters(self.decoder):,}")
        print(f"  总参数量: {self.count_parameters(self.encoder) + self.count_parameters(self.decoder):,}")
        
        return self.train_losses

def main():
    model_type = 'aft'

    config = {

        'model_type': model_type,
        
        # 数据配置
        'image_dir': './dataset/001/images',
        'batch_size': 1,
        'num_workers': 0,
        
        # 模型配置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'freeze_encoder': False,
        
        # 训练配置
        'num_epochs': 500,
        'learning_rate': 1e-3,
        'min_lr': 1e-5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # 损失函数配置
        'use_combined_loss': True,
        'mse_weight': 1.0,
        'l1_weight': 0.5,
        
        # 保存配置 - 使用动态命名
        'save_dir': f'./{model_type}_model_checkpoints',
        'save_every': 100,
        'save_image_every': 100,
        
        # 数据集配置
        'max_samples': 1,
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
    trainer = AutoEncoderTrainer(config)
    
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