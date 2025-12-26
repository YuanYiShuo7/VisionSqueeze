import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import sys
from datetime import datetime
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 根据你的实际路径导入模型
from model.conv2d.encoder import Encoder
from model.conv2d.decoder import Decoder

class ImageDatasetForEvaluation:
    """评估用图像数据集"""
    def __init__(self, image_paths, transform=None, target_divisor=64):
        self.image_paths = image_paths
        self.transform = transform
        self.target_divisor = target_divisor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """加载并预处理图像"""
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
                'unpad_right': W,
                'file_path': str(img_path)
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
                    'unpad_bottom': H + pad_h // 2,
                    'unpad_left': pad_w // 2,
                    'unpad_right': W + pad_w // 2,
                    'padding': padding
                })
            
            return {
                'image': image,  # (C, H, W)
                'padding_info': padding_info
            }
        except Exception as e:
            print(f"加载图像 {self.image_paths[idx]} 时出错: {e}")
            return None

class AutoEncoderEvaluator:
    """AutoEncoder评估器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 创建结果保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.evaluation_id = f"{timestamp}"
        
        # 设置保存路径
        self.results_dir = Path(config['results_dir']) / self.evaluation_id
        self.logs_dir = self.results_dir / "logs"
        self.samples_dir = self.results_dir / "samples"
        
        for dir_path in [self.results_dir, self.logs_dir, self.samples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"评估ID: {self.evaluation_id}")
        print(f"结果保存目录: {self.results_dir}")
        
        # 加载模型
        self.encoder = self.load_encoder(config['encoder_path']).to(self.device)
        self.decoder = self.load_decoder(config['decoder_path']).to(self.device)
        
        # 设置为评估模式
        self.encoder.eval()
        self.decoder.eval()
        
        # 计算模型参数量
        self.encoder_params = self.count_parameters(self.encoder)
        self.decoder_params = self.count_parameters(self.decoder)
        
        print(f"模型加载完成:")
        print(f"  Encoder参数: {self.encoder_params:,}")
        print(f"  Decoder参数: {self.decoder_params:,}")
        
    def load_encoder(self, encoder_path):
        """加载Encoder模型"""
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder权重文件不存在: {encoder_path}")
        
        # 加载模型配置
        checkpoint = torch.load(encoder_path, map_location='cpu')
        
        # 创建模型实例
        encoder = Encoder()
        
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        
        print(f"Encoder从 {encoder_path} 加载成功")
        return encoder
    
    def load_decoder(self, decoder_path):
        """加载Decoder模型"""
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder权重文件不存在: {decoder_path}")
        
        # 加载模型配置
        checkpoint = torch.load(decoder_path, map_location='cpu')
        
        # 创建模型实例
        decoder = Decoder()
        
        if 'decoder_state_dict' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        elif 'state_dict' in checkpoint:
            decoder.load_state_dict(checkpoint['state_dict'])
        else:
            decoder.load_state_dict(checkpoint)
        
        print(f"Decoder从 {decoder_path} 加载成功")
        return decoder
    
    def count_parameters(self, model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters())
    
    def compute_metrics(self, original, reconstructed, latent_shape):
        """计算各项指标"""
        metrics = {}
        
        # 1. 压缩率指标
        original_size = original.numel() * original.element_size()  # 字节数
        latent_size = latent_shape[1] * latent_shape[2] * latent_shape[3] * 4  # 假设float32 (4字节)
        
        metrics['original_size_bytes'] = original_size
        metrics['latent_size_bytes'] = latent_size
        metrics['compression_ratio'] = original_size / latent_size if latent_size > 0 else 0
        
        # 2. 重建质量指标（在[0,1]范围内）
        # MSE
        metrics['mse'] = F.mse_loss(original, reconstructed).item()
        
        # PSNR
        if metrics['mse'] > 0:
            metrics['psnr'] = 20 * np.log10(1.0 / np.sqrt(metrics['mse']))
        else:
            metrics['psnr'] = float('inf')
        
        # SSIM (简化版本)
        metrics['ssim'] = self.compute_ssim_simple(original, reconstructed)
        
        # 3. 结构相似性指标
        metrics['l1'] = F.l1_loss(original, reconstructed).item()
        
        return metrics
    
    def compute_ssim_simple(self, img1, img2, window_size=11):
        """简化的SSIM计算"""
        from math import exp
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 转换为numpy
        img1_np = img1.cpu().numpy().squeeze()
        img2_np = img2.cpu().numpy().squeeze()
        
        # 如果是灰度图，确保是2D
        if len(img1_np.shape) == 3 and img1_np.shape[0] == 1:
            img1_np = img1_np[0]
            img2_np = img2_np[0]
        
        # 计算均值和方差
        mu1 = np.mean(img1_np)
        mu2 = np.mean(img2_np)
        
        sigma1_sq = np.var(img1_np)
        sigma2_sq = np.var(img2_np)
        sigma12 = np.cov(img1_np.flatten(), img2_np.flatten())[0, 1]
        
        # 计算SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        return numerator / denominator if denominator != 0 else 0
    
    def save_comparison_image(self, original, reconstructed, padding_info, 
                             metrics, save_path, filename):
        """保存对比图像"""
        # 移除batch维度
        original = original.squeeze(0).detach().cpu()
        reconstructed = reconstructed.squeeze(0).detach().cpu()
        
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
        
        # 确保在[0,1]范围内
        original = torch.clamp(original, 0, 1)
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 转换为numpy用于显示
        original_np = original.permute(1, 2, 0).numpy()
        reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
        
        # 计算差异图
        diff = np.abs(original_np - reconstructed_np)
        diff_normalized = diff / diff.max() if diff.max() > 0 else diff
        
        # 显示原始图像
        axes[0].imshow(original_np)
        axes[0].set_title(f'Original\n{original.shape[1]}x{original.shape[2]}')
        axes[0].axis('off')
        
        # 显示重建图像
        axes[1].imshow(reconstructed_np)
        axes[1].set_title(f'Reconstructed\nPSNR: {metrics["psnr"]:.2f} dB')
        axes[1].axis('off')
        
        # 显示差异图
        im_diff = axes[2].imshow(diff_normalized, cmap='hot')
        axes[2].set_title(f'Difference\nMSE: {metrics["mse"]:.6f}')
        axes[2].axis('off')
        
        # 添加颜色条
        plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 添加整体标题
        compression_text = f'压缩率: {metrics["compression_ratio"]:.2f}x'
        ssim_text = f'SSIM: {metrics["ssim"]:.4f}'
        plt.suptitle(f'{filename}\n{compression_text} | {ssim_text}', 
                     fontsize=12, y=1.02)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate_single_image(self, image_path, save_comparison=True):
        """评估单张图像"""
        print(f"\n评估单张图像: {image_path}")
        
        # 创建数据集
        dataset = ImageDatasetForEvaluation(
            image_paths=[image_path],
            transform=transforms.ToTensor(),
            target_divisor=64
        )
        
        # 获取图像数据
        data = dataset[0]
        if data is None:
            print(f"无法加载图像: {image_path}")
            return None
        
        original_image = data['image'].unsqueeze(0).to(self.device)  # 添加batch维度
        padding_info = data['padding_info']
        
        with torch.no_grad():
            # 编码
            latent = self.encoder(original_image)
            
            # 解码
            reconstructed = self.decoder(latent)
            
            # 计算指标
            metrics = self.compute_metrics(original_image, reconstructed, latent.shape)
            
            # 添加额外信息
            filename = Path(image_path).name
            metrics['filename'] = filename
            metrics['original_shape'] = padding_info['original_shape']
            metrics['padded'] = padding_info['padded']
            
            # 保存对比图
            if save_comparison:
                save_path = self.samples_dir / f'comparison_{filename}.png'
                self.save_comparison_image(
                    original_image, reconstructed, padding_info, 
                    metrics, save_path, filename
                )
                print(f"对比图已保存: {save_path}")
            
            return metrics
    
    def evaluate_multiple_images(self, image_dir, max_images=None, save_samples=True):
        """评估多张图像"""
        print(f"\n评估多张图像: {image_dir}")
        
        # 收集所有图像文件
        image_dir = Path(image_dir)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(ext)))
        
        # 限制图像数量
        if max_images is not None and max_images > 0:
            image_paths = image_paths[:max_images]
        
        if not image_paths:
            print(f"在 {image_dir} 中未找到图像文件")
            return None
        
        print(f"找到 {len(image_paths)} 张图像")
        
        all_metrics = []
        
        for img_path in tqdm(image_paths, desc="评估图像"):
            metrics = self.evaluate_single_image(
                str(img_path), 
                save_comparison=save_samples
            )
            
            if metrics:
                all_metrics.append(metrics)
        
        # 计算整体统计
        if all_metrics:
            summary = self.compute_summary_statistics(all_metrics)
            return {
                'individual_results': all_metrics,
                'summary': summary
            }
        
        return None
    
    def compute_summary_statistics(self, all_metrics):
        """计算整体统计信息"""
        summary = {}
        
        # 提取所有指标
        metrics_to_aggregate = ['compression_ratio', 'mse', 'psnr', 'ssim', 'l1']
        
        for metric_name in metrics_to_aggregate:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_min'] = np.min(values)
                summary[f'{metric_name}_max'] = np.max(values)
                summary[f'{metric_name}_median'] = np.median(values)
        
        # 模型信息
        summary['encoder_params'] = self.encoder_params
        summary['decoder_params'] = self.decoder_params
        summary['total_params'] = self.encoder_params + self.decoder_params
        summary['num_images_evaluated'] = len(all_metrics)
        
        return summary
    
    def save_results(self, results, output_format='json'):
        """保存评估结果"""
        if results is None:
            print("无结果可保存")
            return
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式结果
        json_path = self.logs_dir / f'evaluation_results_{timestamp}.json'
        
        # 将numpy类型转换为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 转换结果
        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"详细结果已保存: {json_path}")
        
        # 保存文本摘要
        txt_path = self.logs_dir / f'evaluation_summary_{timestamp}.txt'
        self.save_text_summary(results, txt_path)
        
        return json_path, txt_path
    
    def save_text_summary(self, results, txt_path):
        """保存文本摘要"""
        if 'summary' not in results:
            return
        
        summary = results['summary']
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("AutoEncoder 评估结果摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估ID: {self.evaluation_id}\n\n")
            
            f.write("模型信息:\n")
            f.write(f"  Encoder参数量: {summary['encoder_params']:,}\n")
            f.write(f"  Decoder参数量: {summary['decoder_params']:,}\n")
            f.write(f"  总参数量: {summary['total_params']:,}\n\n")
            
            f.write(f"评估图像数量: {summary['num_images_evaluated']}\n\n")
            
            f.write("性能指标统计:\n")
            f.write("-" * 60 + "\n")
            
            metrics_display = {
                'compression_ratio': '压缩率 (x)',
                'psnr': 'PSNR (dB)',
                'ssim': 'SSIM',
                'mse': 'MSE',
                'l1': 'L1 Loss'
            }
            
            for metric_key, metric_name in metrics_display.items():
                if f'{metric_key}_mean' in summary:
                    f.write(f"{metric_name}:\n")
                    f.write(f"  均值: {summary[f'{metric_key}_mean']:.4f}\n")
                    f.write(f"  标准差: {summary[f'{metric_key}_std']:.4f}\n")
                    f.write(f"  最小值: {summary[f'{metric_key}_min']:.4f}\n")
                    f.write(f"  最大值: {summary[f'{metric_key}_max']:.4f}\n")
                    f.write(f"  中位数: {summary[f'{metric_key}_median']:.4f}\n")
                    f.write("\n")
            
            f.write("=" * 60 + "\n")
        
        print(f"文本摘要已保存: {txt_path}")
    
    def plot_metrics_distribution(self, results):
        """绘制指标分布图"""
        if 'individual_results' not in results:
            return
        
        individual_results = results['individual_results']
        
        # 提取指标
        metrics_data = {}
        metrics_names = ['compression_ratio', 'psnr', 'ssim']
        
        for metric_name in metrics_names:
            values = [r[metric_name] for r in individual_results if metric_name in r]
            if values:
                metrics_data[metric_name] = values
        
        if not metrics_data:
            return
        
        # 创建子图
        n_metrics = len(metrics_data)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        # 英文标题和标签
        titles = {
            'compression_ratio': 'Compression Ratio Distribution',
            'psnr': 'PSNR Distribution',
            'ssim': 'SSIM Distribution'
        }
        
        # x轴标签
        x_labels = {
            'compression_ratio': 'Compression Ratio (x)',
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM'
        }
        
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            
            # 绘制直方图
            n, bins, patches = ax.hist(values, bins=20, alpha=0.7, color=colors[idx], 
                                    edgecolor='black')
            
            # 添加均值和标注
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1)
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1)
            
            ax.fill_betweenx([0, max(n)], mean_val - std_val, mean_val + std_val, 
                            alpha=0.2, color='orange', label='±1 std dev')
            
            ax.set_xlabel(x_labels.get(metric_name, metric_name))
            ax.set_ylabel('Frequency')
            ax.set_title(titles.get(metric_name, metric_name))
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('AutoEncoder Evaluation Metrics Distribution', fontsize=14, y=1.05)
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.logs_dir / 'metrics_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics distribution plot saved: {plot_path}")
        return plot_path

def main():
    parser = argparse.ArgumentParser(description='AutoEncoder评估脚本')
    parser.add_argument('--encoder_path', type=str, default="./conv2d_model_checkpoints/20251221_210220/checkpoints/best_encoder.pth",
                       help='Encoder权重文件路径')
    parser.add_argument('--decoder_path', type=str, default="./conv2d_model_checkpoints/20251221_210220/checkpoints/best_decoder.pth",
                       help='Decoder权重文件路径')
    parser.add_argument('--image_path', type=str, default="./dataset/eval/images/00000120.jpg",
                       help='单张图像路径（用于单张图像评估）')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图像目录路径（用于多张图像评估）')
    parser.add_argument('--max_images', type=int, default=100,
                       help='最大评估图像数量（仅用于多图像评估）')
    parser.add_argument('--model_type', type=str, default='conv2d',
                       help='模型类型')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'device': args.device,
        'encoder_path': args.encoder_path,
        'decoder_path': args.decoder_path,
        'model_type': args.model_type,
        'results_dir': f'./eval_results/{args.model_type}',
    }
    
    # 创建评估器
    evaluator = AutoEncoderEvaluator(config)
    
    results = None
    
    # 根据参数选择评估模式
    if args.image_path is not None:
        # 单张图像评估
        metrics = evaluator.evaluate_single_image(args.image_path, save_comparison=True)
        
        if metrics:
            results = {
                'individual_results': [metrics],
                'summary': evaluator.compute_summary_statistics([metrics])
            }
    elif args.image_dir is not None:
        # 多张图像评估
        results = evaluator.evaluate_multiple_images(
            args.image_dir, 
            max_images=args.max_images,
            save_samples=True
        )
    else:
        print("请提供 --image_path 或 --image_dir 参数")
        return
    
    # 保存结果
    if results:
        evaluator.save_results(results)
        
        # 绘制指标分布图
        if len(results.get('individual_results', [])) > 1:
            evaluator.plot_metrics_distribution(results)
        
        # 打印摘要
        summary = results.get('summary', {})
        if summary:
            print("\n" + "=" * 60)
            print("评估摘要:")
            print("=" * 60)
            print(f"评估图像数量: {summary.get('num_images_evaluated', 0)}")
            print(f"平均压缩率: {summary.get('compression_ratio_mean', 0):.2f}x")
            print(f"平均PSNR: {summary.get('psnr_mean', 0):.2f} dB")
            print(f"平均SSIM: {summary.get('ssim_mean', 0):.4f}")
            print(f"平均MSE: {summary.get('mse_mean', 0):.6f}")
            print("=" * 60)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()