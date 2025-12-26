#!/usr/bin/env python3
"""
JPEG图像压缩工具
基于PIL/Pillow库，提供灵活的JPEG压缩功能
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Union, List, Optional, Tuple
from enum import Enum
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings

try:
    from PIL import Image
except ImportError:
    print("错误: 需要安装Pillow库。请运行: pip install Pillow")
    sys.exit(1)


class CompressionMode(Enum):
    """压缩模式"""
    QUALITY = "quality"          # 基于质量因子
    FILESIZE = "filesize"        # 基于目标文件大小
    RESOLUTION = "resolution"    # 基于分辨率
    AUTO = "auto"                # 自动模式


@dataclass
class CompressionResult:
    """压缩结果"""
    input_path: str
    output_path: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    quality_used: int
    resolution_before: Tuple[int, int]
    resolution_after: Tuple[int, int]
    format: str
    compression_time: float
    success: bool
    error_message: Optional[str] = None


class JPEGCompressor:
    """JPEG压缩器"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.jpe', '.jfif',
        '.png', '.bmp', '.tiff', '.tif', 
        '.gif', '.webp', '.ico', '.ppm'
    }
    
    def __init__(
        self,
        quality: int = 85,
        optimize: bool = True,
        progressive: bool = False,
        subsampling: Optional[str] = None,
        dpi: Optional[Tuple[int, int]] = None,
        keep_exif: bool = True,
        strip_metadata: bool = False,
        verbose: bool = False
    ):
        """
        初始化JPEG压缩器
        
        Args:
            quality: JPEG质量 (1-100, 95为高质量, 75为中质量, 50为低质量)
            optimize: 是否优化编码
            progressive: 是否使用渐进式JPEG
            subsampling: 色度子采样 ('4:4:4', '4:2:2', '4:2:0', '4:1:1')
            dpi: 分辨率DPI (horizontal, vertical)
            keep_exif: 是否保留EXIF数据
            strip_metadata: 是否删除所有元数据
            verbose: 是否显示详细信息
        """
        self.quality = self._validate_quality(quality)
        self.optimize = optimize
        self.progressive = progressive
        self.subsampling = self._validate_subsampling(subsampling)
        self.dpi = dpi
        self.keep_exif = keep_exif
        self.strip_metadata = strip_metadata
        self.verbose = verbose
        
        # 保存参数供后续使用
        self._original_params = {
            'quality': quality,
            'optimize': optimize,
            'progressive': progressive,
            'subsampling': subsampling,
            'dpi': dpi,
        }
    
    @staticmethod
    def _validate_quality(quality: int) -> int:
        """验证质量参数"""
        if not 1 <= quality <= 100:
            raise ValueError(f"质量参数必须在1-100之间，当前值: {quality}")
        return quality
    
    @staticmethod
    def _validate_subsampling(subsampling: Optional[str]) -> Optional[str]:
        """验证色度子采样参数"""
        if subsampling is None:
            return None
        
        valid_subsamplings = {'4:4:4', '4:2:2', '4:2:0', '4:1:1'}
        if subsampling not in valid_subsamplings:
            raise ValueError(f"不支持的子采样格式: {subsampling}。支持的格式: {valid_subsamplings}")
        return subsampling
    
    def _get_subsampling_value(self) -> Optional[str]:
        """获取PIL所需的子采样值"""
        if self.subsampling == '4:4:4':
            return 'keep'
        elif self.subsampling == '4:2:2':
            return '2x2,1x1,1x1'
        elif self.subsampling == '4:2:0':
            return '2x2,1x1,1x1'
        elif self.subsampling == '4:1:1':
            return '2x2,1x1,1x1'
        return None
    
    def _calculate_target_size(self, current_size: int, target_ratio: float) -> int:
        """计算目标文件大小"""
        return int(current_size * target_ratio)
    
    def _binary_search_quality(
        self, 
        image: Image.Image, 
        target_size: int, 
        tolerance: float = 0.05,
        max_iterations: int = 20
    ) -> int:
        """
        二分搜索找到达到目标文件大小的最佳质量参数
        
        Args:
            image: PIL图像对象
            target_size: 目标文件大小（字节）
            tolerance: 允许的误差范围（百分比）
            max_iterations: 最大迭代次数
            
        Returns:
            最佳质量参数
        """
        low, high = 1, 100
        best_quality = self.quality
        
        for i in range(max_iterations):
            current_quality = (low + high) // 2
            
            # 保存到内存并计算大小
            import io
            buffer = io.BytesIO()
            
            save_kwargs = self._get_save_kwargs(current_quality)
            image.save(buffer, format='JPEG', **save_kwargs)
            current_size = buffer.tell()
            
            if self.verbose:
                print(f"  迭代 {i+1}: 质量={current_quality}, 大小={current_size:,}字节, "
                      f"目标={target_size:,}字节")
            
            if abs(current_size - target_size) / target_size <= tolerance:
                return current_quality
            
            if current_size > target_size:
                high = current_quality - 1
            else:
                low = current_quality + 1
                best_quality = current_quality
            
            if low > high:
                break
        
        return best_quality
    
    def _get_save_kwargs(self, quality: Optional[int] = None) -> dict:
        """获取保存参数"""
        kwargs = {
            'quality': quality if quality is not None else self.quality,
            'optimize': self.optimize,
            'progressive': self.progressive,
        }
        
        if self.subsampling:
            kwargs['subsampling'] = self._get_subsampling_value()
        
        if self.dpi:
            kwargs['dpi'] = self.dpi
        
        return kwargs
    
    def _preserve_exif(self, original_image: Image.Image, new_image: Image.Image) -> Image.Image:
        """保留EXIF数据"""
        if not self.keep_exif or self.strip_metadata:
            return new_image
        
        try:
            exif = original_image.getexif()
            if exif:
                new_image.info['exif'] = exif.tobytes()
        except Exception as e:
            if self.verbose:
                print(f"警告: 无法保留EXIF数据: {e}")
        
        return new_image
    
    def _resize_image(self, image: Image.Image, max_dimension: int, keep_aspect_ratio: bool = True) -> Image.Image:
        """调整图像大小"""
        original_width, original_height = image.size
        
        if keep_aspect_ratio:
            # 保持宽高比
            if original_width > original_height:
                new_width = max_dimension
                new_height = int(original_height * (max_dimension / original_width))
            else:
                new_height = max_dimension
                new_width = int(original_width * (max_dimension / original_height))
        else:
            # 直接缩放
            new_width = new_height = max_dimension
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def compress_image(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        mode: CompressionMode = CompressionMode.QUALITY,
        target_value: Optional[float] = None,
        overwrite: bool = False,
        suffix: str = "_compressed"
    ) -> CompressionResult:
        """
        压缩单个图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径（如果为None，则自动生成）
            mode: 压缩模式
            target_value: 目标值（质量百分比、文件大小比例等）
            overwrite: 是否覆盖已存在的文件
            suffix: 输出文件名后缀
            
        Returns:
            CompressionResult对象
        """
        import time
        start_time = time.time()
        
        input_path = Path(input_path)
        
        # 验证输入文件
        if not input_path.exists():
            return CompressionResult(
                input_path=str(input_path),
                output_path="",
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                quality_used=0,
                resolution_before=(0, 0),
                resolution_after=(0, 0),
                format="",
                compression_time=0,
                success=False,
                error_message=f"输入文件不存在: {input_path}"
            )
        
        # 生成输出路径
        if output_path is None:
            output_path = self._generate_output_path(input_path, suffix)
        else:
            output_path = Path(output_path)
        
        # 检查输出文件是否已存在
        if output_path.exists() and not overwrite:
            return CompressionResult(
                input_path=str(input_path),
                output_path=str(output_path),
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                quality_used=0,
                resolution_before=(0, 0),
                resolution_after=(0, 0),
                format="",
                compression_time=0,
                success=False,
                error_message=f"输出文件已存在: {output_path} (使用--overwrite覆盖)"
            )
        
        try:
            # 打开图像
            with Image.open(input_path) as img:
                # 转换为RGB模式（如果需要）
                if img.mode not in ('RGB', 'L'):
                    if img.mode == 'RGBA':
                        # 创建白色背景
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')
                
                original_size = input_path.stat().st_size
                resolution_before = img.size
                
                # 根据模式处理
                if mode == CompressionMode.RESOLUTION and target_value is not None:
                    # 调整分辨率
                    img = self._resize_image(img, int(target_value))
                    resolution_after = img.size
                    target_quality = self.quality
                elif mode == CompressionMode.FILESIZE and target_value is not None:
                    # 基于目标文件大小
                    target_size = self._calculate_target_size(original_size, target_value)
                    target_quality = self._binary_search_quality(img, target_size)
                    resolution_after = img.size
                elif mode == CompressionMode.AUTO:
                    # 自动模式：根据原始大小选择质量
                    if original_size > 10 * 1024 * 1024:  # > 10MB
                        target_quality = 60
                    elif original_size > 5 * 1024 * 1024:  # > 5MB
                        target_quality = 70
                    elif original_size > 1 * 1024 * 1024:  # > 1MB
                        target_quality = 80
                    else:
                        target_quality = 90
                    resolution_after = img.size
                else:
                    # 基于质量模式
                    if target_value is not None:
                        target_quality = self._validate_quality(int(target_value))
                    else:
                        target_quality = self.quality
                    resolution_after = img.size
                
                # 保存图像
                save_kwargs = self._get_save_kwargs(target_quality)
                
                # 处理元数据
                if not self.strip_metadata and self.keep_exif:
                    img = self._preserve_exif(Image.open(input_path), img)
                
                # 确保输出目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存文件
                img.save(output_path, **save_kwargs)
                
                # 获取压缩后大小
                compressed_size = output_path.stat().st_size
                compression_time = time.time() - start_time
                
                # 计算压缩比
                if original_size > 0:
                    compression_ratio = compressed_size / original_size
                else:
                    compression_ratio = 0
                
                if self.verbose:
                    self._print_compression_info(
                        input_path, output_path, original_size, 
                        compressed_size, target_quality, compression_time
                    )
                
                return CompressionResult(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    original_size=original_size,
                    compressed_size=compressed_size,
                    compression_ratio=compression_ratio,
                    quality_used=target_quality,
                    resolution_before=resolution_before,
                    resolution_after=resolution_after,
                    format="JPEG",
                    compression_time=compression_time,
                    success=True
                )
                
        except Exception as e:
            return CompressionResult(
                input_path=str(input_path),
                output_path=str(output_path) if output_path else "",
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                quality_used=0,
                resolution_before=(0, 0),
                resolution_after=(0, 0),
                format="",
                compression_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def compress_folder(
        self,
        input_folder: Union[str, Path],
        output_folder: Optional[Union[str, Path]] = None,
        mode: CompressionMode = CompressionMode.QUALITY,
        target_value: Optional[float] = None,
        recursive: bool = False,
        overwrite: bool = False,
        file_extensions: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ) -> List[CompressionResult]:
        """
        压缩文件夹中的所有图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            mode: 压缩模式
            target_value: 目标值
            recursive: 是否递归处理子文件夹
            overwrite: 是否覆盖已存在的文件
            file_extensions: 处理的文件扩展名列表
            max_files: 最大处理文件数
            
        Returns:
            CompressionResult对象列表
        """
        input_folder = Path(input_folder)
        
        if not input_folder.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        # 设置输出文件夹
        if output_folder is None:
            output_folder = input_folder.parent / f"{input_folder.name}_compressed"
        else:
            output_folder = Path(output_folder)
        
        # 获取所有图像文件
        image_files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for ext in file_extensions or self.SUPPORTED_FORMATS:
            image_files.extend(list(input_folder.glob(f"{pattern}{ext}")))
            image_files.extend(list(input_folder.glob(f"{pattern}{ext.upper()}")))
        
        # 去重并排序
        image_files = sorted(set(image_files), key=lambda x: x.name)
        
        # 限制文件数量
        if max_files is not None:
            image_files = image_files[:max_files]
        
        if self.verbose:
            print(f"找到 {len(image_files)} 个图像文件")
        
        results = []
        
        for i, input_file in enumerate(image_files, 1):
            if self.verbose:
                print(f"\n处理文件 {i}/{len(image_files)}: {input_file.name}")
            
            # 计算相对路径用于输出
            if recursive:
                rel_path = input_file.relative_to(input_folder)
                output_file = output_folder / rel_path
            else:
                output_file = output_folder / input_file.name
            
            # 压缩图像
            result = self.compress_image(
                input_path=input_file,
                output_path=output_file,
                mode=mode,
                target_value=target_value,
                overwrite=overwrite
            )
            
            results.append(result)
        
        return results
    
    def _generate_output_path(self, input_path: Path, suffix: str = "_compressed") -> Path:
        """生成输出文件路径"""
        stem = input_path.stem
        parent = input_path.parent
        return parent / f"{stem}{suffix}.jpg"
    
    def _print_compression_info(
        self, 
        input_path: Path, 
        output_path: Path, 
        original_size: int, 
        compressed_size: int,
        quality: int,
        compression_time: float
    ):
        """打印压缩信息"""
        size_reduction = original_size - compressed_size
        reduction_percent = (size_reduction / original_size * 100) if original_size > 0 else 0
        
        print(f"输入: {input_path.name}")
        print(f"输出: {output_path.name}")
        print(f"原始大小: {self._format_size(original_size)}")
        print(f"压缩后大小: {self._format_size(compressed_size)}")
        print(f"减小: {self._format_size(size_reduction)} ({reduction_percent:.1f}%)")
        print(f"质量: {quality}")
        print(f"时间: {compression_time:.2f}秒")
        print("-" * 50)
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def get_compression_stats(self, results: List[CompressionResult]) -> dict:
        """获取压缩统计信息"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if not successful:
            return {
                'total_files': len(results),
                'successful': 0,
                'failed': len(failed),
                'total_original_size': 0,
                'total_compressed_size': 0,
                'average_compression_ratio': 0,
                'total_size_reduction': 0,
                'total_time': sum(r.compression_time for r in results)
            }
        
        total_original = sum(r.original_size for r in successful)
        total_compressed = sum(r.compressed_size for r in successful)
        total_reduction = total_original - total_compressed
        avg_ratio = sum(r.compression_ratio for r in successful) / len(successful)
        
        return {
            'total_files': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_original_size': total_original,
            'total_compressed_size': total_compressed,
            'average_compression_ratio': avg_ratio,
            'total_size_reduction': total_reduction,
            'total_reduction_percent': (total_reduction / total_original * 100) if total_original > 0 else 0,
            'total_time': sum(r.compression_time for r in results),
            'average_time': sum(r.compression_time for r in successful) / len(successful) if successful else 0
        }


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        description="JPEG图像压缩工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 压缩单个图像，质量75
  %(prog)s image.jpg --quality 75
  
  # 压缩文件夹，目标为原大小的50%%
  %(prog)s images/ --mode filesize --target 0.5
  
  # 压缩并调整最大分辨率为1024像素
  %(prog)s image.jpg --mode resolution --target 1024
  
  # 批量压缩，输出详细日志
  %(prog)s photos/ --output compressed/ --verbose --recursive
  
  # 自动模式（根据文件大小智能选择质量）
  %(prog)s image.jpg --mode auto
        """
    )
    
    # 输入输出参数
    parser.add_argument('input', help='输入文件或文件夹路径')
    parser.add_argument('-o', '--output', help='输出文件或文件夹路径')
    
    # 压缩参数
    parser.add_argument('-q', '--quality', type=int, default=85, 
                       help='JPEG质量 (1-100，默认: 85)')
    parser.add_argument('-m', '--mode', choices=['quality', 'filesize', 'resolution', 'auto'], 
                       default='quality', help='压缩模式 (默认: quality)')
    parser.add_argument('-t', '--target', type=float, 
                       help='目标值（质量百分比/文件大小比例/最大分辨率）')
    
    # 高级参数
    parser.add_argument('--no-optimize', action='store_true', 
                       help='禁用编码优化')
    parser.add_argument('--progressive', action='store_true', 
                       help='使用渐进式JPEG')
    parser.add_argument('--subsampling', choices=['4:4:4', '4:2:2', '4:2:0', '4:1:1'],
                       help='色度子采样模式')
    parser.add_argument('--dpi', type=int, nargs=2, metavar=('X', 'Y'),
                       help='设置DPI分辨率')
    parser.add_argument('--strip-metadata', action='store_true',
                       help='删除所有元数据（EXIF等）')
    parser.add_argument('--no-exif', action='store_true',
                       help='不保留EXIF数据')
    
    # 批量处理参数
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归处理子文件夹')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的文件')
    parser.add_argument('--suffix', default='_compressed',
                       help='输出文件名后缀（默认: _compressed）')
    parser.add_argument('--max-files', type=int,
                       help='最大处理文件数')
    parser.add_argument('--extensions', nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'],
                       help='处理的文件扩展名（默认: 常见图像格式）')
    
    # 输出参数
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式，只显示错误')
    parser.add_argument('--stats', action='store_true',
                       help='显示统计信息')
    parser.add_argument('--json', action='store_true',
                       help='以JSON格式输出结果')
    parser.add_argument('--log', help='日志文件路径')
    
    args = parser.parse_args()
    
    # 配置日志
    if args.log:
        import logging
        logging.basicConfig(
            filename=args.log,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    # 创建压缩器
    try:
        compressor = JPEGCompressor(
            quality=args.quality,
            optimize=not args.no_optimize,
            progressive=args.progressive,
            subsampling=args.subsampling,
            dpi=tuple(args.dpi) if args.dpi else None,
            keep_exif=not args.no_exif,
            strip_metadata=args.strip_metadata,
            verbose=args.verbose
        )
    except ValueError as e:
        print(f"参数错误: {e}")
        sys.exit(1)
    
    # 确定模式
    mode_map = {
        'quality': CompressionMode.QUALITY,
        'filesize': CompressionMode.FILESIZE,
        'resolution': CompressionMode.RESOLUTION,
        'auto': CompressionMode.AUTO
    }
    mode = mode_map[args.mode]
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)
    
    results = []
    
    try:
        if input_path.is_file():
            # 压缩单个文件
            result = compressor.compress_image(
                input_path=input_path,
                output_path=args.output,
                mode=mode,
                target_value=args.target,
                overwrite=args.overwrite,
                suffix=args.suffix
            )
            results.append(result)
            
            if not result.success:
                print(f"错误: {result.error_message}")
                sys.exit(1)
                
        elif input_path.is_dir():
            # 压缩文件夹
            results = compressor.compress_folder(
                input_folder=input_path,
                output_folder=args.output,
                mode=mode,
                target_value=args.target,
                recursive=args.recursive,
                overwrite=args.overwrite,
                file_extensions=args.extensions,
                max_files=args.max_files
            )
            
            # 检查是否有失败的文件
            failed = [r for r in results if not r.success]
            if failed and not args.quiet:
                print(f"\n警告: {len(failed)} 个文件处理失败")
                for r in failed[:5]:  # 只显示前5个失败的文件
                    print(f"  - {r.input_path}: {r.error_message}")
                if len(failed) > 5:
                    print(f"  ... 还有 {len(failed) - 5} 个失败文件")
        else:
            print(f"错误: 输入路径不是文件或文件夹: {input_path}")
            sys.exit(1)
        
        # 输出结果
        if args.json:
            import json
            json_output = {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'mode': args.mode,
                    'quality': args.quality,
                    'target_value': args.target,
                },
                'results': [asdict(r) for r in results],
                'statistics': compressor.get_compression_stats(results)
            }
            print(json.dumps(json_output, indent=2, default=str))
        
        elif args.stats:
            stats = compressor.get_compression_stats(results)
            
            print("\n" + "="*60)
            print("压缩统计")
            print("="*60)
            print(f"总文件数: {stats['total_files']}")
            print(f"成功: {stats['successful']}")
            print(f"失败: {stats['failed']}")
            print(f"总原始大小: {compressor._format_size(stats['total_original_size'])}")
            print(f"总压缩大小: {compressor._format_size(stats['total_compressed_size'])}")
            print(f"总减小: {compressor._format_size(stats['total_size_reduction'])} "
                  f"({stats['total_reduction_percent']:.1f}%)")
            print(f"平均压缩比: {stats['average_compression_ratio']:.3f}")
            print(f"总时间: {stats['total_time']:.2f}秒")
            print(f"平均时间: {stats['average_time']:.2f}秒/文件")
            
            if input_path.is_file() and results[0].success:
                result = results[0]
                print(f"\n单个文件详情:")
                print(f"  输入: {result.input_path}")
                print(f"  输出: {result.output_path}")
                print(f"  分辨率: {result.resolution_before[0]}x{result.resolution_before[1]} → "
                      f"{result.resolution_after[0]}x{result.resolution_after[1]}")
                print(f"  质量: {result.quality_used}")
        
        elif not args.quiet:
            if input_path.is_file():
                result = results[0]
                if result.success:
                    reduction = (result.original_size - result.compressed_size) / result.original_size * 100
                    print(f"\n压缩完成!")
                    print(f"  原始: {compressor._format_size(result.original_size)}")
                    print(f"  压缩后: {compressor._format_size(result.compressed_size)}")
                    print(f"  减小: {reduction:.1f}%")
                    print(f"  保存为: {result.output_path}")
            else:
                successful = [r for r in results if r.success]
                if successful:
                    print(f"\n批量压缩完成!")
                    print(f"  处理了 {len(results)} 个文件")
                    print(f"  成功: {len(successful)} 个")
                    if args.failed:
                        print(f"  失败: {len(failed)} 个")
    
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()