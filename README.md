# 混合专家图像压缩系统

## 🚀 快速启动

### 环境配置
```bash
# Windows
setup\setup_env.bat

# Linux/Mac
chmod +x setup/setup_env.sh
./setup/setup_env.sh
```

### 数据准备
```bash
python utils/download_dataset.py
```

### 训练模型
```bash
# 训练传统卷积模型
python train/train_model.py

# 训练MoE卷积模型
python train/train_moe_model.py
```

### 模型评估
```bash
python src/eval.py
```

## 📁 模块说明

### 数据集模块 (`dataset/`)
- **功能**: 包含训练和测试数据集
- **配置**: 使用COCO 2017数据集，划分为训练集和零样本测试集
- **特点**: 支持严格的比特率控制，筛选样本比特率在[μ±σ]范围内

### 模型架构 (`model/`)
- **conv2d/**: 传统卷积自编码器 (Conv基准)
- **moe_conv2d/**: 提出的卷积MoE模型 (Conv backbone + MoE)
- **vit/**: Vision Transformer自编码器 (ViT基准)
- **moe_vit/**: 提出的ViT MoE模型 (ViT backbone + MoE)
- **参数范围**: 所有模型控制在1M-10M参数，确保公平比较

### 训练模块 (`train/`)
- **train_model.py**: 训练传统模型
- **train_moe_model.py**: 训练混合专家模型
- **特点**: 支持梯度传播的压缩率损失计算

### 工具模块 (`utils/`)
- **download_dataset.py**: 数据集下载工具
- **jpeg.py**: JPEG压缩基准实现
- **环境配置**: 包含跨平台的虚拟环境设置脚本

### 评估模块 (`src/eval/`)
- **eval.py**: 统一的模型评估系统
- **评估指标**: 
  - PSNR (峰值信噪比)
  - SSIM (结构相似性指数)
  - 在YCbCr的Y通道上进行评估

## 🔬 实验设置

### 对比方法
1. **JPEG**: 传统编解码器基准
2. **Conv**: 卷积自编码器 (无MoE)
3. **Conv-MoE**: 提出的方法 (Conv骨干 + MoE)
4. **ViT**: Vision Transformer自编码器 (无MoE)
5. **ViT-MoE**: 提出的方法 (ViT骨干 + MoE)

### 受控评估策略
- **比特率控制**: 筛选比特率在[μ±σ]范围内的样本
- **架构有效性隔离**: 排除率分配差异的影响
- **公平比较**: 所有模型参数控制在1M-10M之间

### 实验特性
- **模块化设计**: 易于扩展新的模型架构
- **可复现性**: 包含完整的环境配置脚本
- **灵活训练**: 支持传统模型和MoE模型的独立训练
- **全面评估**: 提供多指标的性能评估系统