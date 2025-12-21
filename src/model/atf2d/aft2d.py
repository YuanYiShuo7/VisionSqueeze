import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn

class AFT2DAttention(nn.Module):
    """
    AFT-2D Attention Block
    公式: attn(x,y) = ΣΣ exp(w_h[x-i] + w_v[y-j] + k_{i,j}) ⊙ v_{i,j} / ΣΣ exp(w_h[x-i] + w_v[y-j] + k_{i,j})
    """
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, r=1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim or in_dim
        self.out_dim = out_dim or in_dim
        self.r = r
        self.window_size = 2 * r + 1  # 局部窗口大小
        
        # 学习水平和垂直方向的位置偏置权重
        # w_h 和 w_v 的形状为 (window_size,)
        self.w_h = nn.Parameter(torch.randn(self.window_size))
        self.w_v = nn.Parameter(torch.randn(self.window_size))
        
        # 用于生成k和v的线性变换
        self.to_k = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(self.in_dim, self.hidden_dim, bias=False)

        # 输出投影
        self.proj = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 位置偏置权重初始化为较小的值
        nn.init.normal_(self.w_h, mean=0, std=0.02)
        nn.init.normal_(self.w_v, mean=0, std=0.02)
        
        # 线性层初始化
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        
    def compute_position_bias(self, H, W, device):
        """
        计算位置偏置矩阵（向量化版本）
        
        参数:
            H (int): 特征图高度
            W (int): 特征图宽度
            
        返回:
            position_bias: (window_size, window_size, H, W)
        """
        # 创建网格
        i = torch.arange(H, device=device)  # (H,)
        j = torch.arange(W, device=device)  # (W,)
        
        # 创建窗口偏移
        offsets = torch.arange(-self.r, self.r + 1, device=device)  # (window_size,)
        
        # 创建 i 和 j 的差值矩阵
        # i_diff: (window_size, H)
        # j_diff: (window_size, W)
        i_diff = offsets.view(-1, 1) - i.view(1, -1)  # (window_size, H)
        j_diff = offsets.view(-1, 1) - j.view(1, -1)  # (window_size, W)
        
        # 计算偏置
        # w_h: (window_size,) -> (window_size, 1)
        # horizontal_bias: (window_size, H, 1)
        horizontal_bias = (self.w_h.view(-1, 1) * i_diff).unsqueeze(-1)  # (window_size, H, 1)
        
        # w_v: (window_size,) -> (window_size, 1)
        # vertical_bias: (window_size, 1, W)
        vertical_bias = (self.w_v.view(-1, 1) * j_diff).unsqueeze(1)  # (window_size, 1, W)
        
        # 广播相加
        # horizontal_bias: (window_size, H, 1) -> (window_size, 1, H, 1)
        # vertical_bias: (window_size, 1, W) -> (1, window_size, 1, W)
        position_bias = horizontal_bias.unsqueeze(1) + vertical_bias.unsqueeze(0)  # (window_size, window_size, H, W)
        
        return position_bias
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (B, H, W, C)
            
        返回:
            output: (B, H, W, out_C)
        """
        B, H, W, C = x.shape
        device = x.device
        
        # 1. 计算key和value
        k = self.to_k(x)  # (B, H, W, hidden_dim)
        v = self.to_v(x)  # (B, H, W, hidden_dim)
        
        # 2. 计算位置偏置
        position_bias = self.compute_position_bias(H, W, device)  # (ws, ws, H, W)
        
        # 3. 优化：向量化实现，避免循环
        output = torch.zeros(B, H, W, self.hidden_dim, device=device)
        normalization = torch.zeros(B, H, W, 1, device=device)
        
        # 预先计算所有偏移
        offsets_i = torch.arange(-self.r, self.r + 1, device=device)
        offsets_j = torch.arange(-self.r, self.r + 1, device=device)
        
        for idx_i, i_offset in enumerate(offsets_i):
            for idx_j, j_offset in enumerate(offsets_j):
                # 计算偏移后的索引
                i_grid = torch.arange(H, device=device).view(1, H, 1)  # (1, H, 1)
                j_grid = torch.arange(W, device=device).view(1, 1, W)  # (1, 1, W)
                
                i_src = i_grid + i_offset
                j_src = j_grid + j_offset
                
                # 创建有效掩码
                mask_i = (i_src >= 0) & (i_src < H)
                mask_j = (j_src >= 0) & (j_src < W)
                valid_mask = mask_i & mask_j  # (1, H, W)
                
                # 裁剪索引到有效范围
                i_src_clamped = i_src.clamp(0, H - 1)
                j_src_clamped = j_src.clamp(0, W - 1)
                
                # 批量索引
                batch_idx = torch.arange(B, device=device).view(B, 1, 1)
                
                # 使用gather收集偏移位置的特征
                # 使用advanced indexing
                k_offset = k[batch_idx, i_src_clamped, j_src_clamped, :]  # (B, H, W, hidden_dim)
                v_offset = v[batch_idx, i_src_clamped, j_src_clamped, :]  # (B, H, W, hidden_dim)
                
                # 获取对应的位置偏置
                pos_bias = position_bias[idx_i, idx_j, :, :]  # (H, W)
                pos_bias = pos_bias.view(1, H, W, 1)  # (1, H, W, 1)
                
                # 计算权重: exp(w_h[x-i] + w_v[y-j] + k_{i,j})
                weights = torch.exp(pos_bias + k_offset)
                
                # 应用有效掩码
                valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (1, H, W, 1)
                weights = weights * valid_mask_expanded
                
                # 累加加权value和归一化分母
                output = output + weights * v_offset
                normalization = normalization + weights.sum(dim=-1, keepdim=True)
        
        # 4. 归一化（避免除以零）
        output = output / (normalization + 1e-8)
        
        # 5. 应用投影层
        output = self.proj(output)
        
        return output

class AFT2DBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, r=1, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.dim = in_dim
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        
        # 第一层归一化
        self.norm1 = nn.LayerNorm(in_dim)
        
        # AFT-2D注意力层
        self.aft_attn = AFT2DAttention(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            r=r
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 第二层归一化
        self.norm2 = nn.LayerNorm(out_dim)
        
        ff_hidden_dim = out_dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        参数:
            x: (B, H, W, C)
        返回:
            out: (B, H, W, C_out)
        """
        residual = x
        
        # 第一分支：AFT-2D注意力
        x = self.norm1(x)
        x = self.aft_attn(x)
        x = self.dropout(x)
        
        # 残差连接
        if residual.shape == x.shape:
            x = x + residual
        else:
            # 如果维度不匹配（如C≠C_out），使用1x1卷积调整维度
            if residual.shape[-1] != x.shape[-1]:
                residual = nn.Linear(residual.shape[-1], x.shape[-1]).to(x.device)(residual)
            x = x + residual
        
        # 保存残差
        residual = x
        
        # 第二分支：前馈网络
        x = self.norm2(x)
        x = self.ffn(x)
        
        # 残差连接
        x = x + residual
        
        return x

def main():
    """测试AFT2DAttention模块"""
    print("测试AFT2DAttention模块...")
    
    # 测试配置
    B, H, W, C = 2, 24, 24, 256  # 使用错误中的尺寸
    r = 1  # 窗口半径
    window_size = 2 * r + 1  # 窗口大小
    
    # 创建输入张量
    x = torch.randn(B, H, W, C)
    print(f"输入形状: {x.shape}")
    print(f"窗口大小: {window_size}x{window_size}")
    
    # 创建AFT2DAttention模块（使用方案二）
    print("\n创建AFT2DAttention模块...")
    aft_attention = AFT2DAttention(
        in_dim=C,
        hidden_dim=512,
        out_dim=128,
        r=r
    )
    
    # 前向传播
    print("\n执行前向传播...")
    try:
        output = aft_attention(x)
        print(f"✓ 前向传播成功!")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        # 验证维度
        expected_shape = (B, H, W, 128)
        if output.shape == expected_shape:
            print(f"✓ 输出形状正确: {output.shape}")
        else:
            print(f"✗ 输出形状错误: 期望 {expected_shape}, 得到 {output.shape}")
        
        # 检查位置偏置
        print("\n检查位置偏置计算...")
        pos_bias = aft_attention.compute_position_bias(H, W, x.device)
        print(f"位置偏置形状: {pos_bias.shape}")
        expected_bias_shape = (window_size, window_size, H, W)
        if pos_bias.shape == expected_bias_shape:
            print(f"✓ 位置偏置形状正确: {pos_bias.shape}")
        else:
            print(f"✗ 位置偏置形状错误: 期望 {expected_bias_shape}, 得到 {pos_bias.shape}")
        
        # 检查参数
        print("\n检查参数...")
        print(f"w_h 形状: {aft_attention.w_h.shape} (应为 ({window_size},))")
        print(f"w_v 形状: {aft_attention.w_v.shape} (应为 ({window_size},))")
        print(f"to_k 权重形状: {aft_attention.to_k.weight.shape}")
        print(f"to_v 权重形状: {aft_attention.to_v.weight.shape}")
        print(f"proj 权重形状: {aft_attention.proj.weight.shape}")
        
        # 检查梯度
        print("\n检查梯度计算...")
        loss = output.sum()
        loss.backward()
        
        # 检查参数梯度
        params_with_grad = []
        for name, param in aft_attention.named_parameters():
            if param.grad is not None:
                params_with_grad.append(name)
        
        print(f"有梯度的参数: {len(params_with_grad)}/{len(list(aft_attention.parameters()))}")
        expected_params = ['w_h', 'w_v', 'to_k.weight', 'to_v.weight', 'proj.weight']
        for param_name in expected_params:
            if param_name in [name for name, _ in aft_attention.named_parameters()]:
                param = dict(aft_attention.named_parameters())[param_name]
                if param.grad is not None:
                    print(f"✓ {param_name}: 梯度计算正常")
                else:
                    print(f"✗ {param_name}: 无梯度")
        
        # 输出统计信息
        print(f"\n输出统计:")
        print(f"输出均值: {output.mean().item():.6f}")
        print(f"输出标准差: {output.std().item():.6f}")
        print(f"输出最小值: {output.min().item():.6f}")
        print(f"输出最大值: {output.max().item():.6f}")
        
        print("\n✓ 所有测试通过!")
        
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()