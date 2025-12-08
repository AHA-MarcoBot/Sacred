"""
Flow Sequence Transformer for Traffic Obfuscation

单层轻量级 Transformer 架构，只处理包长度序列
优化版本：删除时间特征，只保留一层 Window Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for flow sequences"""
    
    def __init__(self, num_flows, seq_len, d_model):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(num_flows, seq_len, d_model) * 0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_flows, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pos_encoding.unsqueeze(0)


class WindowAttention(nn.Module):
    """Local window attention for efficient computation"""
    
    def __init__(self, d_model, num_heads, window_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_flows, seq_len, d_model)
        Returns:
            attention output: same shape as x
        """
        B, F, L, D = x.shape
        
        # Divide sequence into windows
        num_windows = L // self.window_size
        pad_len = 0
        if L % self.window_size != 0:
            # Pad if needed
            pad_len = self.window_size - (L % self.window_size)
            x = nnF.pad(x, (0, 0, 0, pad_len))
            L = x.shape[2]
            num_windows = L // self.window_size
        
        # Reshape to (batch, num_flows, num_windows, window_size, d_model)
        x = x.view(B, F, num_windows, self.window_size, D)
        
        # Reshape to (batch * num_flows * num_windows, window_size, d_model)
        x = x.reshape(B * F * num_windows, self.window_size, D)
        
        # QKV projection
        qkv = self.qkv(x).reshape(B * F * num_windows, self.window_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch*flows*windows, heads, win_size, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = nnF.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = attn_weights @ v  # (batch*flows*windows, heads, win_size, head_dim)
        out = out.transpose(1, 2).reshape(B * F * num_windows, self.window_size, D)
        
        # Project
        out = self.proj(out)
        out = self.dropout(out)
        
        # Reshape back to (batch, num_flows, seq_len, d_model)
        out = out.view(B, F, num_windows, self.window_size, D)
        out = out.view(B, F, num_windows * self.window_size, D)
        
        # Remove padding if added
        if pad_len > 0:
            out = out[:, :, :-pad_len, :]
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(nnF.gelu(self.fc1(x))))


class FlowSequenceTransformer(nn.Module):
    """
    单层 Transformer 用于流序列状态提取（优化版）
    
    架构：
    - Conv1D 嵌入（只处理包长度，不处理时间）
    - Window Attention
    - 全局池化 + 输出投影
    """
    
    def __init__(
        self,
        max_flows=50,
        seq_len=1000,
        output_dim=128,
        dropout=0.1,
        device='cuda',
        mtu=1500.0
    ):
        super().__init__()
        
        self.max_flows = max_flows
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.device = device
        self.mtu = mtu  # Maximum Transmission Unit，用于归一化包长度
        
        # ========== 嵌入层（只处理包长度）==========
        d_model = 64
        self.d_model = d_model
        self.length_embed = nn.Conv1d(1, d_model, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(max_flows, seq_len // 2, d_model)
        
        # ========== Window Attention ==========
        self.norm1 = nn.LayerNorm(d_model)
        self.window_attn = WindowAttention(d_model, num_heads=4, window_size=50, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 4, dropout=dropout)
        
        # ========== 输出层 ==========
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def extract_flow_features(self, flow_data):
        """
        从流数据中只提取 packet_length（优化版）
        
        Args:
            flow_data: dict with 'packet_length' 
                       or (1000,) or (1000, 1) tensor [length only]
        Returns:
            (1000, 1) tensor
        """
        if isinstance(flow_data, dict):
            length = torch.tensor(flow_data['packet_length'], dtype=torch.float32)
            if length.dim() == 1:
                return length.unsqueeze(-1)  # (1000, 1)
            return length
        return flow_data
    
    def forward(self, flow_sequence, num_flows):
        """
        前向传播（优化版：只处理包长度）
        
        Args:
            flow_sequence: (batch, max_flows, seq_len, 1) 包长度序列
            num_flows: (batch,) 每个样本的有效流数量
        
        Returns:
            state: (batch, output_dim) 状态向量
        """
        B = flow_sequence.shape[0]
        
        # 确保输入在正确的设备上
        flow_sequence = flow_sequence.to(self.device)
        if isinstance(num_flows, torch.Tensor):
            num_flows = num_flows.to(self.device)
        else:
            num_flows = torch.tensor(num_flows, device=self.device)
        
        # Create flow mask
        flow_mask = torch.arange(self.max_flows, device=self.device)[None, :] < num_flows[:, None]
        
        # 提取包长度序列（只有一个通道）
        length_seq = flow_sequence[:, :, :, 0]  # (B, max_flows, seq_len)
        
        # 归一化包长度（与 graph_builder.py 保持一致）
        length_seq = length_seq / self.mtu  # 归一化到 [0, 1] 范围
        
        # ========== 嵌入层（只处理包长度）==========
        # Reshape for Conv1D: (B*max_flows, 1, seq_len)
        length_seq = length_seq.reshape(B * self.max_flows, 1, self.seq_len)
        
        # Conv1D embedding
        emb = self.length_embed(length_seq)  # (B*max_flows, 64, seq_len//2)
        emb = self.bn(emb)
        emb = self.relu(emb)
        
        # Reshape: (B, max_flows, seq_len//2, 64)
        seq_len_half = emb.shape[-1]
        emb = emb.permute(0, 2, 1)  # (B*max_flows, seq_len//2, 64)
        emb = emb.reshape(B, self.max_flows, seq_len_half, self.d_model)
        
        # 位置编码
        x = self.pos_encoding(emb)
        
        # ========== Window Attention ==========
        x = x + self.window_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # x: (B, max_flows, 500, 64)
        
        # ========== 全局池化（考虑 mask）==========
        # 先对序列维度池化: (B, max_flows, 64)
        x = x.permute(0, 3, 1, 2)  # (B, 64, max_flows, 500)
        x = nnF.adaptive_avg_pool2d(x, (self.max_flows, 1))  # (B, 64, max_flows, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, max_flows, 64)
        
        # Masked average pooling across flows
        x_masked = x * flow_mask.unsqueeze(-1).float()
        flow_counts = flow_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # 避免除以 0
        pooled = x_masked.sum(dim=1) / flow_counts  # (B, 64)
        
        # ========== 输出投影 ==========
        output = self.output_proj(pooled)  # (B, output_dim)
        
        return output


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Optimized FlowSequenceTransformer (Single Layer)")
    print("=" * 80)
    
    # 初始化模型
    model = FlowSequenceTransformer(
        max_flows=50,
        seq_len=1000,
        output_dim=128,
        device='cpu'  # 测试使用 CPU
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    
    # 创建测试数据（模拟真实的包长度：0-1500 字节）
    batch_size = 32
    flow_sequences = torch.randint(0, 1501, (batch_size, 50, 1000, 1), dtype=torch.float32)  # 包长度范围 0-1500
    num_flows = torch.randint(1, 51, (batch_size,))
    
    print(f"\n输入形状: {flow_sequences.shape}")
    print(f"包长度范围: [{flow_sequences.min().item():.0f}, {flow_sequences.max().item():.0f}] bytes")
    print(f"num_flows: {num_flows[:5].tolist()}")
    print(f"MTU: {model.mtu}")
    
    # 前向传播
    output = model(flow_sequences, num_flows)
    
    print(f"\n输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print("\n" + "=" * 80)
    print("✓ 测试通过！")
    print("=" * 80)

