"""
Flow Sequence Replay Buffer

存储原始流序列数据的经验回放缓冲区
"""

import random
from collections import deque
import numpy as np
import torch


class FlowSequenceExperience:
    """单条经验记录"""
    
    __slots__ = ['flow_sequence', 'num_flows', 'action', 'reward', 
                 'next_flow_sequence', 'next_num_flows', 'done']
    
    def __init__(self, flow_sequence, num_flows, action, reward, 
                 next_flow_sequence, next_num_flows, done):
        self.flow_sequence = flow_sequence  # (50, 1000, 2)
        self.num_flows = num_flows
        self.action = action
        self.reward = reward
        self.next_flow_sequence = next_flow_sequence  # (50, 1000, 2)
        self.next_num_flows = next_num_flows
        self.done = done


class FlowSequenceReplayBuffer:
    """
    经验回放缓冲区，存储原始流序列数据
    
    与原来的 ReplayBuffer 的区别：
    - 存储流序列矩阵而不是编码后的状态向量
    - 存储 num_flows 用于 masking
    - 更大的内存占用（但可训练 Transformer）
    """
    
    def __init__(self, buffer_size, batch_size):
        """
        Args:
            buffer_size: 缓冲区最大容量
            batch_size: 采样批次大小
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, flow_sequence, num_flows, action, reward, 
            next_flow_sequence, next_num_flows, done):
        """
        添加经验到缓冲区
        
        Args:
            flow_sequence: (50, 1000, 2) numpy array
            num_flows: int, 有效流数量
            action: numpy array, 动作向量
            reward: float
            next_flow_sequence: (50, 1000, 2) numpy array
            next_num_flows: int
            done: float (0/1)
        """
        exp = FlowSequenceExperience(
            flow_sequence=flow_sequence,
            num_flows=num_flows,
            action=action,
            reward=reward,
            next_flow_sequence=next_flow_sequence,
            next_num_flows=next_num_flows,
            done=done
        )
        self.memory.append(exp)
    
    def sample(self):
        """
        从缓冲区随机采样一个批次
        
        Returns:
            (flow_sequences, num_flows, actions, rewards, 
             next_flow_sequences, next_num_flows, dones)
            每个都是 torch.Tensor
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Stack into batches
        flow_sequences = np.stack([e.flow_sequence for e in experiences])
        num_flows = np.array([e.num_flows for e in experiences], dtype=np.int64)
        actions = np.stack([e.action for e in experiences])
        rewards = np.array([[e.reward] for e in experiences], dtype=np.float32)
        next_flow_sequences = np.stack([e.next_flow_sequence for e in experiences])
        next_num_flows = np.array([e.next_num_flows for e in experiences], dtype=np.int64)
        dones = np.array([[e.done] for e in experiences], dtype=np.float32)
        
        # Convert to tensors
        flow_sequences = torch.from_numpy(flow_sequences).float()
        num_flows = torch.from_numpy(num_flows).long()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        next_flow_sequences = torch.from_numpy(next_flow_sequences).float()
        next_num_flows = torch.from_numpy(next_num_flows).long()
        dones = torch.from_numpy(dones).float()
        
        return (flow_sequences, num_flows, actions, rewards,
                next_flow_sequences, next_num_flows, dones)
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.memory)
    
    def get_memory_usage(self):
        """
        估算内存占用
        
        Returns:
            dict with memory statistics
        """
        if len(self.memory) == 0:
            return {"total_mb": 0, "per_experience_kb": 0, "num_experiences": 0}
        
        # 估算单条经验的内存
        # flow_sequence: 50 * 1000 * 2 * 4 bytes = 400 KB
        # next_flow_sequence: 50 * 1000 * 2 * 4 bytes = 400 KB
        # other fields: ~1 KB
        bytes_per_experience = 50 * 1000 * 2 * 4 * 2 + 1024
        total_bytes = bytes_per_experience * len(self.memory)
        
        return {
            "total_mb": total_bytes / (1024 * 1024),
            "per_experience_kb": bytes_per_experience / 1024,
            "num_experiences": len(self.memory)
        }


if __name__ == "__main__":
    # 测试经验回放缓冲区
    print("=" * 80)
    print("FlowSequenceReplayBuffer 测试")
    print("=" * 80)
    
    # 创建缓冲区
    buffer = FlowSequenceReplayBuffer(buffer_size=1000, batch_size=32)
    
    # 添加一些经验
    print("\n添加经验...")
    for i in range(100):
        flow_seq = np.random.randn(50, 1000, 2).astype(np.float32)
        num_flows = np.random.randint(1, 51)
        action = np.random.randn(10).astype(np.float32)
        reward = np.random.randn()
        next_flow_seq = np.random.randn(50, 1000, 2).astype(np.float32)
        next_num_flows = np.random.randint(1, 51)
        done = 0.0 if i < 99 else 1.0
        
        buffer.add(flow_seq, num_flows, action, reward, 
                   next_flow_seq, next_num_flows, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 查看内存占用
    mem_stats = buffer.get_memory_usage()
    print(f"\n内存统计:")
    print(f"  总内存: {mem_stats['total_mb']:.2f} MB")
    print(f"  单条经验: {mem_stats['per_experience_kb']:.2f} KB")
    print(f"  经验数量: {mem_stats['num_experiences']}")
    
    # 采样测试
    print("\n采样测试...")
    (flow_seqs, num_flows, actions, rewards,
     next_flow_seqs, next_num_flows, dones) = buffer.sample()
    
    print(f"  flow_sequences: {flow_seqs.shape}, dtype={flow_seqs.dtype}")
    print(f"  num_flows: {num_flows.shape}, dtype={num_flows.dtype}")
    print(f"  actions: {actions.shape}, dtype={actions.dtype}")
    print(f"  rewards: {rewards.shape}, dtype={rewards.dtype}")
    print(f"  next_flow_sequences: {next_flow_seqs.shape}, dtype={next_flow_seqs.dtype}")
    print(f"  next_num_flows: {next_num_flows.shape}, dtype={next_num_flows.dtype}")
    print(f"  dones: {dones.shape}, dtype={dones.dtype}")
    
    print("\n采样值示例:")
    print(f"  num_flows[0:5]: {num_flows[0:5].tolist()}")
    print(f"  rewards[0:5]: {rewards[0:5].squeeze().tolist()}")
    
    # 估算1000条经验的内存
    print("\n" + "=" * 80)
    print("内存占用估算")
    print("=" * 80)
    sizes = [100, 500, 1000, 2000, 5000]
    for size in sizes:
        mem_mb = (50 * 1000 * 2 * 4 * 2 + 1024) * size / (1024 * 1024)
        print(f"  {size:5d} 条经验: {mem_mb:7.2f} MB")
    
    print("\n✓ 测试通过")
    print("=" * 80)

