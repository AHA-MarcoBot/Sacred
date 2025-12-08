"""
基于 Transformer 的流量混淆器

关键改变：
- 使用 Flow Sequence Transformer 替代 Graph Encoder 进行状态提取
- 经验回放存储原始流序列数据而不是编码后的状态
- Graph Builder 仅用于最后的分类评估（FG-net）
- 训练流程：三阶段（Transformer 处理 → Graph Builder 分类 → 填充 reward）
"""

import copy
import logging
import os
import random
from typing import Dict, List, Optional

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flow_transformer import FlowSequenceTransformer
from flow_replay_buffer import FlowSequenceReplayBuffer
from graph_builder import IncrementalGraphBuilder
from traffic_agent import ActionSpace, GaussianPolicy, QNetwork


def hard_update(local_model, target_model):
    """硬更新：初始同步目标网络参数"""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def soft_update(local_model, target_model, tau):
    """软更新：逐步同步目标网络参数"""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class FlowTrafficObfuscator:
    """
    基于 Transformer 的流量混淆器
    
    与原 GraphTrafficObfuscator 的关键区别：
    1. 状态 = Transformer 对流序列的编码（可训练）
    2. 经验回放存储原始流序列
    3. Graph Builder 仅用于分类评估
    4. 训练流程：三阶段（处理 → 分类 → reward填充）
    """

    def __init__(
        self,
        max_flows=50,
        max_dummy_packets=10,
        num_classes=1,
        device='cuda',
        reward_c=None,
        hidden_dim=1024,
        buffer_size=int(1e6),
        batch_size=64,
        gamma=0.99,
        tau=5e-3,
        lr=3e-4,
        alpha=0.2,
        target_update_interval=1,
        reward_type='o',
        transformer_output_dim=256,
        incremental_builder: Optional[IncrementalGraphBuilder] = None,
        defense_site_classes: int = 10,
        max_flows_per_action: int = 5,
        pad_length: int = 1000,
        min_dummy_packet_size: int = 200,
        max_dummy_packet_size: int = 800,
        concurrent_time_threshold: float = 1.0
    ):
        """
        Args:
            max_flows: 每个数据包最大流数量
            max_dummy_packets: 最大dummy包数量（保留用于兼容性）
            num_classes: 类别数量
            device: 设备
            reward_c: 奖励系数字典
            hidden_dim: Actor/Critic 隐藏层维度
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            gamma: 折扣因子
            tau: 软更新系数
            lr: 学习率
            alpha: 温度参数
            target_update_interval: 目标网络更新间隔
            reward_type: 奖励类型
            transformer_output_dim: Transformer 输出维度（状态维度）
            incremental_builder: Graph Builder（用于分类评估）
            defense_site_classes: 防御站点类别数量
            max_flows_per_action: 每次动作最多插入的防御流数量
            pad_length: 流的长度（序列长度）
            min_dummy_packet_size: burst 填充包的最小大小
            max_dummy_packet_size: burst 填充包的最大大小
        """
        self.max_flows = max_flows
        self.max_dummy_packets = max_dummy_packets
        self.pad_length = pad_length
        self.min_dummy_packet_size = min_dummy_packet_size
        self.max_dummy_packet_size = max_dummy_packet_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.concurrent_time_threshold = concurrent_time_threshold
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size

        # 可学习的alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = torch.tensor(alpha, device=self.device)
        self.target_entropy = -1.0

        # ========== 核心改变：使用 Transformer 替代 Graph Encoder ==========
        self.transformer = FlowSequenceTransformer(
            max_flows=max_flows,
            seq_len=pad_length,
            output_dim=transformer_output_dim,
            dropout=0.1,
            device=self.device
        ).to(self.device)
        
        # Transformer 优化器（与 Actor 一起训练）
        self.transformer_optimizer = optim.Adam(self.transformer.parameters(), lr=lr)
        
        state_size = transformer_output_dim

        # 动作空间设计
        self.max_defense_slots = max(0, defense_site_classes)
        self.max_flows_per_action = max_flows_per_action
        # action_dim = (max_defense_slots + 1) * max_flows_per_action + pad_length
        self.action_dim = (
            (self.max_defense_slots + 1) * self.max_flows_per_action + self.pad_length
            if self.max_defense_slots > 0
            else self.pad_length
        )
        self.defense_site_classes = defense_site_classes
        self.defense_pool_by_label: Dict[int, List[Dict]] = {}
        self.defense_label_list: List[int] = []

        # Actor
        self.actor = GaussianPolicy(
            state_dim=state_size,
            action_size=self.action_dim,
            hidden_dim=hidden_dim,
            action_space=ActionSpace(low=0, high=self.max_dummy_packets),
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = QNetwork(
            num_inputs=state_size,
            num_actions=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 目标网络
        self.critic_target = QNetwork(
            num_inputs=state_size,
            num_actions=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        hard_update(local_model=self.critic, target_model=self.critic_target)

        # ========== 核心改变：使用新的经验回放缓冲区 ==========
        self.memory = FlowSequenceReplayBuffer(buffer_size, batch_size)
        
        # 分类器和 Graph Builder（仅用于评估）
        self.classifier = None  # FG-net 分类器
        self.graph_builder = incremental_builder or IncrementalGraphBuilder(
            pad_length=pad_length
        )

        # Reward 系数
        self.reward_c = reward_c or {}
        self.num_classes = num_classes
        self.reward_type = reward_type

        logging.info("=" * 80)
        logging.info("FlowTrafficObfuscator 初始化完成")
        logging.info(f"  Transformer 输出维度: {state_size}")
        logging.info(f"  动作空间维度: {self.action_dim}")
        logging.info(f"  最大流数量: {max_flows}")
        logging.info(f"  经验回放类型: FlowSequenceReplayBuffer")
        logging.info("=" * 80)

    def set_classifier(self, classifier):
        """设置分类器模型（FG-net）"""
        self.classifier = classifier.to(self.device)
        self.classifier.eval()
        logging.info("FG-net classifier model has been set")

    def set_defense_pool(self, defense_flows: List[Dict]):
        """设置防御流池"""
        pool: Dict[int, List[Dict]] = {}
        for flow in defense_flows:
            label = int(flow.get("label", 0))
            pool.setdefault(label, []).append(flow)
        self.defense_pool_by_label = pool
        if pool:
            all_labels = sorted(pool.keys())
            if len(all_labels) > self.max_defense_slots:
                logging.warning(
                    "Defense pool has %d classes, but actor was initialized with %d slots. "
                    "Extra classes will be ignored.",
                    len(all_labels),
                    self.max_defense_slots,
                )
                self.defense_label_list = all_labels[: self.max_defense_slots]
            else:
                self.defense_label_list = all_labels
            self.defense_site_classes = len(self.defense_label_list)
        else:
            self.defense_label_list = []
            self.defense_site_classes = 0

    def reset_graph_builder(self):
        """重置 Graph Builder"""
        if self.graph_builder:
            self.graph_builder.reset()

    def extract_flow_features(self, flow: Dict) -> np.ndarray:
        """
        从流数据中提取特征矩阵（优化版：只提取包长度）
        
        Args:
            flow: 流字典，包含 packet_length
            
        Returns:
            features: (pad_length, 1) numpy array [packet_length]
        """
        packet_length = flow.get("packet_length", [])
        
        # 确保长度为 pad_length
        if len(packet_length) < self.pad_length:
            packet_length = list(packet_length) + [0] * (self.pad_length - len(packet_length))
        else:
            packet_length = list(packet_length[:self.pad_length])
        
        # 返回 (pad_length, 1) 矩阵
        features = np.array(packet_length, dtype=np.float32).reshape(-1, 1)
        return features

    def select_action(self, state, deterministic=False):
        """
        选择动作
        
        Args:
            state: (state_dim,) tensor，Transformer 的输出
            deterministic: 是否使用确定性策略
            
        Returns:
            action_dict: 包含 defense_labels 和 burst_padding_sequence
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        with torch.no_grad():
            if deterministic:
                _, action_mean, _ = self.actor.sample(state)
                action = action_mean
            else:
                action, _, _ = self.actor.sample(state)

        action = action.squeeze(0)
        
        # 提取 burst padding sequence（最后 pad_length 维）
        burst_padding_logits = action[-self.pad_length:]
        burst_padding_sequence = torch.nn.functional.softplus(burst_padding_logits)
        burst_padding_sequence = torch.clamp(burst_padding_sequence, 0, 5.0)

        defense_labels = []
        if self.max_defense_slots > 0:
            # 解析防御流插入动作
            slots_per_flow = self.max_defense_slots + 1
            for flow_idx in range(self.max_flows_per_action):
                start_idx = flow_idx * slots_per_flow
                end_idx = start_idx + slots_per_flow
                defense_logits = action[start_idx:end_idx].clone()
                defense_logits[0] += 5.0  # 偏向"不插入"
                defense_probs = torch.softmax(defense_logits, dim=0)
                
                if deterministic:
                    defense_choice_idx = torch.argmax(defense_probs).item()
                else:
                    defense_choice_idx = torch.multinomial(defense_probs, 1).item()
                
                if defense_choice_idx > 0:
                    defense_label = self._map_defense_choice_to_label(defense_choice_idx - 1)
                    if defense_label is not None:
                        defense_labels.append(defense_label)

        return {
            "defense_labels": defense_labels,
            "burst_padding_sequence": burst_padding_sequence.cpu().numpy(),
            "raw_action": action,
        }

    def _map_defense_choice_to_label(self, choice_idx: int) -> Optional[int]:
        """将动作索引映射到防御站点标签"""
        if not self.defense_label_list:
            return None
        idx = choice_idx % len(self.defense_label_list)
        return self.defense_label_list[idx]

    def _clone_flow(self, flow: Dict) -> Dict:
        """克隆流数据"""
        return {
            "packet_length": list(flow.get("packet_length", [])),
            "arrive_time_delta": list(flow.get("arrive_time_delta", [])),
            "start_timestamp": float(flow.get("start_timestamp", 0.0)),
            "end_timestamp": float(flow.get("end_timestamp", 0.0)),
            "label": int(flow.get("label", -1)),
            "packet_id": flow.get("packet_id", -1),
        }
    
    def _group_flows_by_time_window(self, flows: List[Dict], time_threshold: float) -> List[List[Dict]]:
        """
        按时间窗口分组流（并发流在同一窗口）
        
        Args:
            flows: 流列表
            time_threshold: 时间阈值（秒），小于此值的流视为并发
            
        Returns:
            时间窗口列表，每个窗口包含多个并发流
        """
        if not flows:
            return []
        
        windows = []
        current_window = [flows[0]]
        
        for i in range(1, len(flows)):
            time_gap = abs(flows[i].get("start_timestamp", i) - flows[i-1].get("start_timestamp", i-1))
            
            if time_gap < time_threshold:
                # 并发流，加入当前窗口
                current_window.append(flows[i])
            else:
                # 新窗口
                windows.append(current_window)
                current_window = [flows[i]]
        
        if current_window:
            windows.append(current_window)
        
        return windows

    def _apply_burst_padding_to_flow(self, flow: Dict, burst_padding_sequence: np.ndarray) -> float:
        """
        使用 SAC 生成的 burst padding sequence 为每个 burst 插入 dummy packets
        
        Args:
            flow: 流字典
            burst_padding_sequence: (pad_length,) 每个值表示在对应 burst 后插入多少个 dummy packet
            
        Returns:
            added_bytes: 插入的总字节数
        """
        if burst_padding_sequence is None or len(burst_padding_sequence) == 0:
            return 0.0

        packet_seq = list(flow.get("packet_length", []))
        delta_seq = list(flow.get("arrive_time_delta", []))
        pad_length = len(packet_seq)
        if pad_length == 0:
            return 0.0

        new_packets: List[int] = []
        new_deltas: List[float] = []
        added_bytes = 0.0
        burst_idx = 0

        idx = 0
        while idx < pad_length:
            value = packet_seq[idx]
            if value == 0:
                break
            
            new_packets.append(value)
            new_deltas.append(delta_seq[idx] if idx < len(delta_seq) else 0.0)

            current_sign = 1 if value >= 0 else -1
            next_is_new_burst = (
                idx + 1 >= pad_length or
                packet_seq[idx + 1] == 0 or
                (packet_seq[idx + 1] >= 0) != (value >= 0)
            )

            if next_is_new_burst and burst_idx < len(burst_padding_sequence):
                num_dummies = int(burst_padding_sequence[burst_idx])
                num_dummies = max(0, num_dummies)
                
                if num_dummies > 0:
                    available_time = delta_seq[idx + 1] if idx + 1 < len(delta_seq) else 1.0
                    available_time = max(available_time, 1e-3)
                    
                    for _ in range(num_dummies):
                        pkt_len = random.randint(self.min_dummy_packet_size, self.max_dummy_packet_size)
                        pkt_len = pkt_len if current_sign >= 0 else -pkt_len
                        added_bytes += abs(pkt_len)
                        new_packets.append(pkt_len)
                        new_deltas.append(available_time / (num_dummies + 1))
                
                burst_idx += 1
            idx += 1

        # 填充或截断
        if len(new_packets) > pad_length:
            new_packets = new_packets[:pad_length]
            new_deltas = new_deltas[:pad_length]
        else:
            zeros_to_add = pad_length - len(new_packets)
            new_packets.extend([0] * zeros_to_add)
            new_deltas.extend([0.0] * zeros_to_add)

        flow["packet_length"] = new_packets
        flow["arrive_time_delta"] = new_deltas
        return added_bytes

    def _sample_defense_flow(
        self,
        defense_label: int,
        reference_start_ts: float,
        defense_pool: Dict[int, List[Dict]]
    ) -> Optional[Dict]:
        """采样一条防御流"""
        if defense_label is None or defense_label not in defense_pool:
            return None
        candidates = defense_pool.get(defense_label, [])
        if not candidates:
            return None

        template = random.choice(candidates)
        defense_flow = self._clone_flow(template)
        duration = max(defense_flow["end_timestamp"] - defense_flow["start_timestamp"], 1e-3)
        defense_flow["end_timestamp"] = max(reference_start_ts - 1e-4, 0.0)
        defense_flow["start_timestamp"] = max(defense_flow["end_timestamp"] - duration, 0.0)
        return defense_flow

    def _apply_action_to_flow(
        self,
        flow: Dict,
        action_dict: Optional[Dict],
        defense_pool: Optional[Dict[int, List[Dict]]]
    ) -> (float, float, List[Dict]):
        """
        应用动作到流
        
        Returns:
            (padding_bytes, injected_bytes, defense_flows)
        """
        if action_dict is None:
            return 0.0, 0.0, []

        burst_padding_seq = action_dict.get("burst_padding_sequence")
        padding_bytes = self._apply_burst_padding_to_flow(flow, burst_padding_seq)

        defense_flows: List[Dict] = []
        injected_bytes = 0.0
        pool_mapping = defense_pool if defense_pool is not None else self.defense_pool_by_label
        
        defense_labels = action_dict.get("defense_labels", [])
        if defense_labels and pool_mapping:
            for defense_label in defense_labels:
                defense_flow = self._sample_defense_flow(
                    defense_label,
                    flow.get("start_timestamp", 0.0),
                    pool_mapping
                )
                if defense_flow is not None:
                    defense_flow["packet_id"] = flow.get("packet_id", -1)
                    defense_flows.append(defense_flow)
                    pkt_lengths = defense_flow.get("packet_length", [])
                    injected_bytes += float(np.sum(np.abs(pkt_lengths)))

        return padding_bytes, injected_bytes, defense_flows

    def _compute_reward_from_prob(
        self,
        prob: float,
        padding_bytes: float,
        injected_bytes: float,
        is_correct: bool,
        packet_total_original_bytes: float
    ) -> float:
        """
        计算 reward
        
        Args:
            prob: 分类器对真实类别的置信度
            padding_bytes: padding 开销
            injected_bytes: 注入防御流开销
            is_correct: 分类器是否正确
            packet_total_original_bytes: 整个数据包的原始字节数
        """
        coeff_dummy = float(self.reward_c.get("dummy_c", 1.0))
        coeff_inject = float(self.reward_c.get("morphing_c", 1.0))
        coeff_base = float(self.reward_c.get("base_c", 10.0))

        # Base reward: 成功欺骗 → +1，失败 → -1
        if is_correct:
            base_reward = -1.0 * coeff_base
        else:
            base_reward = 1.0 * coeff_base

        # 计算开销比例
        padding_ratio = padding_bytes / packet_total_original_bytes
        inject_ratio = injected_bytes / packet_total_original_bytes
        
        target_overhead = 0.1
        
        # Padding 惩罚
        # if padding_ratio > target_overhead:
        #     dummy_penalty = coeff_dummy * (padding_ratio / target_overhead) ** 2
        # else:
        #     dummy_penalty = coeff_dummy * padding_ratio
        dummy_penalty = coeff_dummy * padding_ratio
        
        # Inject 惩罚
        # if inject_ratio > target_overhead:
        #     inject_penalty = coeff_inject * (inject_ratio / target_overhead) ** 2
        # else:
        #     inject_penalty = coeff_inject * inject_ratio
        inject_penalty = coeff_inject * inject_ratio

        reward = base_reward - (dummy_penalty + inject_penalty)
        return reward

    def train_on_batches(self, batches: List[List[Dict]]):
        """
        ========== 核心改变：三阶段训练流程 ==========
        
        阶段1：Transformer 处理，收集经验
        阶段2：Graph Builder 分类评估（只建一次图）
        阶段3：填充 reward 并添加到经验回放
        """
        # 设置训练模式
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        self.transformer.train()
        
        stats = {
            "total_steps": 0,
            "correct_predictions": 0,
            "avg_reward": 0.0,
            "total_padding_bytes": 0.0,
            "total_injected_bytes": 0.0,
            "total_original_bytes": 0.0,
            "total_added_bytes": 0.0,
            "total_defense_flows": 0,
            "packet_count": 0,
        }
        total_reward = 0.0
        updates = 0
        pool_mapping = self.defense_pool_by_label
        packet_defense_counts: Dict[int, int] = {}
        seen_packets = set()

        for flow_batch in batches:
            # 按 packet_id 分组
            packet_flows_map: Dict[int, List[Dict]] = {}
            for flow in flow_batch:
                packet_id = flow.get("packet_id", -1)
                if packet_id >= 0:
                    packet_flows_map.setdefault(packet_id, []).append(flow)
            
            # 处理每个数据包
            for packet_id, packet_flows in packet_flows_map.items():
                seen_packets.add(packet_id)
                
                # ========== 阶段1：Transformer 处理，收集经验 ==========
                flow_sequence_matrix = np.zeros((self.max_flows, self.pad_length, 1), dtype=np.float32)
                flow_count = 0
                flow_records = []
                all_processed_flows = []  # 存储所有流（含防御流）
                
                # ========== 时间窗口批处理优化 ==========
                time_windows = self._group_flows_by_time_window(packet_flows, self.concurrent_time_threshold)
                
                for window_idx, window_flows in enumerate(time_windows):
                    window_start_count = flow_count
                    # 先填充窗口内所有流
                    for raw_flow in window_flows:
                        flow = self._clone_flow(raw_flow)
                        flow_features = self.extract_flow_features(flow)
                        if flow_count < self.max_flows:
                            flow_sequence_matrix[flow_count] = flow_features
                            flow_count += 1
                    
                    # 对整个窗口调用一次 Transformer（减少计算）
                    flow_input = torch.from_numpy(flow_sequence_matrix[np.newaxis, ...]).to(self.device)
                    num_flows_tensor = torch.tensor([flow_count], device=self.device)
                    current_state = self.transformer(flow_input, num_flows_tensor).squeeze(0).detach().cpu().numpy()
                    
                    # 对窗口内每个流：使用相同 state，但分别采样不同 action
                    for flow_idx_in_window, raw_flow in enumerate(window_flows):

                        # 【修改点 1】: 在动作发生前，先对当前矩阵进行快照 (这是 S_t)
                        # 即使 current_state 是旧的，Buffer 里的矩阵数据最好是真实的物理状态
                        state_before_action = flow_sequence_matrix.copy()
                        num_flows_before = flow_count

                        # 每个流独立采样动作
                        action_dict = self.select_action(current_state)
                        flow = self._clone_flow(raw_flow)
                        
                        # 应用动作
                        padding_bytes, injected_bytes, defense_flows = self._apply_action_to_flow(
                            flow, action_dict, pool_mapping
                        )


                        # ============================================================
                        # 【修改点 2】: 必须把 Padding 后的新特征写回矩阵！(物理状态更新)
                        # ============================================================
                        matrix_idx = window_start_count + flow_idx_in_window
                        if matrix_idx < self.max_flows:
                            # 重新提取特征 (包含了 padding 后的 packet_length)
                            updated_features = self.extract_flow_features(flow)
                            # 覆盖旧特征
                            flow_sequence_matrix[matrix_idx] = updated_features
                            # ============================================================
                        
                        # 收集处理后的流
                        all_processed_flows.append(flow)
                        original_bytes = float(np.sum(np.abs(flow.get("packet_length", []))))
                        
                        # 插入防御流到序列矩阵
                        for defense_flow in defense_flows:
                            all_processed_flows.append(defense_flow)
                            if flow_count < self.max_flows:
                                defense_features = self.extract_flow_features(defense_flow)
                                flow_sequence_matrix[flow_count] = defense_features
                                flow_count += 1
                            if packet_id >= 0:
                                packet_defense_counts[packet_id] = packet_defense_counts.get(packet_id, 0) + 1
                        
                        # 【修改点 3】: 此时矩阵已更新，可以计算 Next State 了
                        # 虽然这会调用 Transformer，但这是计算 S_{t+1} 所必须的
                        # 如果这一步也想省，那 S_{t+1} 就不准确了
                        state_after_action = flow_sequence_matrix.copy()
                        num_flows_after = min(flow_count, self.max_flows)

                        next_state = self.transformer(
                            torch.from_numpy(state_after_action[np.newaxis, ...]).to(self.device),
                            torch.tensor([num_flows_after], device=self.device)
                        ).squeeze(0).detach().cpu().numpy()
                        
                        # 记录经验（窗口内流共享 state，但有独立 action）
                        flow_records.append({
                            'flow_sequence': state_before_action,
                            'num_flows_before': num_flows_before,
                            'action': action_dict,
                            'next_flow_sequence': state_after_action,
                            'next_num_flows': num_flows_after,
                            'padding_bytes': padding_bytes,
                            'injected_bytes': injected_bytes,
                            'original_bytes': original_bytes,
                            'reward': None,
                            'is_last': (window_idx == len(time_windows) - 1 and flow_idx_in_window == len(window_flows) - 1)
                        })
                        current_state = next_state
                
                # ========== 阶段2：只建一次图，用于分类评估 ==========
                if flow_records and self.classifier is not None:
                    self.reset_graph_builder()
                    for proc_flow in all_processed_flows:
                        self.graph_builder.step(proc_flow)
                    
                    classifier_graph = self.graph_builder._build_graph(include_burst=False)
                    if classifier_graph is not None and classifier_graph.num_nodes() > 0:
                        true_label = int(packet_flows[0].get("label", -1))
                        
                        with torch.no_grad():
                            graph_batch = dgl.batch([classifier_graph.to(self.device)])
                            logits = self.classifier(graph_batch)
                        
                        prob = torch.softmax(logits, dim=1)[0, true_label].item()
                        predicted_label = torch.argmax(logits).item()
                        is_correct = (predicted_label == true_label)
                        
                        # 计算整个数据包的原始字节数
                        packet_total_bytes = sum([r['original_bytes'] for r in flow_records])
                        
                        # ========== 阶段3：填充 reward 并添加到经验回放 ==========
                        for record in flow_records:
                            reward = self._compute_reward_from_prob(
                                prob,
                                record['padding_bytes'],
                                record['injected_bytes'],
                                is_correct,
                                packet_total_bytes
                            )
                            record['reward'] = reward
                            
                            # 添加到经验回放
                            done_flag = 1.0 if record['is_last'] else 0.0
                            self.memory.add(
                                flow_sequence=record['flow_sequence'],
                                num_flows=record['num_flows_before'],
                                action=record['action']['raw_action'].cpu().numpy(),
                                reward=reward,
                                next_flow_sequence=record['next_flow_sequence'],
                                next_num_flows=record['next_num_flows'],
                                done=done_flag
                            )
                            
                            # 更新统计
                            total_reward += reward
                            stats["total_steps"] += 1
                            if is_correct:
                                stats["correct_predictions"] += 1
                            stats["total_padding_bytes"] += record["padding_bytes"]
                            stats["total_injected_bytes"] += record["injected_bytes"]
                            stats["total_original_bytes"] += record["original_bytes"]
                            stats["total_added_bytes"] += record["padding_bytes"] + record["injected_bytes"]
                            
                            # 训练网络
                            if len(self.memory) > self.batch_size:
                                experiences = self.memory.sample()
                                self.learn(experiences, updates)
                                updates += 1

        if stats["total_steps"] > 0:
            stats["avg_reward"] = total_reward / stats["total_steps"]
        total_packets = len(seen_packets)
        stats["packet_count"] = total_packets
        stats["total_defense_flows"] = sum(packet_defense_counts.values())
        stats["avg_defense_flows_per_packet"] = (
            stats["total_defense_flows"] / total_packets if total_packets > 0 else 0.0
        )
        stats["defense_overhead_ratio"] = (
            stats["total_added_bytes"] / max(stats["total_original_bytes"], 1e-6)
            if stats["total_original_bytes"] > 0 else 0.0
        )
        return stats

    def learn(self, experiences, updates):
        """
        ========== 核心改变：重新编码状态，Transformer 可训练 ==========
        
        从经验中学习，更新 Actor、Critic 和 Transformer
        """
        (flow_seqs, num_flows, actions, rewards,
         next_flow_seqs, next_num_flows, dones) = experiences
        
        # 转换为 tensor
        flow_seqs = flow_seqs.to(self.device)
        num_flows = num_flows.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_flow_seqs = next_flow_seqs.to(self.device)
        next_num_flows = next_num_flows.to(self.device)
        dones = dones.to(self.device)

        # ========== 重新通过 Transformer 编码（梯度可以传播！）==========
        states = self.transformer(flow_seqs, num_flows)
        with torch.no_grad():
            next_states = self.transformer(next_flow_seqs, next_num_flows)

        # ---------------------------- 优化Critic ---------------------------- #
        # Critic 不需要更新 Transformer，使用 detach
        q1_current, q2_current = self.critic(states.detach(), actions)

        with torch.no_grad():
            next_actions, _, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target = rewards + (1 - dones) * self.gamma * (q_target - self.alpha * next_log_probs)
            if target.dim() > 2:
                target = target.squeeze(dim=1)

        critic1_loss = F.mse_loss(q1_current, target)
        critic2_loss = F.mse_loss(q2_current, target)
        q_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------------------------- 优化Actor + Transformer ---------------------------- #
        # Actor 使用未 detach 的 states，梯度可以传播到 Transformer
        new_actions, _, log_probs = self.actor.sample(states)
        # Critic 在这里不需要更新，所以对其输入使用 detach
        q1, q2 = self.critic(states.detach(), new_actions)
        q_min = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q_min).mean()

        self.actor_optimizer.zero_grad()
        self.transformer_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.transformer_optimizer.step()

        # ---------------------------- 优化alpha ---------------------------- #
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 软更新目标网络
        if updates % self.target_update_interval == 0:
            soft_update(self.critic, self.critic_target, self.tau)

        # 每100次更新记录一次
        if updates % 100 == 0:
            logging.debug(
                f"[SAC Training] Update {updates}: "
                f"Critic Loss: {q_loss.item():.4f}, "
                f"Actor Loss: {actor_loss.item():.4f}, "
                f"Alpha: {self.alpha.item():.4f}"
            )

    def save_checkpoint(self, ckpt_path='./saved_models/flow-sac-ckpt.pth'):
        """保存模型参数"""
        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save({
            'policy_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'transformer_state_dict': self.transformer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'transformer_optimizer_state_dict': self.transformer_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
        }, ckpt_path)
        logging.info(f"Model saved to {ckpt_path}")

    def load_checkpoint(self, ckpt_path='./saved_models/flow-sac-ckpt.pth', evaluate=False):
        """加载模型参数"""
        if ckpt_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            if 'transformer_state_dict' in checkpoint:
                self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if 'transformer_optimizer_state_dict' in checkpoint:
                self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer_state_dict'])
            if 'alpha_optimizer_state_dict' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.transformer.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()
                self.transformer.train()
            logging.info(f"Model loaded from {ckpt_path}")

    def evaluate_on_test_data(self, test_batches: List[List[Dict]]):
        """在测试数据上评估防御效果"""
        self.actor.eval()
        self.transformer.eval()
        if self.classifier:
            self.classifier.eval()

        class_stats: Dict[int, Dict] = {}
        pool_mapping = self.defense_pool_by_label

        with torch.no_grad():
            for flow_batch in test_batches:
                # 使用相同的三阶段流程（但不训练）
                packet_flows_map: Dict[int, List[Dict]] = {}
                for flow in flow_batch:
                    packet_id = flow.get("packet_id", -1)
                    if packet_id >= 0:
                        packet_flows_map.setdefault(packet_id, []).append(flow)
                
                for packet_id, packet_flows in packet_flows_map.items():
                    # 阶段1：处理
                    flow_sequence_matrix = np.zeros((self.max_flows, self.pad_length, 1), dtype=np.float32)
                    flow_count = 0
                    all_processed_flows = []
                    total_padding = 0.0
                    total_injected = 0.0
                    total_original = 0.0
                    
                    # ========== 时间窗口批处理优化（评估模式）==========
                    time_windows = self._group_flows_by_time_window(packet_flows, self.concurrent_time_threshold)
                    
                    for window_flows in time_windows:
                        # 先填充窗口内所有流
                        window_start_count = flow_count
                        for raw_flow in window_flows:
                            flow = self._clone_flow(raw_flow)
                            flow_features = self.extract_flow_features(flow)
                            if flow_count < self.max_flows:
                                flow_sequence_matrix[flow_count] = flow_features
                                flow_count += 1
                        
                        # 对整个窗口调用一次 Transformer
                        current_state = self.transformer(
                            torch.from_numpy(flow_sequence_matrix[np.newaxis, ...]).to(self.device),
                            torch.tensor([flow_count], device=self.device)
                        ).squeeze(0).detach().cpu().numpy()
                        
                        # 对窗口内每个流：使用相同 state，但分别采样不同 action
                        for flow_idx_in_window, raw_flow in enumerate(window_flows):
                            # 【修改点 1】: 在动作发生前，先对当前矩阵进行快照 (评估时不需要保存，但保持逻辑一致性)
                            # state_before_action = flow_sequence_matrix.copy()  # 评估时不需要
                            
                            # 每个流独立采样动作
                            action_dict = self.select_action(current_state, deterministic=True)
                            flow = self._clone_flow(raw_flow)
                            true_label = int(flow.get("label", -1))
                            
                            padding_bytes, injected_bytes, defense_flows = self._apply_action_to_flow(
                                flow, action_dict, pool_mapping
                            )
                            
                            # ============================================================
                            # 【修改点 2】: 必须把 Padding 后的新特征写回矩阵！(物理状态更新)
                            # ============================================================
                            matrix_idx = window_start_count + flow_idx_in_window
                            if matrix_idx < self.max_flows:
                                # 重新提取特征 (包含了 padding 后的 packet_length)
                                updated_features = self.extract_flow_features(flow)
                                # 覆盖旧特征
                                flow_sequence_matrix[matrix_idx] = updated_features
                            # ============================================================
                            
                            all_processed_flows.append(flow)
                            total_padding += padding_bytes
                            total_injected += injected_bytes
                            total_original += float(np.sum(np.abs(flow.get("packet_length", []))))
                            
                            # 插入防御流到序列矩阵
                            for defense_flow in defense_flows:
                                all_processed_flows.append(defense_flow)
                                if flow_count < self.max_flows:
                                    defense_features = self.extract_flow_features(defense_flow)
                                    flow_sequence_matrix[flow_count] = defense_features
                                    flow_count += 1
                            
                            # 【修改点 3】: 评估时不需要计算 next_state，因为不需要保存经验
                            # 但如果后续流需要基于更新后的状态采样动作，可以在这里更新 current_state
                            # 注意：当前实现中窗口内流共享 current_state，所以这里不更新
                    
                    # 阶段2：分类
                    if self.classifier and all_processed_flows:
                        self.reset_graph_builder()
                        for proc_flow in all_processed_flows:
                            self.graph_builder.step(proc_flow)
                        
                        classifier_graph = self.graph_builder._build_graph(include_burst=False)
                        if classifier_graph and classifier_graph.num_nodes() > 0:
                            graph_batch = dgl.batch([classifier_graph.to(self.device)])
                            logits = self.classifier(graph_batch)
                            predicted_label = torch.argmax(logits).item()
                            
                            if true_label not in class_stats:
                                class_stats[true_label] = {
                                    "correct": 0,
                                    "total": 0,
                                    "total_original_bytes": 0.0,
                                    "total_added_bytes": 0.0
                                }
                            
                            class_stats[true_label]["total"] += 1
                            if predicted_label == true_label:
                                class_stats[true_label]["correct"] += 1
                            class_stats[true_label]["total_original_bytes"] += total_original
                            class_stats[true_label]["total_added_bytes"] += (total_padding + total_injected)

        # 计算总体统计
        total_correct = sum(s["correct"] for s in class_stats.values())
        total_samples = sum(s["total"] for s in class_stats.values())
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        total_original = sum(s["total_original_bytes"] for s in class_stats.values())
        total_added = sum(s["total_added_bytes"] for s in class_stats.values())
        avg_overhead = total_added / total_original if total_original > 0 else 0.0

        # 转换为期望格式
        class_details = {}
        for label, stats in class_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            cost_ratio = stats["total_added_bytes"] / stats["total_original_bytes"] if stats["total_original_bytes"] > 0 else 0.0
            class_details[label] = {
                "accuracy": accuracy,
                "cost_ratio": cost_ratio,
                "correct": stats["correct"],
                "total": stats["total"],
                "original_bytes": stats["total_original_bytes"],
                "defense_bytes": stats["total_added_bytes"]
            }
        
        return {
            "overall_accuracy": overall_accuracy,
            "avg_cost_ratio": avg_overhead,
            "class_details": class_details,
            "total_samples": total_samples
        }

