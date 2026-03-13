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
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class FlowTrafficObfuscator:

    def __init__(self, max_flows=50, max_dummy_packets=10, num_classes=1, device='cuda', reward_c=None, hidden_dim=1024, buffer_size=int(1000000.0), batch_size=64, gamma=0.99, tau=0.005, lr=0.0003, alpha=0.2, target_update_interval=1, reward_type='o', transformer_output_dim=256, incremental_builder: Optional[IncrementalGraphBuilder]=None, defense_site_classes: int=10, pad_length: int=1000, min_dummy_packet_size: int=200, max_dummy_packet_size: int=800, concurrent_time_threshold: float=1.0, max_bursts: int=50, ablation_no_positional_encoding=False, ablation_no_window_attention=False, ablation_no_flow_inter_attention=False, ablation_no_burst_padding=False, ablation_no_defense_flow=False):
        self.max_flows = max_flows
        self.max_dummy_packets = max_dummy_packets
        self.pad_length = pad_length
        self.min_dummy_packet_size = min_dummy_packet_size
        self.max_dummy_packet_size = max_dummy_packet_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.concurrent_time_threshold = concurrent_time_threshold
        self.max_bursts = max_bursts
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.ablation_no_positional_encoding = ablation_no_positional_encoding
        self.ablation_no_window_attention = ablation_no_window_attention
        self.ablation_no_flow_inter_attention = ablation_no_flow_inter_attention
        self.ablation_no_burst_padding = ablation_no_burst_padding
        self.ablation_no_defense_flow = ablation_no_defense_flow
        ablation_flags = [(ablation_no_positional_encoding, 'Positional Encoding'), (ablation_no_window_attention, 'Window Attention'), (ablation_no_flow_inter_attention, 'Flow Inter-Attention'), (ablation_no_burst_padding, 'Burst Padding'), (ablation_no_defense_flow, 'Defense Flow Injection')]
        active_ablations = [name for flag, name in ablation_flags if flag]
        if active_ablations:
            logging.warning('=' * 80)
            logging.warning('ABLATION STUDY MODE ENABLED')
            logging.warning('=' * 80)
            for name in active_ablations:
                logging.warning(f'  [ABLATED] {name}')
            logging.warning('=' * 80)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha = torch.tensor(alpha, device=self.device)
        self.transformer = FlowSequenceTransformer(max_flows=max_flows, seq_len=pad_length, output_dim=transformer_output_dim, dropout=0.1, device=self.device, use_positional_encoding=not ablation_no_positional_encoding, use_window_attention=not ablation_no_window_attention, use_flow_inter_attention=not ablation_no_flow_inter_attention).to(self.device)
        self.transformer_optimizer = optim.Adam(self.transformer.parameters(), lr=lr)
        state_size = transformer_output_dim
        self.defense_site_classes = defense_site_classes
        self.action_dim = defense_site_classes + self.max_bursts
        self.defense_pool_by_label: Dict[int, List[Dict]] = {}
        self.defense_label_list: List[int] = []
        self.target_entropy = -10.0 * float(self.action_dim)
        self.actor = GaussianPolicy(state_dim=state_size, action_size=self.action_dim, hidden_dim=hidden_dim, action_space=ActionSpace(low=0, high=self.max_dummy_packets)).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = QNetwork(num_inputs=state_size, num_actions=self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(num_inputs=state_size, num_actions=self.action_dim, hidden_dim=hidden_dim).to(self.device)
        hard_update(local_model=self.critic, target_model=self.critic_target)
        self.memory = FlowSequenceReplayBuffer(buffer_size, batch_size)
        self.classifier = None
        self.classifier_type = 'fgnet'
        self.graph_builder = incremental_builder or IncrementalGraphBuilder(pad_length=pad_length)
        self.reward_c = reward_c or {}
        self.num_classes = num_classes
        self.reward_type = reward_type
        logging.info('=' * 80)
        logging.info('FlowTrafficObfuscator 初始化完成')
        logging.info(f'  Transformer 输出维度: {state_size}')
        logging.info(f'  动作空间维度: {self.action_dim}')
        logging.info(f'  最大流数量: {max_flows}')
        logging.info(f'  经验回放类型: FlowSequenceReplayBuffer')
        logging.info('=' * 80)

    def set_classifier(self, classifier, classifier_type: str='fgnet'):
        self.classifier = classifier.to(self.device)
        self.classifier.eval()
        self.classifier_type = classifier_type
        logging.info(f'{classifier_type.upper()} classifier model has been set')

    def set_defense_pool(self, defense_flows: List[Dict]):
        pool: Dict[int, List[Dict]] = {}
        for flow in defense_flows:
            label = int(flow.get('label', 0))
            pool.setdefault(label, []).append(flow)
        self.defense_pool_by_label = pool
        if pool:
            all_labels = sorted(pool.keys())
            if len(all_labels) > self.defense_site_classes:
                logging.warning('Defense pool has %d classes, but actor was initialized with %d classes. Extra classes will be ignored.', len(all_labels), self.defense_site_classes)
                self.defense_label_list = all_labels[:self.defense_site_classes]
            else:
                self.defense_label_list = all_labels
            actual_classes = len(self.defense_label_list)
            if actual_classes < self.defense_site_classes:
                logging.info(f'Updating action space: defense_site_classes from {self.defense_site_classes} to {actual_classes}')
                self.defense_site_classes = actual_classes
                self.action_dim = actual_classes + self.max_bursts
                logging.warning('Action space dimension changed. Actor and Critic networks need to be reinitialized!')
        else:
            self.defense_label_list = []
            self.defense_site_classes = 0

    def reset_graph_builder(self):
        if self.graph_builder:
            self.graph_builder.reset()

    def extract_flow_features(self, flow: Dict) -> np.ndarray:
        packet_length = flow.get('packet_length', [])
        if len(packet_length) < self.pad_length:
            packet_length = list(packet_length) + [0] * (self.pad_length - len(packet_length))
        else:
            packet_length = list(packet_length[:self.pad_length])
        features = np.array(packet_length, dtype=np.float32).reshape(-1, 1)
        return features

    def select_action(self, state, deterministic=False):
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
        burst_padding_logits = action[-self.max_bursts:]
        burst_padding_sequence = torch.clamp(burst_padding_logits, min=0.0)
        defense_counts = {}
        if self.defense_site_classes > 0 and len(self.defense_label_list) > 0:
            defense_action_values = action[:self.defense_site_classes]
            defense_counts_raw = torch.clamp(defense_action_values, min=0.0)
            for idx, label in enumerate(self.defense_label_list):
                if idx < len(defense_counts_raw):
                    count = int(defense_counts_raw[idx].item())
                    if count > 0:
                        defense_counts[label] = count
        return {'defense_counts': defense_counts, 'burst_padding_sequence': burst_padding_sequence.cpu().numpy(), 'raw_action': action}

    def _clone_flow(self, flow: Dict) -> Dict:
        return {'packet_length': list(flow.get('packet_length', [])), 'arrive_time_delta': list(flow.get('arrive_time_delta', [])), 'start_timestamp': float(flow.get('start_timestamp', 0.0)), 'end_timestamp': float(flow.get('end_timestamp', 0.0)), 'label': int(flow.get('label', -1)), 'packet_id': flow.get('packet_id', -1)}

    def _group_flows_by_time_window(self, flows: List[Dict], time_threshold: float) -> List[List[Dict]]:
        if not flows:
            return []
        windows = []
        current_window = [flows[0]]
        for i in range(1, len(flows)):
            time_gap = abs(flows[i].get('start_timestamp', i) - flows[i - 1].get('start_timestamp', i - 1))
            if time_gap < time_threshold:
                current_window.append(flows[i])
            else:
                windows.append(current_window)
                current_window = [flows[i]]
        if current_window:
            windows.append(current_window)
        return windows

    def _apply_burst_padding_to_flow(self, flow: Dict, burst_padding_sequence: np.ndarray) -> float:
        if self.ablation_no_burst_padding:
            return 0.0
        if burst_padding_sequence is None or len(burst_padding_sequence) == 0:
            return 0.0
        packet_seq = list(flow.get('packet_length', []))
        delta_seq = list(flow.get('arrive_time_delta', []))
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
            next_is_new_burst = idx + 1 >= pad_length or packet_seq[idx + 1] == 0 or (packet_seq[idx + 1] >= 0) != (value >= 0)
            if next_is_new_burst and burst_idx < len(burst_padding_sequence):
                num_dummies = int(burst_padding_sequence[burst_idx])
                num_dummies = max(0, num_dummies)
                if num_dummies > 0:
                    available_time = delta_seq[idx + 1] if idx + 1 < len(delta_seq) else 1.0
                    available_time = max(available_time, 0.001)
                    for _ in range(num_dummies):
                        pkt_len = random.randint(self.min_dummy_packet_size, self.max_dummy_packet_size)
                        pkt_len = pkt_len if current_sign >= 0 else -pkt_len
                        added_bytes += abs(pkt_len)
                        new_packets.append(pkt_len)
                        new_deltas.append(available_time / (num_dummies + 1))
                burst_idx += 1
            idx += 1
        if len(new_packets) > self.pad_length:
            new_packets = new_packets[:self.pad_length]
            new_deltas = new_deltas[:self.pad_length]
        else:
            zeros_to_add = self.pad_length - len(new_packets)
            new_packets.extend([0] * zeros_to_add)
            new_deltas.extend([0.0] * zeros_to_add)
        flow['packet_length'] = new_packets
        flow['arrive_time_delta'] = new_deltas
        return added_bytes

    def _sample_defense_flow(self, defense_label: int, reference_start_ts: float, defense_pool: Dict[int, List[Dict]]) -> Optional[Dict]:
        if defense_label is None or defense_label not in defense_pool:
            return None
        candidates = defense_pool.get(defense_label, [])
        if not candidates:
            return None
        template = random.choice(candidates)
        defense_flow = self._clone_flow(template)
        duration = max(defense_flow['end_timestamp'] - defense_flow['start_timestamp'], 0.001)
        defense_flow['end_timestamp'] = max(reference_start_ts - 0.0001, 0.0)
        defense_flow['start_timestamp'] = max(defense_flow['end_timestamp'] - duration, 0.0)
        return defense_flow

    def _apply_action_to_flow(self, flow: Dict, action_dict: Optional[Dict], defense_pool: Optional[Dict[int, List[Dict]]]) -> (float, float, List[Dict]):
        if action_dict is None:
            return (0.0, 0.0, [])
        burst_padding_seq = action_dict.get('burst_padding_sequence')
        padding_bytes = self._apply_burst_padding_to_flow(flow, burst_padding_seq)
        defense_flows: List[Dict] = []
        injected_bytes = 0.0
        if not self.ablation_no_defense_flow:
            pool_mapping = defense_pool if defense_pool is not None else self.defense_pool_by_label
            defense_counts = action_dict.get('defense_counts', {})
            if defense_counts and pool_mapping:
                for defense_label, count in defense_counts.items():
                    for _ in range(count):
                        defense_flow = self._sample_defense_flow(defense_label, flow.get('start_timestamp', 0.0), pool_mapping)
                    if defense_flow is not None:
                        defense_flow['packet_id'] = flow.get('packet_id', -1)
                        defense_flows.append(defense_flow)
                        pkt_lengths = defense_flow.get('packet_length', [])
                        injected_bytes += float(np.sum(np.abs(pkt_lengths)))
        return (padding_bytes, injected_bytes, defense_flows)

    def _compute_reward_from_prob(self, prob: float, padding_bytes: float, injected_bytes: float, is_correct: bool, packet_total_original_bytes: float) -> float:
        TARGET_OVERHEAD = 0.2
        TARGET_CONFIDENCE = 0.02
        packet_total_original_bytes = max(packet_total_original_bytes, 100.0)
        coeff_dummy = float(self.reward_c.get('dummy_c', 1.0))
        coeff_inject = float(self.reward_c.get('morphing_c', 1.0))
        coeff_base = float(self.reward_c.get('base_c', 10.0))
        padding_ratio = padding_bytes / packet_total_original_bytes
        inject_ratio = injected_bytes / packet_total_original_bytes
        if prob > TARGET_CONFIDENCE:
            base_reward = -1.0 * coeff_base * (1.0 + prob - TARGET_CONFIDENCE)
        else:
            base_reward = 0.2 * coeff_base * (TARGET_CONFIDENCE - prob)
        dummy_penalty = coeff_dummy * padding_ratio
        inject_penalty = coeff_inject * inject_ratio
        reward = base_reward - (dummy_penalty + inject_penalty)
        return reward

    def train_on_batches(self, batches: List[List[Dict]]):
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        self.transformer.train()
        stats = {'total_steps': 0, 'correct_predictions': 0, 'avg_reward': 0.0, 'total_padding_bytes': 0.0, 'total_injected_bytes': 0.0, 'total_original_bytes': 0.0, 'total_added_bytes': 0.0, 'total_defense_flows': 0, 'packet_count': 0}
        total_reward = 0.0
        updates = 0
        pool_mapping = self.defense_pool_by_label
        packet_defense_counts: Dict[int, int] = {}
        seen_packets = set()
        batch_packet_data = []
        for flow_batch in batches:
            packet_flows_map: Dict[int, List[Dict]] = {}
            for flow in flow_batch:
                packet_id = flow.get('packet_id', -1)
                if packet_id >= 0:
                    packet_flows_map.setdefault(packet_id, []).append(flow)
            for packet_id, packet_flows in packet_flows_map.items():
                seen_packets.add(packet_id)
                flow_sequence_matrix = torch.zeros((self.max_flows, self.pad_length, 1), dtype=torch.float32, device=self.device)
                flow_count = 0
                flow_records = []
                all_processed_flows = []
                time_windows = self._group_flows_by_time_window(packet_flows, self.concurrent_time_threshold)
                for window_idx, window_flows in enumerate(time_windows):
                    window_start_count = flow_count
                    window_features_list = []
                    window_update_indices = []
                    for raw_flow in window_flows:
                        flow = self._clone_flow(raw_flow)
                        if flow_count < self.max_flows:
                            feat = self.extract_flow_features(flow)
                            window_features_list.append(feat)
                            window_update_indices.append(flow_count)
                            flow_count += 1
                    if window_features_list:
                        features_np = np.stack(window_features_list)
                        features_tensor = torch.from_numpy(features_np).to(self.device)
                        start_idx = window_update_indices[0]
                        end_idx = window_update_indices[-1] + 1
                        flow_sequence_matrix[start_idx:end_idx] = features_tensor
                    flow_input = flow_sequence_matrix.unsqueeze(0)
                    num_flows_tensor = torch.tensor([flow_count], device=self.device)
                    current_state = self.transformer(flow_input, num_flows_tensor).squeeze(0)
                    for flow_idx_in_window, raw_flow in enumerate(window_flows):
                        state_before_action = flow_sequence_matrix.clone()
                        num_flows_before = flow_count
                        action_dict = self.select_action(current_state, deterministic=False)
                        flow = self._clone_flow(raw_flow)
                        padding_bytes, injected_bytes, defense_flows = self._apply_action_to_flow(flow, action_dict, pool_mapping)
                        matrix_idx = window_start_count + flow_idx_in_window
                        if matrix_idx < self.max_flows:
                            updated_features = self.extract_flow_features(flow)
                            flow_sequence_matrix[matrix_idx] = torch.from_numpy(updated_features).to(self.device)
                        all_processed_flows.append(flow)
                        original_bytes = float(np.sum(np.abs(flow.get('packet_length', []))))
                        defense_feats_list = []
                        defense_indices = []
                        for defense_flow in defense_flows:
                            all_processed_flows.append(defense_flow)
                            if flow_count < self.max_flows:
                                defense_features = self.extract_flow_features(defense_flow)
                                defense_feats_list.append(defense_features)
                                defense_indices.append(flow_count)
                                flow_count += 1
                            if packet_id >= 0:
                                packet_defense_counts[packet_id] = packet_defense_counts.get(packet_id, 0) + 1
                        if defense_feats_list:
                            d_feats_np = np.stack(defense_feats_list)
                            d_feats_tensor = torch.from_numpy(d_feats_np).to(self.device)
                            start_d = defense_indices[0]
                            end_d = defense_indices[-1] + 1
                            flow_sequence_matrix[start_d:end_d] = d_feats_tensor
                        state_after_action = flow_sequence_matrix.clone()
                        num_flows_after = min(flow_count, self.max_flows)
                        next_state_gpu = self.transformer(state_after_action.unsqueeze(0), torch.tensor([num_flows_after], device=self.device)).squeeze(0)
                        flow_records.append({'flow_sequence': state_before_action.cpu().numpy(), 'num_flows_before': num_flows_before, 'action': action_dict, 'next_flow_sequence': state_after_action.cpu().numpy(), 'next_num_flows': num_flows_after, 'padding_bytes': padding_bytes, 'injected_bytes': injected_bytes, 'original_bytes': original_bytes, 'reward': None, 'is_last': window_idx == len(time_windows) - 1 and flow_idx_in_window == len(window_flows) - 1})
                        current_state = next_state_gpu
                if flow_records:
                    batch_packet_data.append({'processed_flows': all_processed_flows, 'records': flow_records, 'true_label': int(packet_flows[0].get('label', -1)), 'packet_id': packet_id})
        if self.classifier is not None and batch_packet_data:
            graphs = []
            valid_indices = []
            for i, p_data in enumerate(batch_packet_data):
                self.reset_graph_builder()
                for proc_flow in p_data['processed_flows']:
                    self.graph_builder.step(proc_flow)
                classifier_graph = self.graph_builder.build_classifier_graph(classifier_type=self.classifier_type)
                if classifier_graph is not None and classifier_graph.num_nodes() > 0:
                    graphs.append(classifier_graph.to(self.device))
                    valid_indices.append(i)
            if graphs:
                graph_batch = dgl.batch(graphs)
                with torch.no_grad():
                    logits = self.classifier(graph_batch)
                    probs = torch.softmax(logits, dim=1)
                for batch_idx, global_idx in enumerate(valid_indices):
                    p_data = batch_packet_data[global_idx]
                    true_label = p_data['true_label']
                    flow_records = p_data['records']
                    sample_prob = probs[batch_idx, true_label].item()
                    predicted_label = torch.argmax(logits[batch_idx]).item()
                    is_correct = predicted_label == true_label
                    for record in flow_records:
                        reward = self._compute_reward_from_prob(sample_prob, record['padding_bytes'], record['injected_bytes'], is_correct, record['original_bytes'])
                        record['reward'] = reward
                        done_flag = 1.0 if record['is_last'] else 0.0
                        self.memory.add(flow_sequence=record['flow_sequence'], num_flows=record['num_flows_before'], action=record['action']['raw_action'].cpu().numpy(), reward=reward, next_flow_sequence=record['next_flow_sequence'], next_num_flows=record['next_num_flows'], done=done_flag)
                        total_reward += reward
                        stats['total_steps'] += 1
                        if is_correct:
                            stats['correct_predictions'] += 1
                        stats['total_padding_bytes'] += record['padding_bytes']
                        stats['total_injected_bytes'] += record['injected_bytes']
                        stats['total_original_bytes'] += record['original_bytes']
                        stats['total_added_bytes'] += record['padding_bytes'] + record['injected_bytes']
                        if len(self.memory) > self.batch_size:
                            experiences = self.memory.sample()
                            self.learn(experiences, updates)
                            updates += 1
        if stats['total_steps'] > 0:
            stats['avg_reward'] = total_reward / stats['total_steps']
        total_packets = len(seen_packets)
        stats['packet_count'] = total_packets
        stats['total_defense_flows'] = sum(packet_defense_counts.values())
        stats['avg_defense_flows_per_packet'] = stats['total_defense_flows'] / total_packets if total_packets > 0 else 0.0
        stats['defense_overhead_ratio'] = stats['total_added_bytes'] / max(stats['total_original_bytes'], 1e-06) if stats['total_original_bytes'] > 0 else 0.0
        return stats

    def learn(self, experiences, updates):
        flow_seqs, num_flows, actions, rewards, next_flow_seqs, next_num_flows, dones = experiences
        flow_seqs = flow_seqs.to(self.device)
        num_flows = num_flows.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_flow_seqs = next_flow_seqs.to(self.device)
        next_num_flows = next_num_flows.to(self.device)
        dones = dones.to(self.device)
        states = self.transformer(flow_seqs, num_flows)
        with torch.no_grad():
            next_states = self.transformer(next_flow_seqs, next_num_flows)
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
        new_actions, _, log_probs = self.actor.sample(states)
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
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        if updates % self.target_update_interval == 0:
            soft_update(self.critic, self.critic_target, self.tau)
        if updates % 100 == 0:
            logging.debug(f'[SAC Training] Update {updates}: Critic Loss: {q_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Alpha: {self.alpha.item():.4f}')

    def save_checkpoint(self, ckpt_path='./saved_models/flow-sac-ckpt.pth'):
        ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        torch.save({'policy_state_dict': self.actor.state_dict(), 'critic_state_dict': self.critic.state_dict(), 'critic_target_state_dict': self.critic_target.state_dict(), 'transformer_state_dict': self.transformer.state_dict(), 'critic_optimizer_state_dict': self.critic_optimizer.state_dict(), 'policy_optimizer_state_dict': self.actor_optimizer.state_dict(), 'transformer_optimizer_state_dict': self.transformer_optimizer.state_dict(), 'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path='./saved_models/flow-sac-ckpt.pth', evaluate=False):
        if ckpt_path is not None and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            if 'transformer_state_dict' in checkpoint:
                self.transformer.load_state_dict(checkpoint['transformer_state_dict'], strict=False)
            if not evaluate:
                try:
                    if 'critic_optimizer_state_dict' in checkpoint:
                        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                    if 'policy_optimizer_state_dict' in checkpoint:
                        self.actor_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                    if 'transformer_optimizer_state_dict' in checkpoint:
                        self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer_state_dict'])
                    if 'alpha_optimizer_state_dict' in checkpoint:
                        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                except (ValueError, KeyError) as e:
                    logging.warning(f'Failed to load optimizer state dict: {e}. This is OK for evaluation mode.')
            else:
                try:
                    if 'critic_optimizer_state_dict' in checkpoint:
                        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                except (ValueError, KeyError):
                    pass
                try:
                    if 'policy_optimizer_state_dict' in checkpoint:
                        self.actor_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                except (ValueError, KeyError):
                    pass
                try:
                    if 'transformer_optimizer_state_dict' in checkpoint:
                        self.transformer_optimizer.load_state_dict(checkpoint['transformer_optimizer_state_dict'])
                except (ValueError, KeyError):
                    pass
                try:
                    if 'alpha_optimizer_state_dict' in checkpoint:
                        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                except (ValueError, KeyError):
                    pass
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
            logging.info(f'Model loaded from {ckpt_path}')

    def evaluate_on_test_data(self, test_batches: List[List[Dict]], num_classes: int=None):
        self.actor.eval()
        self.transformer.eval()
        if self.classifier:
            self.classifier.eval()
        class_stats: Dict[int, Dict] = {}
        pool_mapping = self.defense_pool_by_label
        confusion_matrix: Dict[int, Dict[int, int]] = {}
        if num_classes is None:
            if self.classifier and hasattr(self.classifier, 'nb_classes'):
                num_classes = self.classifier.nb_classes
            else:
                num_classes = 10
        batch_packet_data = []
        with torch.no_grad():
            for flow_batch in test_batches:
                packet_flows_map: Dict[int, List[Dict]] = {}
                for flow in flow_batch:
                    packet_id = flow.get('packet_id', -1)
                    if packet_id >= 0:
                        packet_flows_map.setdefault(packet_id, []).append(flow)
                for packet_id, packet_flows in packet_flows_map.items():
                    flow_sequence_matrix = torch.zeros((self.max_flows, self.pad_length, 1), dtype=torch.float32, device=self.device)
                    flow_count = 0
                    all_processed_flows = []
                    total_padding = 0.0
                    total_injected = 0.0
                    total_original = 0.0
                    total_defense_flows_count = 0
                    time_windows = self._group_flows_by_time_window(packet_flows, self.concurrent_time_threshold)
                    for window_flows in time_windows:
                        window_start_count = flow_count
                        window_features_list = []
                        window_update_indices = []
                        for raw_flow in window_flows:
                            flow = self._clone_flow(raw_flow)
                            if flow_count < self.max_flows:
                                feat = self.extract_flow_features(flow)
                                window_features_list.append(feat)
                                window_update_indices.append(flow_count)
                                flow_count += 1
                        if window_features_list:
                            features_np = np.stack(window_features_list)
                            features_tensor = torch.from_numpy(features_np).to(self.device)
                            start_idx = window_update_indices[0]
                            end_idx = window_update_indices[-1] + 1
                            flow_sequence_matrix[start_idx:end_idx] = features_tensor
                        current_state = self.transformer(flow_sequence_matrix.unsqueeze(0), torch.tensor([flow_count], device=self.device)).squeeze(0)
                        for flow_idx_in_window, raw_flow in enumerate(window_flows):
                            action_dict = self.select_action(current_state, deterministic=True)
                            flow = self._clone_flow(raw_flow)
                            padding_bytes, injected_bytes, defense_flows = self._apply_action_to_flow(flow, action_dict, pool_mapping)
                            matrix_idx = window_start_count + flow_idx_in_window
                            if matrix_idx < self.max_flows:
                                updated_features = self.extract_flow_features(flow)
                                flow_sequence_matrix[matrix_idx] = torch.from_numpy(updated_features).to(self.device)
                            all_processed_flows.append(flow)
                            total_padding += padding_bytes
                            total_injected += injected_bytes
                            total_original += float(np.sum(np.abs(flow.get('packet_length', []))))
                            total_defense_flows_count += len(defense_flows)
                            defense_feats_list = []
                            defense_indices = []
                            for defense_flow in defense_flows:
                                all_processed_flows.append(defense_flow)
                                if flow_count < self.max_flows:
                                    defense_features = self.extract_flow_features(defense_flow)
                                    defense_feats_list.append(defense_features)
                                    defense_indices.append(flow_count)
                                    flow_count += 1
                            if defense_feats_list:
                                d_feats_np = np.stack(defense_feats_list)
                                d_feats_tensor = torch.from_numpy(d_feats_np).to(self.device)
                                start_d = defense_indices[0]
                                end_d = defense_indices[-1] + 1
                                flow_sequence_matrix[start_d:end_d] = d_feats_tensor
                    if all_processed_flows:
                        batch_packet_data.append({'processed_flows': all_processed_flows, 'true_label': int(packet_flows[0].get('label', -1)), 'total_original': total_original, 'total_added': total_padding + total_injected, 'total_injected': total_injected, 'total_defense_flows': total_defense_flows_count})
            if self.classifier and batch_packet_data:
                graphs = []
                valid_indices = []
                for i, p_data in enumerate(batch_packet_data):
                    self.reset_graph_builder()
                    for proc_flow in p_data['processed_flows']:
                        self.graph_builder.step(proc_flow)
                    classifier_graph = self.graph_builder.build_classifier_graph(classifier_type=self.classifier_type)
                    if classifier_graph and classifier_graph.num_nodes() > 0:
                        graphs.append(classifier_graph.to(self.device))
                        valid_indices.append(i)
                if graphs:
                    graph_batch = dgl.batch(graphs)
                    logits = self.classifier(graph_batch)
                    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
                    for batch_idx, global_idx in enumerate(valid_indices):
                        p_data = batch_packet_data[global_idx]
                        true_label = p_data['true_label']
                        predicted_label = predicted_labels[batch_idx]
                        if true_label not in confusion_matrix:
                            confusion_matrix[true_label] = {}
                        if predicted_label not in confusion_matrix[true_label]:
                            confusion_matrix[true_label][predicted_label] = 0
                        confusion_matrix[true_label][predicted_label] += 1
                        if true_label not in class_stats:
                            class_stats[true_label] = {'correct': 0, 'total': 0, 'total_original_bytes': 0.0, 'total_added_bytes': 0.0, 'total_injected_bytes': 0.0, 'total_defense_flows': 0}
                        class_stats[true_label]['total'] += 1
                        if predicted_label == true_label:
                            class_stats[true_label]['correct'] += 1
                        class_stats[true_label]['total_original_bytes'] += p_data['total_original']
                        class_stats[true_label]['total_added_bytes'] += p_data['total_added']
                        class_stats[true_label]['total_injected_bytes'] += p_data.get('total_injected', 0.0)
                        class_stats[true_label]['total_defense_flows'] += p_data.get('total_defense_flows', 0)
        total_correct = sum((s['correct'] for s in class_stats.values()))
        total_samples = sum((s['total'] for s in class_stats.values()))
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        total_original = sum((s['total_original_bytes'] for s in class_stats.values()))
        total_added = sum((s['total_added_bytes'] for s in class_stats.values()))
        total_injected = sum((s.get('total_injected_bytes', 0.0) for s in class_stats.values()))
        total_defense_flows = sum((s.get('total_defense_flows', 0) for s in class_stats.values()))
        avg_overhead = total_added / total_original if total_original > 0 else 0.0
        avg_defense_flows_per_packet = total_defense_flows / total_samples if total_samples > 0 else 0.0
        defense_flow_overhead_ratio = total_injected / total_added if total_added > 0 else 0.0
        all_classes = set()
        for true_label in confusion_matrix.keys():
            all_classes.add(true_label)
            for pred_label in confusion_matrix[true_label].keys():
                all_classes.add(pred_label)
        class_details = {}
        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            total = stats['total']
            correct = stats['correct']
            tp = confusion_matrix.get(label, {}).get(label, 0)
            fn = sum((confusion_matrix.get(label, {}).get(other_label, 0) for other_label in all_classes if other_label != label))
            fp = sum((confusion_matrix.get(true_label, {}).get(label, 0) for true_label in all_classes if true_label != label))
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            cost_ratio = stats['total_added_bytes'] / stats['total_original_bytes'] if stats['total_original_bytes'] > 0 else 0.0
            class_details[label] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'cost_ratio': cost_ratio, 'correct': stats['correct'], 'total': stats['total'], 'original_bytes': stats['total_original_bytes'], 'defense_bytes': stats['total_added_bytes']}
        return {'overall_accuracy': overall_accuracy, 'avg_cost_ratio': avg_overhead, 'class_details': class_details, 'total_samples': total_samples, 'avg_defense_flows_per_packet': avg_defense_flows_per_packet, 'defense_flow_overhead_ratio': defense_flow_overhead_ratio, 'total_defense_flows': total_defense_flows, 'total_injected_bytes': total_injected}
