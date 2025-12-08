import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import torch


def _pad_sequence(values: Optional[np.ndarray], pad_length: int) -> np.ndarray:
    """Pad or truncate a 1-D sequence to pad_length."""
    padded = np.zeros(pad_length, dtype=np.float32)
    if values is None:
        return padded
    array = np.asarray(values, dtype=np.float32).flatten()
    length = min(len(array), pad_length)
    padded[:length] = array[:length]
    return padded


def _packet_to_bursts(packet_lengths: np.ndarray, pad_length: int) -> np.ndarray:
    """
    Convert packet length sequence to burst count sequence.
    Each burst is the count of consecutive same-direction packets.
    """
    bursts: List[float] = []
    if packet_lengths is None or len(packet_lengths) == 0:
        return np.zeros(pad_length, dtype=np.float32)

    seq = np.asarray(packet_lengths, dtype=np.float32)
    current_sign = math.copysign(1.0, seq[0]) if seq[0] != 0 else 1.0
    packet_count = 0
    for value in seq:
        if value == 0:  # 遇到0，保存当前 burst 并结束
            if packet_count > 0:
                bursts.append(float(packet_count))
            break
        sign = math.copysign(1.0, value)
        if sign == current_sign:
            packet_count += 1  # 同方向，计数+1
        else:
            bursts.append(float(packet_count))  # 方向改变，保存当前 burst 的包个数
            current_sign = sign
            packet_count = 1  # 重新开始计数
    
    # 保存最后一个 burst
    if packet_count > 0:
        bursts.append(float(packet_count))

    padded = np.zeros(pad_length, dtype=np.float32)
    length = min(len(bursts), pad_length)
    padded[:length] = bursts[:length]
    return padded


@dataclass
class FlowNode:
    node_id: int
    packet_length: np.ndarray
    arrive_time_delta: np.ndarray
    burst: np.ndarray
    start_timestamp: float
    label: int
    meta: Dict = field(default_factory=dict)
    step_index: int = 0
    burst_id: int = 0


class IncrementalGraphBuilder:
    """
    Incrementally builds graph representations as new flows arrive.

    Two graph views are generated per step:
        1. For SAC agent (includes burst features)
        2. For substitute classifier (original FG-net format)
    """

    def __init__(
        self,
        pad_length: int = 1000,
        mtu: float = 1500.0,
        mtime: float = 10.0,
        concurrent_time_threshold: float = 1.0,
        time_window: Optional[float] = None,
    ):
        self.pad_length = pad_length
        self.mtu = mtu
        self.mtime = mtime
        self.concurrent_time_threshold = concurrent_time_threshold
        self.time_window = time_window
        self.nodes: List[FlowNode] = []
        self.node_counter = 0

    def reset(self):
        self.nodes = []
        self.node_counter = 0

    def step(self, flow: Dict, step_index: int = 0) -> Tuple[Optional[dgl.DGLGraph], Optional[dgl.DGLGraph], Optional[FlowNode]]:
        """
        Update builder with a new flow and return updated graphs.
        Args:
            flow: Dict containing packet_length, arrive_time_delta, start_timestamp, label, etc.
            step_index: Logical time index of this flow inside the batch.
        Returns:
            agent_graph: Graph with burst features for SAC.
            classifier_graph: Graph for substitute classifier.
            FlowNode: metadata of the inserted node.
        """
        packet_length = flow.get("packet_length")
        arrive_time_delta = flow.get("arrive_time_delta")
        start_timestamp = float(flow.get("start_timestamp", step_index))
        label = int(flow.get("label", -1))

        pkt_feature = _pad_sequence(packet_length, self.pad_length)
        arv_feature = _pad_sequence(arrive_time_delta, self.pad_length)
        burst_feature = _packet_to_bursts(packet_length, self.pad_length)

        node = FlowNode(
            node_id=self.node_counter,
            packet_length=pkt_feature,
            arrive_time_delta=arv_feature,
            burst=burst_feature,
            start_timestamp=start_timestamp,
            label=label,
            meta={key: flow[key] for key in flow if key not in {"packet_length", "arrive_time_delta"}},
            step_index=step_index,
        )
        self.node_counter += 1

        if self.time_window is not None:
            self._expire_old_flows(current_time=start_timestamp, current_step=step_index)
        self.nodes.append(node)

        if not self.nodes:
            return None, None, None

        agent_graph = self._build_graph(include_burst=True)
        classifier_graph = self._build_graph(include_burst=False)
        return agent_graph, classifier_graph, node

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _expire_old_flows(self, current_time: float, current_step: int):
        """Keep only flows within the configured sliding window."""
        filtered: List[FlowNode] = []
        for node in self.nodes:
            time_gap = current_time - node.start_timestamp
            step_gap = current_step - node.step_index
            if time_gap <= self.time_window and step_gap >= 0:
                filtered.append(node)
        self.nodes = filtered

    def _build_graph(self, include_burst: bool) -> dgl.DGLGraph:
        num_nodes = len(self.nodes)
        bursts, node_to_burst = self._group_nodes_into_bursts()
        edge_mode = "agent" if include_burst else "classifier"
        src, dst = self._build_edges(bursts, edge_mode)

        graph = dgl.graph((src, dst), num_nodes=num_nodes)

        pkt_matrix = np.stack([node.packet_length for node in self.nodes], axis=0)
        arv_matrix = np.stack([node.arrive_time_delta for node in self.nodes], axis=0)

        pkt_tensor = torch.from_numpy(pkt_matrix).unsqueeze(1) / self.mtu
        arv_tensor = torch.from_numpy(arv_matrix).unsqueeze(1) / self.mtime

        if num_nodes > 0:
            graph.ndata["pkt_length"] = pkt_tensor
            graph.ndata["arv_time"] = arv_tensor
            graph.ndata["label"] = torch.tensor([node.label for node in self.nodes], dtype=torch.long).unsqueeze(-1)
            graph.ndata["step_index"] = torch.tensor([node.step_index for node in self.nodes], dtype=torch.long).unsqueeze(-1)
            graph.ndata["start_ts"] = torch.tensor([node.start_timestamp for node in self.nodes], dtype=torch.float32).unsqueeze(-1)
            graph.ndata["burst_id"] = torch.tensor(node_to_burst, dtype=torch.long).unsqueeze(-1)

        if include_burst and num_nodes > 0:
            burst_matrix = np.stack([node.burst for node in self.nodes], axis=0)
            graph.ndata["burst"] = torch.from_numpy(burst_matrix).unsqueeze(1) / self.mtu

        return graph

    def _group_nodes_into_bursts(self) -> Tuple[List[List[int]], List[int]]:
        bursts: List[List[int]] = []
        node_to_burst: List[int] = [0] * len(self.nodes)

        current_burst: List[int] = []
        for idx, node in enumerate(self.nodes):
            if not current_burst:
                current_burst.append(idx)
                continue
            last_idx = current_burst[-1]
            time_gap = abs(node.start_timestamp - self.nodes[last_idx].start_timestamp)
            if time_gap < self.concurrent_time_threshold:
                current_burst.append(idx)
            else:
                bursts.append(current_burst)
                current_burst = [idx]

        if current_burst:
            bursts.append(current_burst)

        for burst_id, burst in enumerate(bursts):
            for node_idx in burst:
                node_to_burst[node_idx] = burst_id

        return bursts, node_to_burst

    def _build_edges(self, bursts: List[List[int]], mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.nodes) <= 1:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)

        src: List[int] = []
        dst: List[int] = []

        if mode == "classifier":
            for burst in bursts:
                for j in range(len(burst) - 1):
                    src.append(burst[j])
                    dst.append(burst[j + 1])

            for b_idx in range(1, len(bursts)):
                prev_burst = bursts[b_idx - 1]
                curr_burst = bursts[b_idx]
                if prev_burst and curr_burst:
                    src.append(prev_burst[-1])
                    dst.append(curr_burst[0])
                    if len(curr_burst) > 1:
                        src.append(prev_burst[-1])
                        dst.append(curr_burst[-1])
        else:
            for burst in bursts:
                for j in range(len(burst) - 1):
                    src.append(burst[j])
                    dst.append(burst[j + 1])
                    src.append(burst[j + 1])
                    dst.append(burst[j])

            for b_idx in range(1, len(bursts)):
                prev_burst = bursts[b_idx - 1]
                curr_burst = bursts[b_idx]
                if prev_burst and curr_burst:
                    src.append(prev_burst[-1])
                    dst.append(curr_burst[0])
                    src.append(curr_burst[0])
                    dst.append(prev_burst[-1])
                    if len(curr_burst) > 1:
                        src.append(prev_burst[-1])
                        dst.append(curr_burst[-1])
                        src.append(curr_burst[-1])
                        dst.append(prev_burst[-1])

        if not src:
            return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        return torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)

