import json
import numpy as np
from pathlib import Path

TIME_PERIOD = 60.0  # 秒，与 FG-net 保持一致
MIN_FLOW_LEN = 10   # 至少保留 10 个时间间隔


def pad_or_truncate(seq, length=1000, pad_value=0):
    if len(seq) >= length:
        return seq[:length]
    return seq + [pad_value] * (length - len(seq))


def split_flows_by_time_period(flows, time_period):
    """按时间窗口切分（与 FG-net construct_graph 一致）"""
    if not flows:
        return []
    flows = sorted(flows, key=lambda x: x["start_ts"])
    packets = []
    current = []
    window_start = None
    for flow in flows:
        if not current:
            current = [flow]
            window_start = flow["start_ts"]
            continue
        if flow["start_ts"] - window_start < time_period:
            current.append(flow)
        else:
            packets.append(current)
            current = [flow]
            window_start = flow["start_ts"]
    if current:
        packets.append(current)
    return packets


def convert_json_to_npz(
    src_root,
    output_npz="dataset.npz",
    max_flows_per_packet=50,
    max_len=1000,
    report_path="missing_fields_report.txt",
    time_period=TIME_PERIOD,
):
    """
    将 src_root 下的 JSON 流量数据转换为 FG-net 兼容的 NPZ。

    npz 结构：
        packet_length      -> shape (N, max_flows_per_packet, max_len)
        arrive_time_delta  -> shape (N, max_flows_per_packet, max_len)
        start_timestamp    -> shape (N, max_flows_per_packet)
        end_timestamp      -> shape (N, max_flows_per_packet)
        labels             -> shape (N,)  每个数据包一个标签
    """

    required_fields = [
        "packet_length",
        "arrive_time_delta",
        "start_timestamp",
        "end_timestamp",
    ]

    src_root = Path(src_root)

    all_packets_pl = []
    all_packets_delta = []
    all_packets_start_ts = []
    all_packets_end_ts = []
    all_packet_labels = []

    missing_reports = []
    class_map = {}
    next_label = 0

    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_map[class_dir.name] = next_label
        label = next_label
        next_label += 1

        for json_file in class_dir.rglob("*.json"):
            try:
                with open(json_file, "r") as f:
                    json_list = json.load(f)
            except Exception:
                missing_reports.append(f"{json_file} 无法解析 JSON 文件")
                continue

            flow_entries = []
            for i, flow in enumerate(json_list):
                missing = [k for k in required_fields if k not in flow]
                if missing:
                    missing_reports.append(f"{json_file} flow[{i}] 缺字段: {missing}")
                    continue

                pl = flow["packet_length"]
                ad = flow["arrive_time_delta"]
                valid_len = min(len(pl), max_len)
                if valid_len < MIN_FLOW_LEN:
                    continue

                pl_fixed = np.array(pad_or_truncate(pl, max_len, pad_value=0), dtype=np.int32)
                ad_fixed = np.array(pad_or_truncate(ad, max_len, pad_value=0.0), dtype=np.float32)

                flow_entries.append(
                    {
                        "packet_length": pl_fixed,
                        "arrive_time_delta": ad_fixed,
                        "start_ts": float(flow["start_timestamp"]),
                        "end_ts": float(flow["end_timestamp"]),
                    }
                )

            if not flow_entries:
                missing_reports.append(f"{json_file} 没有满足要求的流")
                continue

            packet_groups = split_flows_by_time_period(flow_entries, time_period)
            for group_idx, group in enumerate(packet_groups):
                if not group:
                    continue

                packet_pl = np.zeros((max_flows_per_packet, max_len), dtype=np.int32)
                packet_delta = np.zeros((max_flows_per_packet, max_len), dtype=np.float32)
                packet_start_ts = np.zeros(max_flows_per_packet, dtype=np.float64)
                packet_end_ts = np.zeros(max_flows_per_packet, dtype=np.float64)

                for flow_idx, flow_entry in enumerate(group[:max_flows_per_packet]):
                    packet_pl[flow_idx] = flow_entry["packet_length"]
                    packet_delta[flow_idx] = flow_entry["arrive_time_delta"]
                    packet_start_ts[flow_idx] = flow_entry["start_ts"]
                    packet_end_ts[flow_idx] = flow_entry["end_ts"]

                all_packets_pl.append(packet_pl)
                all_packets_delta.append(packet_delta)
                all_packets_start_ts.append(packet_start_ts)
                all_packets_end_ts.append(packet_end_ts)
                all_packet_labels.append(label)

    if not all_packets_pl:
        raise RuntimeError("未生成任何数据包，请检查输入数据或过滤条件。")

    all_packets_pl = np.stack(all_packets_pl, axis=0)
    all_packets_delta = np.stack(all_packets_delta, axis=0)
    all_packets_start_ts = np.stack(all_packets_start_ts, axis=0)
    all_packets_end_ts = np.stack(all_packets_end_ts, axis=0)
    all_packet_labels = np.array(all_packet_labels, dtype=np.int32)

    np.savez(
        output_npz,
        packet_length=all_packets_pl,
        arrive_time_delta=all_packets_delta,
        start_timestamp=all_packets_start_ts,
        end_timestamp=all_packets_end_ts,
        labels=all_packet_labels,
    )

    with open(report_path, "w") as f:
        for line in missing_reports:
            f.write(line + "\n")

    print(f"\nNPZ 已生成: {output_npz}")
    print(f"数据包总数: {len(all_packet_labels)}")
    print(f"每个数据包保留流数: {max_flows_per_packet}")
    print(f"packet_length shape: {all_packets_pl.shape}")
    print(f"arrive_time_delta shape: {all_packets_delta.shape}")
    print(f"start_timestamp shape: {all_packets_start_ts.shape}")
    print(f"end_timestamp shape: {all_packets_end_ts.shape}")
    print(f"labels shape: {all_packet_labels.shape}")
    print(f"类别映射（class_map）: {class_map}")
    print(f"缺字段报告写入: {report_path}")


if __name__ == "__main__":
    convert_json_to_npz(
        src_root=r"./fgnet-main/dataset/D1",
        output_npz="./sacred_dataset/fgnet_dataset_d1.npz",
        max_flows_per_packet=50,
        max_len=1000,
        report_path="missing_fields_report.txt",
    )
