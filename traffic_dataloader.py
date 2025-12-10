import numpy as np
from typing import Tuple, Dict, Union
import random


def load_spilt_dataset(
    data_path: str,
    split_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    加载 npz 数据集并按照指定比例划分训练集和测试集。
    
    Args:
        data_path: npz 文件路径
        split_ratio: 测试集比例（默认 0.2，即占 20%）
        random_seed: 随机种子，用于保证划分的可重复性
    
    Returns:
        train_dataset: 训练集字典，包含 packet_length, arrive_time_delta, 
                      start_timestamp, end_timestamp, labels
        test_dataset: 测试集字典，结构同训练集
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 加载 npz 文件
    data = np.load(data_path, allow_pickle=True)
    
    # 提取所有字段
    packet_length = data['packet_length']
    arrive_time_delta = data['arrive_time_delta']
    start_timestamp = data['start_timestamp']
    end_timestamp = data['end_timestamp']
    labels = data['labels']
    
    # 获取数据总数
    num_samples = len(labels)
    
    # 计算划分点
    test_size = int(num_samples * split_ratio)
    train_size = num_samples - test_size
    
    # 生成随机索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # 划分索引
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 构建训练集
    train_dataset = {
        'packet_length': packet_length[train_indices],
        'arrive_time_delta': arrive_time_delta[train_indices],
        'start_timestamp': start_timestamp[train_indices],
        'end_timestamp': end_timestamp[train_indices],
        'labels': labels[train_indices]
    }
    
    # 构建测试集
    test_dataset = {
        'packet_length': packet_length[test_indices],
        'arrive_time_delta': arrive_time_delta[test_indices],
        'start_timestamp': start_timestamp[test_indices],
        'end_timestamp': end_timestamp[test_indices],
        'labels': labels[test_indices]
    }
    
    print(f"数据集划分完成:")
    print(f"  训练集: {train_size} 样本 ({train_size/num_samples*100:.2f}%)")
    print(f"  测试集: {test_size} 样本 ({test_size/num_samples*100:.2f}%)")
    
    return train_dataset, test_dataset


def get_dataset_statistics(dataset: Dict) -> Union[str, Dict]:
    """
    对数据集进行统计分析并输出统计信息。
    
    Args:
        dataset: 数据集字典，包含 packet_length, arrive_time_delta, 
                start_timestamp, end_timestamp, labels
    
    Returns:
        包含统计信息的字典，也可以打印统计信息
    """
    if dataset is None or len(dataset) == 0:
        return "数据集为空"
    
    labels = dataset['labels']
    packet_length = dataset['packet_length']
    arrive_time_delta = dataset['arrive_time_delta']
    start_timestamp = dataset.get('start_timestamp', None)
    end_timestamp = dataset.get('end_timestamp', None)
    
    num_samples = len(labels)
    
    # 统计类别分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)
    
    # 统计 packet_length
    # 对于每个样本，packet_length 可能是一个数组，需要统计每个样本的长度
    packet_length_stats = []
    total_packets = 0
    for pl in packet_length:
        if isinstance(pl, np.ndarray) or isinstance(pl, list):
            sample_length = len(pl)
            packet_length_stats.append(sample_length)
            total_packets += sample_length
        else:
            packet_length_stats.append(1)
            total_packets += 1
    
    packet_length_stats = np.array(packet_length_stats)
    
    # 统计 arrive_time_delta
    # 对于每个样本，arrive_time_delta 可能是一个数组，需要统计总的时间跨度
    time_delta_stats = []
    for atd in arrive_time_delta:
        if isinstance(atd, np.ndarray) or isinstance(atd, list):
            if len(atd) > 0:
                time_delta_stats.append(np.sum(atd))
            else:
                time_delta_stats.append(0)
        else:
            time_delta_stats.append(atd if atd > 0 else 0)
    
    time_delta_stats = np.array(time_delta_stats)
    
    # 构建统计信息字典
    stats_dict = {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'class_distribution': {int(label): int(count) for label, count in zip(unique_labels, label_counts)},
        'packet_length_stats': {
            'mean': float(np.mean(packet_length_stats)),
            'std': float(np.std(packet_length_stats)),
            'min': int(np.min(packet_length_stats)),
            'max': int(np.max(packet_length_stats)),
            'median': float(np.median(packet_length_stats)),
            'total_packets': int(total_packets)
        },
        'time_delta_stats': {
            'mean': float(np.mean(time_delta_stats[time_delta_stats > 0])) if np.any(time_delta_stats > 0) else 0.0,
            'std': float(np.std(time_delta_stats[time_delta_stats > 0])) if np.any(time_delta_stats > 0) else 0.0,
            'min': float(np.min(time_delta_stats[time_delta_stats > 0])) if np.any(time_delta_stats > 0) else 0.0,
            'max': float(np.max(time_delta_stats)),
            'median': float(np.median(time_delta_stats[time_delta_stats > 0])) if np.any(time_delta_stats > 0) else 0.0,
            'total_time': float(np.sum(time_delta_stats))
        }
    }
    
    # 如果有时间戳信息，添加时间范围统计
    if start_timestamp is not None and end_timestamp is not None:
        duration = end_timestamp - start_timestamp
        stats_dict['timestamp_stats'] = {
            'start_min': float(np.min(start_timestamp)),
            'start_max': float(np.max(start_timestamp)),
            'end_min': float(np.min(end_timestamp)),
            'end_max': float(np.max(end_timestamp)),
            'duration_mean': float(np.mean(duration)),
            'duration_std': float(np.std(duration)),
            'duration_min': float(np.min(duration)),
            'duration_max': float(np.max(duration))
        }
    
    # 打印统计信息
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    print(f"样本总数: {num_samples}")
    print(f"类别总数: {num_classes}")
    print(f"\n类别分布:")
    for label, count in zip(unique_labels, label_counts):
        percentage = count / num_samples * 100
        print(f"  类别 {int(label)}: {int(count)} 样本 ({percentage:.2f}%)")
    
    print(f"\n数据包长度统计 (每个样本的数据包数量):")
    print(f"  均值: {stats_dict['packet_length_stats']['mean']:.2f}")
    print(f"  标准差: {stats_dict['packet_length_stats']['std']:.2f}")
    print(f"  最小值: {stats_dict['packet_length_stats']['min']}")
    print(f"  最大值: {stats_dict['packet_length_stats']['max']}")
    print(f"  中位数: {stats_dict['packet_length_stats']['median']:.2f}")
    print(f"  总数据包数: {stats_dict['packet_length_stats']['total_packets']}")
    
    print(f"\n时间间隔统计 (每个样本的总时间跨度):")
    print(f"  均值: {stats_dict['time_delta_stats']['mean']:.4f}")
    print(f"  标准差: {stats_dict['time_delta_stats']['std']:.4f}")
    print(f"  最小值: {stats_dict['time_delta_stats']['min']:.4f}")
    print(f"  最大值: {stats_dict['time_delta_stats']['max']:.4f}")
    print(f"  中位数: {stats_dict['time_delta_stats']['median']:.4f}")
    print(f"  总时间: {stats_dict['time_delta_stats']['total_time']:.4f}")
    
    if 'timestamp_stats' in stats_dict:
        print(f"\n时间戳统计:")
        print(f"  开始时间范围: [{stats_dict['timestamp_stats']['start_min']:.2f}, {stats_dict['timestamp_stats']['start_max']:.2f}]")
        print(f"  结束时间范围: [{stats_dict['timestamp_stats']['end_min']:.2f}, {stats_dict['timestamp_stats']['end_max']:.2f}]")
        print(f"  样本持续时间统计:")
        print(f"    均值: {stats_dict['timestamp_stats']['duration_mean']:.4f}")
        print(f"    标准差: {stats_dict['timestamp_stats']['duration_std']:.4f}")
        print(f"    最小值: {stats_dict['timestamp_stats']['duration_min']:.4f}")
        print(f"    最大值: {stats_dict['timestamp_stats']['duration_max']:.4f}")
    
    print("="*60 + "\n")
    
    return stats_dict

