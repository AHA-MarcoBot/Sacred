import json
import  numpy as np
import copy
import torch
import matplotlib.pylab as plt
from torch import nn
from torch import optim
from torch.nn import functional as F
import tqdm
import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List
from traffic_dataloader import load_spilt_dataset, get_dataset_statistics
from graph_builder import IncrementalGraphBuilder
from graph_traffic_obfuscator import FlowTrafficObfuscator

def ensure_directories():
    """
    确保项目所需的日志、可视化、模型等目录存在，如果不存在则创建。
    """
    # 定义需要创建的路径
    paths_to_create = [
        "./logs/tb",  # TensorBoard 日志
        "./logs/vis",  # 可视化图像或其他输出
        "./logs",  # 日志根目录（即使上面两个存在，也确保父目录存在）
        "./saved_models",  # 模型保存路径
    ]

    for path_str in paths_to_create:
        path = Path(path_str)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {path.resolve()}")
        else:
            print(f"📁 目录已存在: {path.resolve()}")
            pass


DEFAULT_REWARD_C = {
    "dummy_c": 20,           # burst padding（在当前 burst 尾部插入随机 dummy 包）的权重（使用平方惩罚）
    "morphing_c": 20,       # 插入陌生站点流量的惩罚权重（使用平方惩罚）
    "base_c": 10,            # 基础奖励系数，控制对数概率项的影响力
}

DEFAULT_ARGS = {
    "data_path": './sacred_dataset/fgnet_dataset_d1.npz',
    "defense_data_path": './sacred_dataset/random_website_50x10_dataset.npz',
    "dataset_split_ratio": 0.05,
    "fgnet_model_path": './fgnet_state_dict.pth',
    "fgnet_layer_type": 'GAT',
    # SAC 训练参数
    "episodes": 10,
    "hidden_size": 1024,
    "gamma": 0.99,
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "batch_size": 16,
    "target_update_interval": 1,
    "replay_size": 1000000,
    "seed": 42,
    "max_dummy": 10,
    "state_size": 128,
    "graph_encoder_latent_dim": 128,
    "reward_type": 'o',
    "pad_length": 1000,
    "concurrent_time_threshold": 1.0,
    "defense_site_classes": 10,
    "max_flows_per_action": 5,
    "min_dummy_packet_size": 200,
    "max_dummy_packet_size": 800,
    "label_min": 0,
    "label_max": 10,
    **DEFAULT_REWARD_C,
}


def add_sac_args(parser, defaults=None):
    # ==SAC ARGUMENTS
    if defaults is None:
        defaults = DEFAULT_ARGS
    parser.add_argument('--data_path', type=str, default=defaults['data_path'],
                        help='Path to the npz data file')
    parser.add_argument('--defense_data_path', type=str, default=defaults['defense_data_path'],
                        help='Path to the random website pool npz file')
    parser.add_argument('--dataset_split_ratio', type=float, default=defaults['dataset_split_ratio'], )
    parser.add_argument('--fgnet_model_path', type=str, default=defaults['fgnet_model_path'],
                        help='Path to the converted FG-net state_dict (.pth)')
    parser.add_argument('--fgnet_layer_type', type=str, default=defaults['fgnet_layer_type'],
                        choices=['GCN', 'GAT'], help='Layer type for FG-net model (GCN or GAT)')
    parser.add_argument('--label_min', type=int, default=defaults['label_min'],
                        help='Minimum label id (inclusive) to train SAC on')
    parser.add_argument('--label_max', type=int, default=defaults['label_max'],
                        help='Maximum label id (inclusive) to train SAC on')
    
    # SAC 训练参数
    parser.add_argument('--episodes', type=int, default=defaults['episodes'],
                        help='Number of training episodes')
    parser.add_argument('--hidden_size', type=int, default=defaults['hidden_size'], metavar='N',
                        help='hidden size')
    parser.add_argument('--gamma', type=float, default=defaults['gamma'], metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=defaults['tau'], metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=defaults['lr'], metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=defaults['alpha'], metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'],
                        help='Training batch size')
    parser.add_argument('--target_update_interval', type=int, default=defaults['target_update_interval'], metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=defaults['replay_size'], metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--seed', type=int, default=defaults['seed'], help='Training seed')
    parser.add_argument('--max_dummy', type=int, default=defaults['max_dummy'],
                        help='Maximum number of dummy packets')
    parser.add_argument('--state_size', type=int, default=defaults['state_size'],
                        help='State dimension (graph encoder output dimension)')
    parser.add_argument('--graph_encoder_latent_dim', type=int, default=defaults['graph_encoder_latent_dim'],
                        help='Graph encoder latent dimension')
    parser.add_argument('--reward_type', type=str, default=defaults['reward_type'],
                        help='Reward type')
    parser.add_argument('--pad_length', type=int, default=defaults['pad_length'],
                        help='Padding length for packet/time sequences')
    parser.add_argument('--concurrent_time_threshold', type=float, default=defaults['concurrent_time_threshold'],
                        help='Max time difference (seconds) to connect flows concurrently')
    parser.add_argument('--defense_site_classes', type=int, default=defaults['defense_site_classes'],
                        help='Number of defense website classes considered by SAC')
    parser.add_argument('--max_flows_per_action', type=int, default=defaults['max_flows_per_action'],
                        help='Maximum number of defense flows that can be inserted per action (default: 5)')
    parser.add_argument('--min_dummy_packet_size', type=int, default=defaults['min_dummy_packet_size'],
                        help='Minimum dummy packet size in bytes')
    parser.add_argument('--max_dummy_packet_size', type=int, default=defaults['max_dummy_packet_size'],
                        help='Maximum dummy packet size in bytes')
    
    # Reward Coefficient
    for key in DEFAULT_REWARD_C:
        parser.add_argument(f'--{key}', type=float, default=defaults[key])
    
    return parser


def load_npz_flows(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    packet_length = data['packet_length']
    arrive_time_delta = data['arrive_time_delta']
    start_timestamp = data['start_timestamp']
    end_timestamp = data['end_timestamp']
    labels = data['labels']

    flows = []
    for idx in range(packet_length.shape[0]):
        flow = {
            "packet_length": packet_length[idx].astype(int).tolist(),
            "arrive_time_delta": arrive_time_delta[idx].astype(float).tolist(),
            "start_timestamp": float(start_timestamp[idx]),
            "end_timestamp": float(end_timestamp[idx]),
            "label": int(labels[idx]),
        }
        flows.append(flow)
    return flows


def dataset_to_packets(dataset: Dict) -> List[List[Dict]]:
    """
    将数据集转换为数据包列表，每个数据包包含多条流。
    labels 为 shape (N,) 的包级标签。
    """
    packet_length = dataset['packet_length']
    arrive_time_delta = dataset['arrive_time_delta']
    start_timestamp = dataset['start_timestamp']
    end_timestamp = dataset['end_timestamp']
    labels = dataset['labels']

    num_packets = packet_length.shape[0]
    num_flows_per_packet = packet_length.shape[1]
    
    packets = []
    
    for pkt_idx in range(num_packets):
        packet_flows = []
        packet_label = int(labels[pkt_idx])
        for flow_idx in range(num_flows_per_packet):
            pl = packet_length[pkt_idx, flow_idx]
            ad = arrive_time_delta[pkt_idx, flow_idx]
            start_ts = float(start_timestamp[pkt_idx, flow_idx])
            end_ts = float(end_timestamp[pkt_idx, flow_idx])
            
            # 全零（包长与时间序列同时为零）视为数据包结束，直接停止
            if np.all(pl == 0) and np.all(ad == 0):
                break
            
            flow = {
                "packet_length": pl.astype(int).tolist(),
                "arrive_time_delta": ad.astype(float).tolist(),
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "label": packet_label,
                "packet_id": pkt_idx,
            }
            packet_flows.append(flow)
        
        # 只添加包含至少一条有效流的数据包
        if len(packet_flows) > 0:
            packets.append(packet_flows)
    
    return packets


def group_packets_by_label(packets: List[List[Dict]]) -> Dict[int, List[int]]:
    """
    按数据包的主要标签分组。
    """
    label_to_indices: Dict[int, List[int]] = {}
    for pkt_idx, packet_flows in enumerate(packets):
        if len(packet_flows) == 0:
            continue
        label = int(packet_flows[0].get("label", -1))
        if label >= 0:
            label_to_indices.setdefault(label, []).append(pkt_idx)
    return label_to_indices


def packets_from_indices(packets: List[List[Dict]], indices: List[int]) -> List[List[Dict]]:
    """
    根据索引列表提取对应数据包。
    """
    result = []
    for idx in indices:
        if 0 <= idx < len(packets):
            result.append(packets[idx])
    return result


def build_index_batches(n_samples: int, batch_size: int, shuffle: bool = True):
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def load_fgnet_model(model, checkpoint_path, optimizer=None):
    """
    加载 FG-net 模型权重。
    
    Args:
        model: 模型实例
        optimizer: 优化器实例（此函数中未使用，但为保持兼容性保留）
        checkpoint_path: 模型检查点目录路径
    
    Returns:
        加载的模型对象
    """
    ckpt_path = Path(checkpoint_path).expanduser()
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "fgnet_state_dict.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"FG-net checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        layer_states = payload.get("layer_state_dicts")
    elif isinstance(payload, dict):
        state_dict = payload
        layer_states = None
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    model.load_state_dict(state_dict)
    if layer_states and len(layer_states) == len(model.layers):
        for layer, layer_state in zip(model.layers, layer_states):
            layer.load_state_dict(layer_state)
    logging.info(f"Loaded FG-net state_dict from {ckpt_path}")
    return model

def main():
    ensure_directories()
    logging.basicConfig(filename='./logs/myapp.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Sacred Program')
    parser = add_sac_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading data from {args.data_path}")
    train_dataset, test_dataset, vaildation_dataset = load_spilt_dataset(data_path=args.data_path,
                                                    split_ratio=args.dataset_split_ratio)

    labels_flat = np.array(train_dataset['labels']).astype(int)
    unique_labels = np.unique(labels_flat)
    num_classes = len(unique_labels)
    logging.info(f"{num_classes} unique classes in training set")

    train_dataset_stats = get_dataset_statistics(train_dataset)
    logging.info(f"Training Dataset stats: {train_dataset_stats}")

    # Load pre-trained FG-net model
    try:
        import spapp_classifier
        
        logging.info(f"Loading FG-net model from {args.fgnet_model_path}")
        logging.info(f"Model layer type: {args.fgnet_layer_type}")
        
        # 确定设备类型（适配 train.py 的格式）
        use_gpu = torch.cuda.is_available() and str(device) != "cpu"
        if use_gpu:
            device_id = str(device)  # e.g., "cuda:0"
        else:
            device_id = "cpu"
        
        # 初始化模型（在加载权重之前创建）
        fgnet_model = spapp_classifier.App_Classifier(
            nb_classes=num_classes,
            use_gpu=use_gpu,
            device=device_id,
            layer_type=args.fgnet_layer_type
        )
        
        # 加载模型权重（load_fgnet_model 函数会返回完整的模型对象）
        fgnet_model = load_fgnet_model(fgnet_model, optimizer=None, checkpoint_path=args.fgnet_model_path)
        
        # 检查模型类别数是否与数据集匹配
        model_num_classes = fgnet_model.nb_classes
        if model_num_classes != num_classes:
            logging.warning(f"Model number of classes ({model_num_classes}) does not match dataset number of classes ({num_classes})")
            logging.warning(f"Using model's number of classes: {model_num_classes}")
        
        # 确保模型在正确的设备上
        if use_gpu:
            device_obj = torch.device(device_id)
            fgnet_model = fgnet_model.to(device_obj)
            fgnet_model.device = device_id
        else:
            fgnet_model = fgnet_model.to(torch.device("cpu"))
            fgnet_model.device = "cpu"
        
        # 设置为评估模式
        fgnet_model.eval()
        
        logging.info(f"FG-net model loaded successfully")
        logging.info(f"Model device: {device}")
        logging.info(f"Model number of classes: {fgnet_model.nb_classes}")
        
    except ImportError as e:
        logging.error(f"Failed to import FG-net modules: {e}")
        logging.error("Make sure fgnet-main/source_code/fgnet_code exists and is accessible")
        fgnet_model = None
    except Exception as e:
        logging.error(f"Failed to load FG-net model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        fgnet_model = None
    
    # 检查 FG-net 模型是否加载成功
    if fgnet_model is None:
        logging.error("FG-net model is required but failed to load. Exiting.")
        return
    
    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 准备奖励系数字典
    reward_c = {}
    for key in DEFAULT_REWARD_C:
        reward_c[key] = vars(args)[key]
    logging.info(f"Reward coefficients: {reward_c}")
    
    logging.info("Preparing traffic packets for training and evaluation...")
    train_packets = dataset_to_packets(train_dataset)
    test_packets = dataset_to_packets(test_dataset)
    defense_flows = load_npz_flows(args.defense_data_path)
    logging.info(f"Loaded {len(train_packets)} training packets from split dataset")
    logging.info(f"Loaded {len(test_packets)} test packets from split dataset")
    logging.info(f"Loaded {len(defense_flows)} defense flows from {args.defense_data_path}")
    # 统计每个数据包内的流数
    train_flows_per_packet = [len(pkt) for pkt in train_packets[:10]]
    logging.info(f"Sample flows per packet (first 10): {train_flows_per_packet}")
    train_label_map = group_packets_by_label(train_packets)
    test_label_map = group_packets_by_label(test_packets)

    selected_labels = [
        label for label in sorted(train_label_map.keys())
        if args.label_min <= label <= args.label_max
    ]

    if not selected_labels:
        logging.error(
            f"No training data found within label range [{args.label_min}, {args.label_max}]."
        )
        return

    logging.info(f"Selected labels for training: {selected_labels}")

    for label in selected_labels:
        label_train_indices = train_label_map.get(label, [])
        if not label_train_indices:
            logging.warning(f"Label {label} has no training packets; skipping.")
            continue

        label_train_packets = packets_from_indices(train_packets, label_train_indices)
        label_test_packets = packets_from_indices(test_packets, test_label_map.get(label, []))

        logging.info("=" * 80)
        logging.info(f"Starting SAC training for label {label} with {len(label_train_packets)} packets")
        logging.info("=" * 80)

        graph_builder = IncrementalGraphBuilder(
            pad_length=args.pad_length,
            concurrent_time_threshold=args.concurrent_time_threshold,
        )

        obfuscator = FlowTrafficObfuscator(
            max_flows=50,  # 最大流数量
            max_dummy_packets=args.max_dummy,
            num_classes=1,  # 单标签训练
            device=device,
            reward_c=reward_c,
            hidden_dim=args.hidden_size,
            buffer_size=args.replay_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            lr=args.lr,
            alpha=args.alpha,
            target_update_interval=args.target_update_interval,
            reward_type=args.reward_type,
            transformer_output_dim=args.graph_encoder_latent_dim,
            incremental_builder=graph_builder,
            defense_site_classes=args.defense_site_classes,
            max_flows_per_action=args.max_flows_per_action,
            pad_length=args.pad_length,
            min_dummy_packet_size=args.min_dummy_packet_size,
            max_dummy_packet_size=args.max_dummy_packet_size,
            concurrent_time_threshold=args.concurrent_time_threshold,
        )
        obfuscator.set_classifier(fgnet_model)
        obfuscator.set_defense_pool(defense_flows)

        for episode in range(1, args.episodes + 1):
            episode_stats = {
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

            batch_iterator = build_index_batches(len(label_train_packets), args.batch_size, shuffle=True)
            total_batches = max(1, (len(label_train_packets) + args.batch_size - 1) // args.batch_size)
            progress = tqdm.tqdm(
                batch_iterator,
                total=total_batches,
                desc=f"Label {label} - Episode {episode}",
                unit="batch",
            )

            valid_batches = 0
            for batch_indices in progress:
                batch_flows = []
                for pkt_idx in batch_indices:
                    if pkt_idx < len(label_train_packets):
                        packet_flows = label_train_packets[pkt_idx]
                        for flow in packet_flows:
                            if len(flow.get("packet_length", [])) > 0:
                                batch_flows.append(copy.deepcopy(flow))
                if not batch_flows:
                    continue
                stats = obfuscator.train_on_batches([batch_flows])
                total_reward += stats["avg_reward"] * stats["total_steps"]
                episode_stats["total_steps"] += stats["total_steps"]
                episode_stats["correct_predictions"] += stats["correct_predictions"]
                episode_stats["total_padding_bytes"] += stats.get("total_padding_bytes", 0.0)
                episode_stats["total_injected_bytes"] += stats.get("total_injected_bytes", 0.0)
                episode_stats["total_original_bytes"] += stats.get("total_original_bytes", 0.0)
                episode_stats["total_added_bytes"] += stats.get("total_added_bytes", 0.0)
                episode_stats["total_defense_flows"] += stats.get("total_defense_flows", 0)
                episode_stats["packet_count"] += stats.get("packet_count", 0)
                valid_batches += 1

            progress.close()
            if valid_batches == 0 or episode_stats["total_steps"] == 0:
                logging.warning(
                    f"[Label {label}] No valid batches (with non-empty flows) in episode {episode}; skipping."
                )
                continue

            episode_stats["avg_reward"] = total_reward / episode_stats["total_steps"]
            correct_ratio = episode_stats["correct_predictions"] / episode_stats["total_steps"]
            avg_padding = episode_stats["total_padding_bytes"] / episode_stats["total_steps"]
            avg_injected = episode_stats["total_injected_bytes"] / episode_stats["total_steps"]
            defense_overhead = (
                episode_stats["total_added_bytes"] / max(episode_stats["total_original_bytes"], 1e-6)
                if episode_stats["total_original_bytes"] > 0 else 0.0
            )
            avg_defense_flows = (
                episode_stats["total_defense_flows"] / episode_stats["packet_count"]
                if episode_stats["packet_count"] > 0 else 0.0
            )
            evaluation_acc = None
            if label_test_packets:
                sample_count = min(args.batch_size, len(label_test_packets))
                sample_indices = random.sample(range(len(label_test_packets)), sample_count)
                eval_batch = []
                for pkt_idx in sample_indices:
                    packet_flows = label_test_packets[pkt_idx]
                    for flow in packet_flows:
                        if len(flow.get("packet_length", [])) > 0:
                            eval_batch.append(copy.deepcopy(flow))
                if eval_batch:
                    eval_snapshot = obfuscator.evaluate_on_test_data([eval_batch])
                    evaluation_acc = eval_snapshot.get("overall_accuracy", 0.0)

            logging.info(
                f"[Label {label}] [Episode {episode}/{args.episodes}] "
                f"Steps: {episode_stats['total_steps']}, "
                f"AvgReward: {episode_stats['avg_reward']:.4f}, "
                f"Classifier Acc: {correct_ratio:.4f}, "
                f"AvgPaddingBytes: {avg_padding:.2f}, "
                f"AvgInjectedBytes: {avg_injected:.2f}, "
                f"OverheadRatio: {defense_overhead:.4f}, "
                f"AvgDefenseFlowsPerPacket: {avg_defense_flows:.2f}, "
                f"TestBatchAcc: {(evaluation_acc if evaluation_acc is not None else float('nan')):.4f}"
            )

        model_save_path = f'./saved_models/graph-sac-label{label}-ep{args.episodes}.pth'
        obfuscator.save_checkpoint(model_save_path)
        logging.info(f"[Label {label}] Training completed. Model saved to {model_save_path}")

        # 只对当前标签的测试数据进行评估
        if not label_test_packets:
            logging.warning(f"[Label {label}] No test packets available; skipping evaluation.")
            continue

        logging.info(f"[Label {label}] Starting evaluation on {len(label_test_packets)} test packets")
        test_batches = []
        for batch_idx in build_index_batches(len(label_test_packets), args.batch_size, shuffle=False):
            batch_flows = []
            for pkt_idx in batch_idx:
                if pkt_idx < len(label_test_packets):
                    packet_flows = label_test_packets[pkt_idx]
                    for flow in packet_flows:
                        if len(flow.get("packet_length", [])) > 0:
                            batch_flows.append(copy.deepcopy(flow))
            if batch_flows:
                test_batches.append(batch_flows)

        eval_results = obfuscator.evaluate_on_test_data(test_batches)

        logging.info("=" * 80)
        logging.info(f"[Label {label}] EVALUATION RESULTS")
        logging.info("=" * 80)
        logging.info(f"Total test samples: {eval_results['total_samples']}")
        logging.info(
            f"Overall accuracy: {eval_results['overall_accuracy']:.4f} "
            f"({eval_results['overall_accuracy']*100:.2f}%)"
        )
        logging.info(
            f"Overall average cost ratio: {eval_results['avg_cost_ratio']:.4f} "
            f"({eval_results['avg_cost_ratio']*100:.2f}% overhead)"
        )

        sorted_classes = sorted(eval_results['class_details'].keys())
        for class_id in sorted_classes:
            details = eval_results['class_details'][class_id]
            logging.info(
                f"{class_id:<10} "
                f"{details['accuracy']:<15.4f} "
                f"{details['cost_ratio']:<15.4f} "
                f"{details['correct']}/{details['total']:<18} "
                f"{details['original_bytes']:<20.2f} "
                f"{details['defense_bytes']:<20.2f}"
            )

        eval_output_path = f'./logs/eval_results_label{label}_ep{args.episodes}.json'
        os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
        with open(eval_output_path, 'w') as f:
            json.dump({
                'label': int(label),
                'overall_accuracy': float(eval_results['overall_accuracy']),
                'overall_avg_cost': float(eval_results['avg_cost_ratio']),
                'total_samples': int(eval_results['total_samples']),
                'per_class_accuracy': {str(k): float(v['accuracy']) for k, v in eval_results['class_details'].items()},
                'per_class_cost': {str(k): float(v['cost_ratio']) for k, v in eval_results['class_details'].items()},
                'class_details': {
                    str(k): {
                        'accuracy': float(v['accuracy']),
                        'cost_ratio': float(v['cost_ratio']),
                        'correct': int(v['correct']),
                        'total': int(v['total']),
                        'original_bytes': float(v['original_bytes']),
                        'defense_bytes': float(v['defense_bytes']),
                    }
                    for k, v in eval_results['class_details'].items()
                }
            }, f, indent=2)
        logging.info(f"[Label {label}] Evaluation results saved to {eval_output_path}")

    
if __name__ == "__main__":
    main()