import json
import numpy as np
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
    paths_to_create = ['./logs/tb', './logs/vis', './logs', './saved_models']
    for path_str in paths_to_create:
        path = Path(path_str)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f'✅ 创建目录: {path.resolve()}')
        else:
            print(f'📁 目录已存在: {path.resolve()}')
            pass
DEFAULT_REWARD_C = {'dummy_c': 10000, 'morphing_c': 10000, 'base_c': 15000}
DEFAULT_ARGS = {'data_path': './sacred_dataset/fgnet_dataset_d1.npz', 'defense_data_path': './sacred_dataset/random_website_50x10_dataset.npz', 'dataset_split_ratio': 0.25, 'fgnet_model_path': './fgnet_state_dict.pth', 'fgnet_layer_type': 'GAT', 'classifier_type': 'fgnet', 'transgraphnet_model_path': './Transgraphnet/transgraphnet_from_npz_state_dict.pth', 'df_model_path': './DF/df_model_state_dict.pth', 'eval_classifiers': 'default', 'episodes': 15, 'hidden_size': 1024, 'gamma': 0.95, 'tau': 0.005, 'lr': 0.0001, 'alpha': 0.2, 'batch_size': 32, 'target_update_interval': 2, 'replay_size': 100000, 'seed': 42, 'max_dummy': 10, 'state_size': 128, 'graph_encoder_latent_dim': 128, 'reward_type': 'o', 'pad_length': 1000, 'concurrent_time_threshold': 1.0, 'defense_site_classes': 10, 'min_dummy_packet_size': 200, 'max_dummy_packet_size': 800, 'label_min': 21, 'label_max': 30, 'max_bursts': 50, 'ablation_no_positional_encoding': False, 'ablation_no_window_attention': False, 'ablation_no_flow_inter_attention': False, 'ablation_no_burst_padding': False, 'ablation_no_defense_flow': False, **DEFAULT_REWARD_C}

def add_sac_args(parser, defaults=None):
    if defaults is None:
        defaults = DEFAULT_ARGS
    parser.add_argument('--data_path', type=str, default=defaults['data_path'], help='Path to the npz data file')
    parser.add_argument('--defense_data_path', type=str, default=defaults['defense_data_path'], help='Path to the random website pool npz file')
    parser.add_argument('--dataset_split_ratio', type=float, default=defaults['dataset_split_ratio'])
    parser.add_argument('--fgnet_model_path', type=str, default=defaults['fgnet_model_path'], help='Path to the converted FG-net state_dict (.pth)')
    parser.add_argument('--fgnet_layer_type', type=str, default=defaults['fgnet_layer_type'], choices=['GCN', 'GAT'], help='Layer type for FG-net model (GCN or GAT)')
    parser.add_argument('--classifier_type', type=str, default=defaults['classifier_type'], choices=['fgnet', 'transgraphnet', 'df'], help='Classifier type: fgnet, transgraphnet, or df')
    parser.add_argument('--transgraphnet_model_path', type=str, default=defaults['transgraphnet_model_path'], help='Path to the TransGraphNet state_dict (.pth)')
    parser.add_argument('--df_model_path', type=str, default=defaults['df_model_path'], help='Path to the DF model state_dict (.pth)')
    parser.add_argument('--eval_classifiers', type=str, default=defaults['eval_classifiers'], help='Comma-separated list of classifier types for evaluation (e.g., "fgnet,transgraphnet,df"), or "default" to use only classifier_type')
    parser.add_argument('--label_min', type=int, default=defaults['label_min'], help='Minimum label id (inclusive) to train SAC on')
    parser.add_argument('--label_max', type=int, default=defaults['label_max'], help='Maximum label id (inclusive) to train SAC on')
    parser.add_argument('--episodes', type=int, default=defaults['episodes'], help='Number of training episodes')
    parser.add_argument('--hidden_size', type=int, default=defaults['hidden_size'], metavar='N', help='hidden size')
    parser.add_argument('--gamma', type=float, default=defaults['gamma'], metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=defaults['tau'], metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=defaults['lr'], metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=defaults['alpha'], metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help='Training batch size')
    parser.add_argument('--target_update_interval', type=int, default=defaults['target_update_interval'], metavar='N', help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=defaults['replay_size'], metavar='N', help='size of replay buffer (default: 1000000)')
    parser.add_argument('--seed', type=int, default=defaults['seed'], help='Training seed')
    parser.add_argument('--max_dummy', type=int, default=defaults['max_dummy'], help='Maximum number of dummy packets')
    parser.add_argument('--state_size', type=int, default=defaults['state_size'], help='State dimension (graph encoder output dimension)')
    parser.add_argument('--graph_encoder_latent_dim', type=int, default=defaults['graph_encoder_latent_dim'], help='Graph encoder latent dimension')
    parser.add_argument('--reward_type', type=str, default=defaults['reward_type'], help='Reward type')
    parser.add_argument('--pad_length', type=int, default=defaults['pad_length'], help='Padding length for packet/time sequences')
    parser.add_argument('--concurrent_time_threshold', type=float, default=defaults['concurrent_time_threshold'], help='Max time difference (seconds) to connect flows concurrently')
    parser.add_argument('--defense_site_classes', type=int, default=defaults['defense_site_classes'], help='Number of defense website classes considered by SAC (action space dimension)')
    parser.add_argument('--min_dummy_packet_size', type=int, default=defaults['min_dummy_packet_size'], help='Minimum dummy packet size in bytes')
    parser.add_argument('--max_dummy_packet_size', type=int, default=defaults['max_dummy_packet_size'], help='Maximum dummy packet size in bytes')
    parser.add_argument('--max_bursts', type=int, default=defaults['max_bursts'], help='Maximum number of bursts to generate padding actions for')
    parser.add_argument('--ablation_no_positional_encoding', action='store_true', default=defaults['ablation_no_positional_encoding'], help='Ablation: Disable positional encoding')
    parser.add_argument('--ablation_no_window_attention', action='store_true', default=defaults['ablation_no_window_attention'], help='Ablation: Disable window attention')
    parser.add_argument('--ablation_no_flow_inter_attention', action='store_true', default=defaults['ablation_no_flow_inter_attention'], help='Ablation: Disable flow inter-attention')
    parser.add_argument('--ablation_no_burst_padding', action='store_true', default=defaults['ablation_no_burst_padding'], help='Ablation: Disable burst padding')
    parser.add_argument('--ablation_no_defense_flow', action='store_true', default=defaults['ablation_no_defense_flow'], help='Ablation: Disable defense flow injection')
    for key in DEFAULT_REWARD_C:
        parser.add_argument(f'--{key}', type=float, default=defaults[key])
    return parser

def load_classifier_model(classifier_type: str, args, num_classes: int, device, train_dataset=None):
    classifier_model = None
    if classifier_type == 'fgnet':
        try:
            import spapp_classifier
            logging.info(f'Loading FG-net model from {args.fgnet_model_path}')
            logging.info(f'Model layer type: {args.fgnet_layer_type}')
            use_gpu = torch.cuda.is_available() and str(device) != 'cpu'
            if use_gpu:
                device_id = str(device)
            else:
                device_id = 'cpu'
            classifier_model = spapp_classifier.App_Classifier(nb_classes=num_classes, use_gpu=use_gpu, device=device_id, layer_type=args.fgnet_layer_type)
            classifier_model = load_fgnet_model(classifier_model, optimizer=None, checkpoint_path=args.fgnet_model_path)
            model_num_classes = classifier_model.nb_classes
            if model_num_classes != num_classes:
                logging.warning(f'Model number of classes ({model_num_classes}) does not match dataset number of classes ({num_classes})')
                logging.warning(f"Using model's number of classes: {model_num_classes}")
            if use_gpu:
                device_obj = torch.device(device_id)
                classifier_model = classifier_model.to(device_obj)
                classifier_model.device = device_id
            else:
                classifier_model = classifier_model.to(torch.device('cpu'))
                classifier_model.device = 'cpu'
            classifier_model.eval()
            logging.info(f'FG-net model loaded successfully')
            logging.info(f'Model device: {device}')
            logging.info(f'Model number of classes: {classifier_model.nb_classes}')
        except ImportError as e:
            logging.error(f'Failed to import FG-net modules: {e}')
            logging.error('Make sure spapp_classifier.py exists in the current directory')
            classifier_model = None
        except Exception as e:
            logging.error(f'Failed to load FG-net model: {e}')
            import traceback
            logging.error(traceback.format_exc())
            classifier_model = None
    elif classifier_type == 'transgraphnet':
        try:
            import sys
            import dgl
            sys.path.insert(0, './Transgraphnet')
            from TransGraphNet import Classifier
            logging.info(f'Loading TransGraphNet model from {args.transgraphnet_model_path}')
            use_gpu = torch.cuda.is_available() and str(device) != 'cpu'
            if use_gpu:
                device_id = str(device)
            else:
                device_id = 'cpu'
            device_obj = torch.device(device_id)
            classifier_model = Classifier(nb_classes=num_classes, latent_feature_length=256, use_gpu=use_gpu, device=device_id, layer_type='GAT', pad_length=args.pad_length, bert_dir='./Transgraphnet/bert-mini' if os.path.exists('./Transgraphnet/bert-mini') else 'bert-mini', use_bert_encoder=True, bert_num_layers=None, freeze_cnn=False)
            classifier_model = classifier_model.to(device_obj)
            classifier_model.eval()
            if train_dataset is not None:
                logging.info('Initializing TransGraphNet model with dummy input...')
                try:
                    dummy_graph_builder = IncrementalGraphBuilder(pad_length=args.pad_length, concurrent_time_threshold=args.concurrent_time_threshold)
                    train_packets_temp = dataset_to_packets(train_dataset)
                    dummy_graphs = []
                    for packet_idx in range(min(3, len(train_packets_temp))):
                        if packet_idx >= len(train_packets_temp):
                            break
                        packet_flows = train_packets_temp[packet_idx]
                        if not packet_flows:
                            continue
                        dummy_graph_builder.reset()
                        for flow_idx, flow in enumerate(packet_flows[:5]):
                            dummy_graph_builder.step(flow, step_index=flow_idx)
                        dummy_graph = dummy_graph_builder.build_classifier_graph(classifier_type='transgraphnet')
                        if dummy_graph is not None and dummy_graph.num_nodes() > 0:
                            dummy_graphs.append(dummy_graph.to(device_obj))
                    if len(dummy_graphs) > 0:
                        with torch.no_grad():
                            dummy_batched = dgl.batch(dummy_graphs)
                            _ = classifier_model(dummy_batched)
                        logging.info(f'Model initialization completed with {len(dummy_graphs)} dummy graphs')
                    else:
                        logging.warning('Could not create dummy graphs for initialization, proceeding anyway...')
                except Exception as init_error:
                    logging.warning(f'Failed to initialize model with dummy input: {init_error}')
                    logging.warning('Proceeding with model loading anyway, but may encounter shape mismatch errors...')
            classifier_model = load_transgraphnet_model(classifier_model, optimizer=None, checkpoint_path=args.transgraphnet_model_path)
            model_num_classes = classifier_model.nb_classes
            if model_num_classes != num_classes:
                logging.warning(f'Model number of classes ({model_num_classes}) does not match dataset number of classes ({num_classes})')
                logging.warning(f"Using model's number of classes: {model_num_classes}")
            classifier_model = classifier_model.to(device_obj)
            classifier_model.device = device_id
            classifier_model.eval()
            logging.info(f'TransGraphNet model loaded successfully')
            logging.info(f'Model device: {device_obj}')
            logging.info(f'Model number of classes: {classifier_model.nb_classes}')
            logging.info(f'Model pad_length: {args.pad_length}')
        except ImportError as e:
            logging.error(f'Failed to import TransGraphNet modules: {e}')
            logging.error('Make sure Transgraphnet/TransGraphNet.py exists and is accessible')
            classifier_model = None
        except Exception as e:
            logging.error(f'Failed to load TransGraphNet model: {e}')
            import traceback
            logging.error(traceback.format_exc())
            classifier_model = None
    elif classifier_type == 'df':
        try:
            import sys
            sys.path.insert(0, './DF')
            from df_classifier import DFClassifier
            logging.info(f'Loading DF model from {args.df_model_path}')
            use_gpu = torch.cuda.is_available() and str(device) != 'cpu'
            if use_gpu:
                device_id = str(device)
            else:
                device_id = 'cpu'
            device_obj = torch.device(device_id)
            checkpoint = torch.load(args.df_model_path, map_location=device_obj)
            model_num_classes = checkpoint.get('num_classes', num_classes)
            model_pad_length = checkpoint.get('pad_length', args.pad_length)
            model_mtu = checkpoint.get('mtu', 1500.0)
            classifier_model = DFClassifier(nb_classes=model_num_classes, pad_length=model_pad_length, mtu=model_mtu, use_gpu=use_gpu, device=device_id, aggregation_method='mean')
            classifier_model.load_state_dict(checkpoint, strict=False)
            classifier_model = classifier_model.to(device_obj)
            classifier_model.eval()
            if model_num_classes != num_classes:
                logging.warning(f'Model number of classes ({model_num_classes}) does not match dataset number of classes ({num_classes})')
                logging.warning(f"Using model's number of classes: {model_num_classes}")
            logging.info(f'DF model loaded successfully')
            logging.info(f'Model device: {device_obj}')
            logging.info(f'Model number of classes: {classifier_model.nb_classes}')
            logging.info(f'Model pad_length: {model_pad_length}')
            logging.info(f'Model MTU: {model_mtu}')
        except ImportError as e:
            logging.error(f'Failed to import DF modules: {e}')
            logging.error('Make sure DF/df_classifier.py exists and is accessible')
            classifier_model = None
        except Exception as e:
            logging.error(f'Failed to load DF model: {e}')
            import traceback
            logging.error(traceback.format_exc())
            classifier_model = None
    return classifier_model

def load_npz_flows(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    packet_length = data['packet_length']
    arrive_time_delta = data['arrive_time_delta']
    start_timestamp = data['start_timestamp']
    end_timestamp = data['end_timestamp']
    labels = data['labels']
    flows = []
    for idx in range(packet_length.shape[0]):
        flow = {'packet_length': packet_length[idx].astype(int).tolist(), 'arrive_time_delta': arrive_time_delta[idx].astype(float).tolist(), 'start_timestamp': float(start_timestamp[idx]), 'end_timestamp': float(end_timestamp[idx]), 'label': int(labels[idx])}
        flows.append(flow)
    return flows

def dataset_to_packets(dataset: Dict) -> List[List[Dict]]:
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
            if np.all(pl == 0) and np.all(ad == 0):
                break
            flow = {'packet_length': pl.astype(int).tolist(), 'arrive_time_delta': ad.astype(float).tolist(), 'start_timestamp': start_ts, 'end_timestamp': end_ts, 'label': packet_label, 'packet_id': pkt_idx}
            packet_flows.append(flow)
        if len(packet_flows) > 0:
            packets.append(packet_flows)
    return packets

def group_packets_by_label(packets: List[List[Dict]]) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = {}
    for pkt_idx, packet_flows in enumerate(packets):
        if len(packet_flows) == 0:
            continue
        label = int(packet_flows[0].get('label', -1))
        if label >= 0:
            label_to_indices.setdefault(label, []).append(pkt_idx)
    return label_to_indices

def packets_from_indices(packets: List[List[Dict]], indices: List[int]) -> List[List[Dict]]:
    result = []
    for idx in indices:
        if 0 <= idx < len(packets):
            result.append(packets[idx])
    return result

def build_index_batches(n_samples: int, batch_size: int, shuffle: bool=True):
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]

def load_fgnet_model(model, checkpoint_path, optimizer=None):
    ckpt_path = Path(checkpoint_path).expanduser()
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / 'fgnet_state_dict.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'FG-net checkpoint not found: {ckpt_path}')
    payload = torch.load(ckpt_path, map_location='cpu')
    if isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
        layer_states = payload.get('layer_state_dicts')
    elif isinstance(payload, dict):
        state_dict = payload
        layer_states = None
    else:
        raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')
    model.load_state_dict(state_dict)
    if layer_states and len(layer_states) == len(model.layers):
        for layer, layer_state in zip(model.layers, layer_states):
            layer.load_state_dict(layer_state)
    logging.info(f'Loaded FG-net state_dict from {ckpt_path}')
    return model

def load_transgraphnet_model(model, checkpoint_path, optimizer=None):
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f'TransGraphNet checkpoint not found: {ckpt_path}')
    payload = torch.load(ckpt_path, map_location='cpu')
    if isinstance(payload, dict) and 'state_dict' in payload:
        state_dict = payload['state_dict']
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')
    model.load_state_dict(state_dict, strict=False)
    logging.info(f'Loaded TransGraphNet state_dict from {ckpt_path}')
    return model

def main():
    ensure_directories()
    logging.basicConfig(filename='./logs/myapp.log', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Sacred Program')
    parser = add_sac_args(parser)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    logging.info(f'Loading data from {args.data_path}')
    train_dataset, test_dataset = load_spilt_dataset(data_path=args.data_path, split_ratio=args.dataset_split_ratio)
    labels_flat = np.array(train_dataset['labels']).astype(int)
    unique_labels = np.unique(labels_flat)
    num_classes = len(unique_labels)
    logging.info(f'{num_classes} unique classes in training set')
    train_dataset_stats = get_dataset_statistics(train_dataset)
    logging.info(f'Training Dataset stats: {train_dataset_stats}')
    classifier_type = args.classifier_type
    classifier_model = load_classifier_model(classifier_type, args, num_classes, device, train_dataset)
    if classifier_model is None:
        logging.error(f'{classifier_type.upper()} model is required but failed to load. Exiting.')
        return
    eval_classifiers_str = args.eval_classifiers.strip().lower()
    if eval_classifiers_str == 'default':
        eval_classifier_types = [classifier_type]
    else:
        eval_classifier_types = [c.strip() for c in eval_classifiers_str.split(',') if c.strip()]
        valid_types = {'fgnet', 'transgraphnet', 'df'}
        eval_classifier_types = [c for c in eval_classifier_types if c in valid_types]
        if not eval_classifier_types:
            logging.warning(f'No valid classifier types in eval_classifiers, using default: {classifier_type}')
            eval_classifier_types = [classifier_type]
    logging.info(f'Training classifier: {classifier_type}')
    logging.info(f'Evaluation classifiers: {eval_classifier_types}')
    eval_classifiers = {}
    for eval_type in eval_classifier_types:
        if eval_type == classifier_type:
            eval_classifiers[eval_type] = classifier_model
        else:
            logging.info(f'Loading evaluation classifier: {eval_type}')
            eval_model = load_classifier_model(eval_type, args, num_classes, device, train_dataset)
            if eval_model is not None:
                eval_classifiers[eval_type] = eval_model
            else:
                logging.warning(f'Failed to load evaluation classifier {eval_type}, skipping...')
    if not eval_classifiers:
        logging.error('No valid evaluation classifiers available. Exiting.')
        return
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    reward_c = {}
    for key in DEFAULT_REWARD_C:
        reward_c[key] = vars(args)[key]
    logging.info(f'Reward coefficients: {reward_c}')
    logging.info('Preparing traffic packets for training and evaluation...')
    train_packets = dataset_to_packets(train_dataset)
    test_packets = dataset_to_packets(test_dataset)
    defense_flows = load_npz_flows(args.defense_data_path)
    logging.info(f'Loaded {len(train_packets)} training packets from split dataset')
    logging.info(f'Loaded {len(test_packets)} test packets from split dataset')
    logging.info(f'Loaded {len(defense_flows)} defense flows from {args.defense_data_path}')
    train_label_map = group_packets_by_label(train_packets)
    test_label_map = group_packets_by_label(test_packets)
    selected_labels = [label for label in sorted(train_label_map.keys()) if args.label_min <= label <= args.label_max]
    if not selected_labels:
        logging.error(f'No training data found within label range [{args.label_min}, {args.label_max}].')
        return
    logging.info(f'Selected labels for training: {selected_labels}')
    all_label_eval_results = []
    for label in selected_labels:
        label_train_indices = train_label_map.get(label, [])
        if not label_train_indices:
            logging.warning(f'Label {label} has no training packets; skipping.')
            continue
        label_train_packets = packets_from_indices(train_packets, label_train_indices)
        label_test_packets = packets_from_indices(test_packets, test_label_map.get(label, []))
        logging.info('=' * 80)
        logging.info(f'Starting SAC training for label {label} with {len(label_train_packets)} packets')
        logging.info('=' * 80)
        graph_builder = IncrementalGraphBuilder(pad_length=args.pad_length, concurrent_time_threshold=args.concurrent_time_threshold)
        obfuscator = FlowTrafficObfuscator(max_flows=50, max_dummy_packets=args.max_dummy, num_classes=1, device=device, reward_c=reward_c, hidden_dim=args.hidden_size, buffer_size=args.replay_size, batch_size=args.batch_size, gamma=args.gamma, tau=args.tau, lr=args.lr, alpha=args.alpha, target_update_interval=args.target_update_interval, reward_type=args.reward_type, transformer_output_dim=args.graph_encoder_latent_dim, incremental_builder=graph_builder, defense_site_classes=args.defense_site_classes, pad_length=args.pad_length, min_dummy_packet_size=args.min_dummy_packet_size, max_dummy_packet_size=args.max_dummy_packet_size, ablation_no_positional_encoding=args.ablation_no_positional_encoding, ablation_no_window_attention=args.ablation_no_window_attention, ablation_no_flow_inter_attention=args.ablation_no_flow_inter_attention, ablation_no_burst_padding=args.ablation_no_burst_padding, ablation_no_defense_flow=args.ablation_no_defense_flow, concurrent_time_threshold=args.concurrent_time_threshold, max_bursts=args.max_bursts)
        obfuscator.set_classifier(classifier_model, classifier_type=classifier_type)
        obfuscator.set_defense_pool(defense_flows)
        best_avg_reward = -float('inf')
        for episode in range(1, args.episodes + 1):
            episode_stats = {'total_steps': 0, 'correct_predictions': 0, 'avg_reward': 0.0, 'total_padding_bytes': 0.0, 'total_injected_bytes': 0.0, 'total_original_bytes': 0.0, 'total_added_bytes': 0.0, 'total_defense_flows': 0, 'packet_count': 0}
            total_reward = 0.0
            batch_iterator = build_index_batches(len(label_train_packets), args.batch_size, shuffle=True)
            total_batches = max(1, (len(label_train_packets) + args.batch_size - 1) // args.batch_size)
            progress = tqdm.tqdm(batch_iterator, total=total_batches, desc=f'Label {label} - Episode {episode}', unit='batch')
            valid_batches = 0
            for batch_indices in progress:
                batch_flows = []
                for pkt_idx in batch_indices:
                    if pkt_idx < len(label_train_packets):
                        packet_flows = label_train_packets[pkt_idx]
                        for flow in packet_flows:
                            if len(flow.get('packet_length', [])) > 0:
                                batch_flows.append(flow.copy())
                if not batch_flows:
                    continue
                stats = obfuscator.train_on_batches([batch_flows])
                total_reward += stats['avg_reward'] * stats['total_steps']
                episode_stats['total_steps'] += stats['total_steps']
                episode_stats['correct_predictions'] += stats['correct_predictions']
                episode_stats['total_padding_bytes'] += stats.get('total_padding_bytes', 0.0)
                episode_stats['total_injected_bytes'] += stats.get('total_injected_bytes', 0.0)
                episode_stats['total_original_bytes'] += stats.get('total_original_bytes', 0.0)
                episode_stats['total_added_bytes'] += stats.get('total_added_bytes', 0.0)
                episode_stats['total_defense_flows'] += stats.get('total_defense_flows', 0)
                episode_stats['packet_count'] += stats.get('packet_count', 0)
                valid_batches += 1
            progress.close()
            if valid_batches == 0 or episode_stats['total_steps'] == 0:
                logging.warning(f'[Label {label}] No valid batches (with non-empty flows) in episode {episode}; skipping.')
                continue
            episode_stats['avg_reward'] = total_reward / episode_stats['total_steps']
            correct_ratio = episode_stats['correct_predictions'] / episode_stats['total_steps']
            avg_padding = episode_stats['total_padding_bytes'] / episode_stats['total_steps']
            avg_injected = episode_stats['total_injected_bytes'] / episode_stats['total_steps']
            defense_overhead = episode_stats['total_added_bytes'] / max(episode_stats['total_original_bytes'], 1e-06) if episode_stats['total_original_bytes'] > 0 else 0.0
            avg_defense_flows = episode_stats['total_defense_flows'] / episode_stats['packet_count'] if episode_stats['packet_count'] > 0 else 0.0
            evaluation_acc = None
            if label_test_packets:
                test_batches = []
                for batch_idx in build_index_batches(len(label_test_packets), args.batch_size, shuffle=False):
                    batch_flows = []
                    for pkt_idx in batch_idx:
                        if pkt_idx < len(label_test_packets):
                            packet_flows = label_test_packets[pkt_idx]
                            for flow in packet_flows:
                                if len(flow.get('packet_length', [])) > 0:
                                    batch_flows.append(flow.copy())
                    if batch_flows:
                        test_batches.append(batch_flows)
                if test_batches:
                    eval_snapshot = obfuscator.evaluate_on_test_data(test_batches)
                    evaluation_acc = eval_snapshot.get('overall_accuracy', 0.0)
            logging.info(f'[Label {label}] [Episode {episode}/{args.episodes}] Steps: {episode_stats['total_steps']}, AvgReward: {episode_stats['avg_reward']:.4f}, Classifier Acc: {correct_ratio:.4f}, AvgPaddingBytes: {avg_padding:.2f}, AvgInjectedBytes: {avg_injected:.2f}, OverheadRatio: {defense_overhead:.4f}, AvgDefenseFlowsPerPacket: {avg_defense_flows:.2f}, TestBatchAcc: {(evaluation_acc if evaluation_acc is not None else float('nan')):.4f}')
            if episode_stats['avg_reward'] > best_avg_reward:
                best_avg_reward = episode_stats['avg_reward']
                best_model_path = f'./saved_models/graph-sac-label{label}-best.pth'
                obfuscator.save_checkpoint(best_model_path)
        model_save_path = f'./saved_models/graph-sac-label{label}-ep{args.episodes}.pth'
        obfuscator.save_checkpoint(model_save_path)
        logging.info(f'[Label {label}] Training completed. Model saved to {model_save_path}')
        if not label_test_packets:
            logging.warning(f'[Label {label}] No test packets available; skipping evaluation.')
            continue
        logging.info(f'[Label {label}] Starting model selection evaluation on {len(label_test_packets)} test packets')
        test_batches = []
        for batch_idx in build_index_batches(len(label_test_packets), args.batch_size, shuffle=False):
            batch_flows = []
            for pkt_idx in batch_idx:
                if pkt_idx < len(label_test_packets):
                    packet_flows = label_test_packets[pkt_idx]
                    for flow in packet_flows:
                        if len(flow.get('packet_length', [])) > 0:
                            batch_flows.append(flow.copy())
            if batch_flows:
                test_batches.append(batch_flows)
        if not test_batches:
            logging.warning(f'[Label {label}] No valid test batches; skipping model selection.')
            continue
        logging.info(f'[Label {label}] Evaluating last episode model...')
        last_model_results = obfuscator.evaluate_on_test_data(test_batches, num_classes=num_classes)
        last_model_accuracy = last_model_results['overall_accuracy']
        best_model_accuracy = float('inf')
        best_model_path = None
        best_model_results = None
        if best_avg_reward > -float('inf'):
            best_model_path = f'./saved_models/graph-sac-label{label}-best.pth'
            if os.path.exists(best_model_path):
                obfuscator.load_checkpoint(best_model_path, evaluate=True)
                logging.info(f'[Label {label}] Evaluating best reward model...')
                best_model_results = obfuscator.evaluate_on_test_data(test_batches, num_classes=num_classes)
                best_model_accuracy = best_model_results['overall_accuracy']
            else:
                logging.warning(f'[Label {label}] Best reward model not found, using last model only')
                best_model_path = None
        if best_model_path and best_model_accuracy < last_model_accuracy:
            logging.info(f'[Label {label}] Best reward model selected (accuracy: {best_model_accuracy:.4f} < {last_model_accuracy:.4f})')
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
                logging.info(f'[Label {label}] Deleted last episode model: {model_save_path}')
            final_model_path = f'./saved_models/graph-sac-label{label}-best.pth'
            eval_results = best_model_results
        else:
            best_acc_str = f'{best_model_accuracy:.4f}' if best_model_path else 'inf'
            logging.info(f'[Label {label}] Last episode model selected (accuracy: {last_model_accuracy:.4f} <= {best_acc_str})')
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                logging.info(f'[Label {label}] Deleted best reward model: {best_model_path}')
            final_model_path = f'./saved_models/graph-sac-label{label}-best.pth'
            if os.path.exists(model_save_path):
                os.rename(model_save_path, final_model_path)
                logging.info(f'[Label {label}] Renamed last episode model to: {final_model_path}')
            eval_results = last_model_results
        all_eval_results = {}
        for eval_type, eval_model in eval_classifiers.items():
            logging.info('=' * 80)
            logging.info(f'[Label {label}] Evaluating with {eval_type.upper()} classifier...')
            logging.info('=' * 80)
            obfuscator.set_classifier(eval_model, classifier_type=eval_type)
            if os.path.exists(final_model_path):
                obfuscator.load_checkpoint(final_model_path, evaluate=True)
            eval_result = obfuscator.evaluate_on_test_data(test_batches, num_classes=num_classes)
            all_eval_results[eval_type] = eval_result
            logging.info(f'[{eval_type.upper()}] Total test samples: {eval_result['total_samples']}')
            logging.info(f'[{eval_type.upper()}] Overall accuracy: {eval_result['overall_accuracy']:.4f} ({eval_result['overall_accuracy'] * 100:.2f}%)')
            logging.info(f'[{eval_type.upper()}] Overall average cost ratio: {eval_result['avg_cost_ratio']:.4f} ({eval_result['avg_cost_ratio'] * 100:.2f}% overhead)')
            avg_defense_flows = eval_result.get('avg_defense_flows_per_packet', 0.0)
            defense_overhead_ratio = eval_result.get('defense_flow_overhead_ratio', 0.0)
            logging.info(f'[{eval_type.upper()}] AvgDefenseFlowsPerPacket: {avg_defense_flows:.2f}')
            logging.info(f'[{eval_type.upper()}] Defense flow overhead ratio: {defense_overhead_ratio:.4f} ({defense_overhead_ratio * 100:.2f}% of total overhead)')
        obfuscator.set_classifier(classifier_model, classifier_type=classifier_type)
        eval_results = all_eval_results.get(classifier_type, eval_results)
        logging.info('=' * 80)
        logging.info(f'[Label {label}] FINAL EVALUATION RESULTS SUMMARY (All Classifiers)')
        logging.info('=' * 80)
        logging.info(f'{'Classifier':<20} {'Accuracy':<15} {'Cost Ratio':<15}')
        logging.info('-' * 50)
        for eval_type, result in all_eval_results.items():
            logging.info(f'{eval_type:<20} {result['overall_accuracy']:<15.4f} {result['avg_cost_ratio']:<15.4f}')
        logging.info('=' * 80)
        logging.info(f'Total test samples: {eval_results['total_samples']}')
        logging.info(f'Overall accuracy: {eval_results['overall_accuracy']:.4f} ({eval_results['overall_accuracy'] * 100:.2f}%)')
        logging.info(f'Overall average cost ratio: {eval_results['avg_cost_ratio']:.4f} ({eval_results['avg_cost_ratio'] * 100:.2f}% overhead)')
        avg_defense_flows = eval_results.get('avg_defense_flows_per_packet', 0.0)
        defense_overhead_ratio = eval_results.get('defense_flow_overhead_ratio', 0.0)
        logging.info(f'AvgDefenseFlowsPerPacket: {avg_defense_flows:.2f}')
        logging.info(f'Defense flow overhead ratio: {defense_overhead_ratio:.4f} ({defense_overhead_ratio * 100:.2f}% of total overhead)')
        sorted_classes = sorted(eval_results['class_details'].keys())
        logging.info(f'\nPer-class metrics:')
        logging.info(f'{'Class':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Cost Ratio':<12}')
        logging.info('-' * 70)
        for class_id in sorted_classes:
            details = eval_results['class_details'][class_id]
            logging.info(f'{class_id:<10} {details['accuracy']:<12.4f} {details['precision']:<12.4f} {details['recall']:<12.4f} {details['f1']:<12.4f} {details['cost_ratio']:<12.4f}')
        all_label_eval_results.append({'label': label, 'overall_accuracy': eval_results['overall_accuracy'], 'avg_cost_ratio': eval_results['avg_cost_ratio'], 'total_samples': eval_results['total_samples'], 'class_details': eval_results['class_details'], 'avg_defense_flows_per_packet': eval_results.get('avg_defense_flows_per_packet', 0.0), 'defense_flow_overhead_ratio': eval_results.get('defense_flow_overhead_ratio', 0.0), 'total_defense_flows': eval_results.get('total_defense_flows', 0), 'total_injected_bytes': eval_results.get('total_injected_bytes', 0.0)})
    if all_label_eval_results:
        logging.info('=' * 80)
        logging.info('OVERALL STATISTICS (All Labels)')
        logging.info('=' * 80)
        total_samples_all = sum((r['total_samples'] for r in all_label_eval_results))
        total_correct_all = 0
        total_original_bytes_all = 0.0
        total_defense_bytes_all = 0.0
        total_injected_bytes_all = 0.0
        total_defense_flows_all = 0
        for result in all_label_eval_results:
            for class_id, details in result['class_details'].items():
                total_correct_all += details['correct']
                total_original_bytes_all += details['original_bytes']
                total_defense_bytes_all += details['defense_bytes']
            total_injected_bytes_all += result.get('total_injected_bytes', 0.0)
            total_defense_flows_all += result.get('total_defense_flows', 0)
        overall_success_rate = total_correct_all / total_samples_all if total_samples_all > 0 else 0.0
        overall_overhead_percentage = total_defense_bytes_all / total_original_bytes_all * 100.0 if total_original_bytes_all > 0 else 0.0
        avg_defense_flows_all = total_defense_flows_all / total_samples_all if total_samples_all > 0 else 0.0
        defense_flow_overhead_ratio_all = total_injected_bytes_all / total_defense_bytes_all if total_defense_bytes_all > 0 else 0.0
        all_class_metrics = []
        for result in all_label_eval_results:
            for class_id, details in result['class_details'].items():
                if details['total'] > 0:
                    all_class_metrics.append({'accuracy': details['accuracy'], 'precision': details['precision'], 'recall': details['recall'], 'f1': details['f1']})
        avg_accuracy = sum((m['accuracy'] for m in all_class_metrics)) / len(all_class_metrics) if all_class_metrics else 0.0
        avg_precision = sum((m['precision'] for m in all_class_metrics)) / len(all_class_metrics) if all_class_metrics else 0.0
        avg_recall = sum((m['recall'] for m in all_class_metrics)) / len(all_class_metrics) if all_class_metrics else 0.0
        avg_f1 = sum((m['f1'] for m in all_class_metrics)) / len(all_class_metrics) if all_class_metrics else 0.0
        logging.info(f'Total test samples across all labels: {total_samples_all}')
        logging.info(f'Total correct predictions: {total_correct_all}')
        logging.info(f'Overall success rate: {overall_success_rate:.4f} ({overall_success_rate * 100:.2f}%)')
        logging.info(f'Total original bytes: {total_original_bytes_all:.2f}')
        logging.info(f'Total defense bytes: {total_defense_bytes_all:.2f}')
        logging.info(f'Overall bandwidth overhead: {overall_overhead_percentage:.2f}%')
        logging.info(f'AvgDefenseFlowsPerPacket: {avg_defense_flows_all:.2f}')
        logging.info(f'Defense flow overhead ratio: {defense_flow_overhead_ratio_all:.4f} ({defense_flow_overhead_ratio_all * 100:.2f}% of total overhead)')
        logging.info('\nAverage metrics across all classes:')
        logging.info(f'  Average Accuracy:  {avg_accuracy:.4f}')
        logging.info(f'  Average Precision: {avg_precision:.4f}')
        logging.info(f'  Average Recall:    {avg_recall:.4f}')
        logging.info(f'  Average F1:        {avg_f1:.4f}')
        logging.info('\nPer-label statistics:')
        logging.info(f'{'Label':<10} {'Accuracy':<15} {'Cost Ratio':<15} {'Samples':<15}')
        logging.info('-' * 55)
        for result in all_label_eval_results:
            logging.info(f'{result['label']:<10} {result['overall_accuracy']:<15.4f} {result['avg_cost_ratio']:<15.4f} {result['total_samples']:<15}')
        logging.info('=' * 80)
        overall_stats_path = f'./logs/overall_stats_all_labels_ep{args.episodes}.json'
        os.makedirs(os.path.dirname(overall_stats_path), exist_ok=True)
        with open(overall_stats_path, 'w') as f:
            json.dump({'overall_success_rate': float(overall_success_rate), 'overall_success_rate_percentage': float(overall_success_rate * 100.0), 'overall_bandwidth_overhead_percentage': float(overall_overhead_percentage), 'total_samples': int(total_samples_all), 'total_correct': int(total_correct_all), 'total_original_bytes': float(total_original_bytes_all), 'total_defense_bytes': float(total_defense_bytes_all), 'average_metrics': {'accuracy': float(avg_accuracy), 'precision': float(avg_precision), 'recall': float(avg_recall), 'f1': float(avg_f1)}, 'per_label_results': [{'label': int(r['label']), 'overall_accuracy': float(r['overall_accuracy']), 'avg_cost_ratio': float(r['avg_cost_ratio']), 'total_samples': int(r['total_samples'])} for r in all_label_eval_results]}, f, indent=2)
        logging.info(f'Overall statistics saved to {overall_stats_path}')
if __name__ == '__main__':
    main()
