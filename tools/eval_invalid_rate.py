#!/usr/bin/env python3
"""
计算 Invalid ID Rate — T5 生成的 SID 不在有效集合中的比例。

用法：
    cd /data/gtx/project/code/TIGER_minilm
    conda activate tiger_t5

    # 单数据集
    python3 tools/eval_invalid_rate.py --dataset Sport

    # 全部三个
    python3 tools/eval_invalid_rate.py --dataset Sport,Toys,Beauty

不需要重新训练 T5，用已有的最佳 checkpoint 做 evaluation。
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add model dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from main import TIGER, evaluate, set_seed


def load_valid_sids(code_path: str, codebook_size: int = 256):
    """
    加载 SID codes → 构建有效的 tokenized SID 集合。

    raw SID [c1,c2,c3,c4] → tokenized [c1+1, c2+256+1, c3+512+1, c4+768+1]
    """
    data = np.load(code_path, allow_pickle=True)
    if data.dtype == np.object_:
        data = data.item()
        sid_list = list(data.values())
    else:
        sid_list = data.tolist() if data.ndim == 1 else data.tolist()

    valid_sids = set()
    for code in sid_list:
        offsets = tuple(c + i * codebook_size + 1 for i, c in enumerate(code))
        valid_sids.add(offsets)

    return valid_sids


def evaluate_with_invalid_rate(model, eval_loader, valid_sids, topk_list, beam_size, device):
    """
    标准 evaluate + Invalid ID rate 计算。

    Invalid@K = 在 top-K beam 中，生成的 SID 不对应任何真实 item 的比例
    """
    model.eval()

    recalls = {f'Recall@{k}': [] for k in topk_list}
    ndcgs = {f'NDCG@{k}': [] for k in topk_list}
    invalid_rates = {f'Invalid@{k}': [] for k in topk_list}

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # (B, beam_size, seq_len)

            # ---- Standard metrics ----
            pos_index = torch.zeros((preds.shape[0], beam_size), dtype=torch.bool)
            for i in range(preds.shape[0]):
                cur_label = labels[i].tolist()
                for j in range(beam_size):
                    cur_pred = preds[i, j].tolist()
                    if cur_pred == cur_label:
                        pos_index[i, j] = True
                        break

            for k in topk_list:
                recall = pos_index[:, :k].sum(dim=1).cpu().float().mean().item()
                ranks = torch.arange(1, beam_size + 1).to(pos_index.device)
                dcg = 1.0 / torch.log2(ranks + 1)
                dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
                ndcg = dcg[:, :k].sum(dim=1).cpu().float().mean().item()

                recalls[f'Recall@{k}'].append(recall)
                ndcgs[f'NDCG@{k}'].append(ndcg)

            # ---- Invalid ID rate ----
            for i in range(preds.shape[0]):
                for k in topk_list:
                    invalid_count = 0
                    for j in range(k):
                        cur_pred = tuple(preds[i, j].tolist())
                        if cur_pred not in valid_sids:
                            invalid_count += 1
                    invalid_rates[f'Invalid@{k}'].append(invalid_count / k)

    # Average
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    avg_invalid = {k: sum(v) / len(v) * 100 for k, v in invalid_rates.items()}  # in %

    return avg_recalls, avg_ndcgs, avg_invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name(s), comma-separated: Sport,Toys,Beauty')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--beam_size', type=int, default=30)
    parser.add_argument('--topk_list', type=str, default='5,10,20',
                        help='Comma-separated top-k values, e.g. 5,10,20,50,100')
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    args.topk_list = [int(k) for k in args.topk_list.split(',')]

    datasets = [d.strip() for d in args.dataset.split(',')]

    # 数据集 → checkpoint / code_path 映射
    CKPT_MAP = {
        'Sport': 'model/ckpt/tiger_ta_sid_768d_sport_seed2025.pth',
        'Toys': 'model/ckpt/tiger_ta_sid_768d_toys_seed2025.pth',
        'Beauty': 'model/ckpt/tiger_ta_sid_768d_beauty_seed2025.pth',
        'Beauty_baseline': 'model/ckpt/tiger_content768d_baseline_beauty_seed2025.pth',
    }
    CODE_MAP = {
        'Sport': 'data/Sport/Sport_ta_sid_768d_sid.npy',
        'Toys': 'data/Toys/Toys_ta_sid_768d_sid.npy',
        'Beauty': 'data/Beauty/Beauty_ta_sid_768d_sid.npy',
        'Beauty_baseline': 'data/Beauty/Beauty_content768d_baseline_sid.npy',
    }
    DATA_DIR_MAP = {
        'Sport': 'Sport',
        'Toys': 'Toys',
        'Beauty': 'Beauty',
        'Beauty_baseline': 'Beauty',
    }

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    for ds in datasets:
        print(f'\n{"="*60}')
        print(f'  {ds} — Invalid ID Rate Evaluation')
        print(f'{"="*60}')

        ckpt_path = CKPT_MAP.get(ds)
        code_path = CODE_MAP.get(ds)

        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f'  ❌ Checkpoint not found: {ckpt_path}')
            continue
        if not code_path or not os.path.exists(code_path):
            print(f'  ❌ SID codes not found: {code_path}')
            continue

        # Load valid SID set
        print(f'  Loading valid SIDs from {code_path} ...')
        valid_sids = load_valid_sids(code_path)
        print(f'    {len(valid_sids)} unique valid SIDs')

        # 确定数据目录名
        ds_dir = ds if ds not in DATA_DIR_MAP else DATA_DIR_MAP[ds]

        # Init model
        print(f'  Loading T5 checkpoint: {ckpt_path} ...')
        config = {
            'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128,
            'd_ff': 1024, 'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
            'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
            'feed_forward_proj': 'relu', 'device': args.device,
            'max_len': 20, 'beam_size': args.beam_size,
            'topk_list': args.topk_list,
            'batch_size': args.batch_size, 'infer_size': 96,
            'num_epochs': 200, 'lr': 1e-4, 'seed': args.seed,
            'dataset_path': f'data/{ds_dir}',
            'code_path': code_path,
            'save_path': ckpt_path,
            'log_path': f'model/logs/{ds_dir}_r50_eval.log',
            'mode': 'evaluation',
            'early_stop': 10,
        }

        set_seed(args.seed)
        model = TIGER(config).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f'    Model loaded ✅')

        # Data
        test_dataset = GenRecDataset(
            dataset_path=f'data/{ds_dir}' + '/test.parquet',
            code_path=code_path,
            mode='evaluation',
            max_len=config['max_len']
        )
        test_loader = GenRecDataLoader(
            test_dataset, batch_size=config['infer_size'], shuffle=False
        )
        print(f'    Test samples: {len(test_dataset)}')

        # Evaluate
        print(f'  Running evaluation (beam_size={args.beam_size})...')
        recalls, ndcgs, invalid = evaluate_with_invalid_rate(
            model, test_loader, valid_sids,
            args.topk_list, args.beam_size, device
        )

        # Results — 动态打印所有 topk 值
        print(f'\n  📊 {ds} — Results:')
        k_str = ' '.join(f'{f"@{k}":<10}' for k in args.topk_list)
        print(f'  {"Metric":<15} {k_str}')
        print(f'  {"-" * (15 + 11 * len(args.topk_list))}')
        ndcg_str = ' '.join(f'{ndcgs[f"NDCG@{k}"]:<10.4f}' for k in args.topk_list)
        rec_str = ' '.join(f'{recalls[f"Recall@{k}"]:<10.4f}' for k in args.topk_list)
        inv_str = ' '.join(f'{invalid[f"Invalid@{k}"]:<10.2f}' for k in args.topk_list)
        print(f'  {"NDCG":<15} {ndcg_str}')
        print(f'  {"Recall":<15} {rec_str}')
        print(f'  {"Invalid%":<15} {inv_str}')


if __name__ == '__main__':
    main()
