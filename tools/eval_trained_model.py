#!/usr/bin/env python3
"""
eval_trained_model.py — 评估已训练的 T5 模型，打印 NDCG/Recall

用法：
    python3 tools/eval_trained_model.py \\
        --dataset Sport \\
        --ckpt_path model/ckpt/tiger_random_ablation_sport.pth \\
        --code_path data/Sport/Sport_random_ablation_sid.npy \\
        --beam_size 30 \\
        --topk_list 5,10,20

适用于加载已有 checkpoint 做快速评估。
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from main import TIGER, evaluate, set_seed
from dataset import GenRecDataset
from dataloader import GenRecDataLoader

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset directory name (Sport/Toys/Beauty)')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to T5 checkpoint .pth')
    parser.add_argument('--code_path', type=str, required=True,
                        help='Path to SID codes .npy')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--infer_size', type=int, default=96)
    parser.add_argument('--beam_size', type=int, default=30)
    parser.add_argument('--topk_list', type=str, default='5,10,20',
                        help='Comma-separated, e.g. 5,10,20')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root data directory')
    args = parser.parse_args()

    topk_list = [int(k) for k in args.topk_list.split(',')]
    data_path = f'{args.data_dir}/{args.dataset}'
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    # Load model
    print(f'Loading model from {args.ckpt_path}')
    config = {
        'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128,
        'd_ff': 1024, 'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1,
        'vocab_size': 1025, 'pad_token_id': 0, 'eos_token_id': 0,
        'feed_forward_proj': 'relu', 'device': args.device,
        'max_len': 20, 'beam_size': args.beam_size,
        'topk_list': topk_list,
        'batch_size': args.batch_size, 'infer_size': args.infer_size,
        'num_epochs': 200, 'lr': 1e-4, 'seed': args.seed,
        'dataset_path': data_path,
        'code_path': args.code_path,
        'save_path': args.ckpt_path,
        'log_path': 'tmp_eval.log',
        'mode': 'evaluation',
        'early_stop': 10,
    }

    model = TIGER(config).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()
    print(f'Model loaded ✅')

    # Data
    test_dataset = GenRecDataset(
        dataset_path=f'{data_path}/test.parquet',
        code_path=args.code_path,
        mode='evaluation',
        max_len=config['max_len']
    )
    test_loader = GenRecDataLoader(
        test_dataset, batch_size=config['infer_size'], shuffle=False
    )
    print(f'Test samples: {len(test_dataset)}')

    # Evaluate
    print(f'Evaluating (beam_size={args.beam_size}, topk={topk_list})...')
    recalls, ndcgs = evaluate(model, test_loader, topk_list, args.beam_size, device)

    # Print results
    k_str = ' '.join(f'{f"@{k}":<10}' for k in topk_list)
    print(f'\n  📊 {args.dataset} — Results:')
    print(f'  {"Metric":<15} {k_str}')
    print(f'  {"-" * (15 + 11 * len(topk_list))}')
    ndcg_str = ' '.join(f'{ndcgs[f"NDCG@{k}"]:<10.4f}' for k in topk_list)
    rec_str = ' '.join(f'{recalls[f"Recall@{k}"]:<10.4f}' for k in topk_list)
    print(f'  {"NDCG":<15} {ndcg_str}')
    print(f'  {"Recall":<15} {rec_str}')
    print()


if __name__ == '__main__':
    main()
