"""Resume T5 training from a saved checkpoint.
Usage:
    python resume_train.py --resume ./ckpt/toys.pth
All other arguments are read from the checkpoint's config.
"""
import torch
import argparse
import logging
import os
import random
import numpy as np
from main import TIGER, train, evaluate, set_seed
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from transformers import T5ForConditionalGeneration, T5Config

DATASET = "Toys"   # Change to "Beauty" or "Sports" for other datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume TIGER training")
    parser.add_argument('--resume', type=str, required=True, help='Path to checkpoint to resume from')
    parser.add_argument('--save_path', type=str, default='./ckpt/toys.pth', help='Path to save the model')
    parser.add_argument('--log_path', type=str, default='./logs/toys.log', help='Path to the log file')
    parser.add_argument('--num_epochs', type=int, default=200, help='Total number of epochs (including completed)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--dataset_path', type=str, default=f'../data/{DATASET}', help='Dataset path')
    parser.add_argument('--code_path', type=str, default=f'../data/{DATASET}/{DATASET}_t5_rqvae.npy', help='Code path')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=str, default='5,10,20', help='Top-k list (comma-separated)')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=20, help='Max sequence length')
    args = parser.parse_args()

    # Set up logging (append mode)
    logging.basicConfig(
        filename=args.log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    logging.info(f"Resuming from checkpoint: {args.resume}")
    logging.info(f"Args: {args}")

    # Parse topk_list from comma-separated string
    topk_list = [int(k) for k in args.topk_list.split(',')]

    # Set random seed
    set_seed(args.seed)

    # Load checkpoint
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
    config = {
        'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
        'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
        'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
        'max_len': args.max_len, 'topk_list': topk_list, 'beam_size': args.beam_size,
    }
    model = TIGER(config)
    model.load_state_dict(ckpt)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = GenRecDataset(
        dataset_path=args.dataset_path + '/train.parquet',
        code_path=args.code_path, mode='train', max_len=args.max_len
    )
    validation_dataset = GenRecDataset(
        dataset_path=args.dataset_path + '/valid.parquet',
        code_path=args.code_path, mode='evaluation', max_len=args.max_len
    )
    test_dataset = GenRecDataset(
        dataset_path=args.dataset_path + '/test.parquet',
        code_path=args.code_path, mode='evaluation', max_len=args.max_len
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=args.infer_size, shuffle=False)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=args.infer_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    model.to(device)
    best_ndcg = 0.0
    early_stop_counter = 0

    for epoch in range(args.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")

        avg_recalls, avg_ndcgs = evaluate(model, validation_dataloader, topk_list, args.beam_size, device)
        logging.info(f"Validation Dataset: {avg_recalls}")
        logging.info(f"Validation Dataset: {avg_ndcgs}")

        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0
            test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, topk_list, args.beam_size, device)
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test Dataset: {test_avg_recalls}")
            logging.info(f"Test Dataset: {test_avg_ndcgs}")
            torch.save(model.state_dict(), args.save_path)
            logging.info(f"Best model saved to {args.save_path}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= args.early_stop:
                logging.info("Early stopping triggered.")
                break
