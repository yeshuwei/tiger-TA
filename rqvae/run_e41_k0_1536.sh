#!/usr/bin/env bash
# 4.1 补跑 K0=1536（用户方案含此点）
cd "$(dirname "$0")/.."

echo "[$(date '+%F %T')] Training K0=1536"
python rqvae/main.py \
  --data_path data/Toys/fused_item_emb_768d.parquet \
  --ckpt_dir rqvae/ckpt/Toys_TA_SID_L01536 \
  --num_emb_list 1536 256 256 \
  --e_dim 32 \
  --layers 512 256 128 64 \
  --sk_epsilons 0.0 0.0 0.003 \
  --sk_iters 50 \
  --kmeans_init True --kmeans_iters 100 \
  --epochs 3000 --batch_size 1024 --num_workers 4 \
  --eval_step 50 --lr 1e-3 --lr_scheduler_type linear --warmup_epochs 50 \
  --device cuda:0 \
  --lambda_trans 0.0 --lambda_balance 0.0 \
  2>&1 | tee rqvae/logs/e41_toys_K01536_train.log
echo "[$(date '+%F %T')] Done K0=1536"
