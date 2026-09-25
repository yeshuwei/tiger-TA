#!/usr/bin/env bash
# 4.1 K0 sweep — Toys RQ-VAE 只训量化端
# 已有数据点: K0=256 (Toys_TA_SID_768d), K0=1024 (Toys_TA_SID_L0Big)
# 本脚本补: 512 / 768 / 2048
# 所有参数与 E3 L0Big 完全一致，仅 num_emb_list 的 L0 不同。

cd "$(dirname "$0")/.."

for K0 in 512 768 2048; do
  echo "=========================================================="
  echo "[$(date '+%F %T')] Training K0=${K0}"
  echo "=========================================================="
  python rqvae/main.py \
    --data_path data/Toys/fused_item_emb_768d.parquet \
    --ckpt_dir rqvae/ckpt/Toys_TA_SID_L0${K0} \
    --num_emb_list ${K0} 256 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --sk_epsilons 0.0 0.0 0.003 \
    --sk_iters 50 \
    --kmeans_init True --kmeans_iters 100 \
    --epochs 3000 --batch_size 1024 --num_workers 4 \
    --eval_step 50 --lr 1e-3 --lr_scheduler_type linear --warmup_epochs 50 \
    --device cuda:0 \
    --lambda_trans 0.0 --lambda_balance 0.0 \
    2>&1 | tee rqvae/logs/e41_toys_K0${K0}_train.log
  echo "[$(date '+%F %T')] Done K0=${K0} exit=$?"
done
echo "[$(date '+%F %T')] ALL K0 SWEEP DONE"
