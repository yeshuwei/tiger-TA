#!/usr/bin/env bash
# 4.3 T5 重训: Toys K0∈{512,1024}, 3-seed(2025/2026/2027), 与主实验逐参数一致
# 用法: bash rqvae/run_e43_toys_t5.sh [K0...]  默认 512 1024
cd "$(dirname "$0")/.."

# K0 → code 文件名映射
declare -A CODES=( [512]=Toys_ta_sid_L0512_sid.npy [1024]=Toys_ta_sid_L0Big_sid.npy )
KS=("$@"); [ ${#KS[@]} -eq 0 ] && KS=(512 1024)
SEEDS=(2025 2026 2027)

for K0 in "${KS[@]}"; do
  CODE="data/Toys/${CODES[$K0]}"
  [ -f "$CODE" ] || { echo "⚠ 缺 code: $CODE"; continue; }
  for S in "${SEEDS[@]}"; do
    TAG="tiger_ta_sid_768d_toys_L0${K0}_seed${S}"
    CKPT="model/ckpt/${TAG}.pth"
    if [ -f "$CKPT" ] && [ -f "model/logs/${TAG}_eval.log" ]; then
      echo "[$(date '+%F %T')] $TAG 已完成, 跳过"; continue
    fi
    echo "[$(date '+%F %T')] === 训练 $TAG (K0=${K0}, seed=${S}) ==="
    python model/main.py \
      --dataset_path data/Toys \
      --code_path "$CODE" \
      --save_path "$CKPT" \
      --log_path "model/logs/${TAG}.log" \
      --seed "$S" 2>&1 | tee "model/logs/${TAG}_t5_train.log"
    echo "[$(date '+%F %T')] === 评估 $TAG ==="
    python3 tools/eval_trained_model.py \
      --dataset Toys --ckpt_path "$CKPT" --code_path "$CODE" \
      --beam_size 30 --topk_list 5,10,20,50,100 --seed "$S" \
      2>&1 | tee "model/logs/${TAG}_eval.log"
  done
done
echo "[$(date '+%F %T')] ALL 4.3 TOYS T5 DONE"
