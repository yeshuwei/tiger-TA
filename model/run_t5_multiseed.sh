#!/usr/bin/env bash
# run_t5_multiseed.sh — 对指定 SID codes 跑 T5 3-seed 训练 + 评估（推荐层闭环）
#
# 用途：验证"某个 RQ-VAE 种子生成的 codes"在推荐层的表现。
#   典型场景：Toys seed2024 codes 塌缩(L0 62/256,z-gap 0.093) → T5 NDCG@20=0.0367
#             seed2025 codes 健康(L0 194/256,z-gap 0.0016) → 本脚本跑其 T5，
#             判断 512 的 +3.8% 提升是"修好坏种子"还是"Toys 真需更大容量"。
#
# 用法:
#   bash model/run_t5_multiseed.sh <DS> <codes_basename> <tag> [SEED...]
# 例:
#   bash model/run_t5_multiseed.sh Toys Toys_ta_sid_768d_seed2025_sid.npy rq2025 2025 2026 2027
#
# 命名: model/ckpt/tiger_ta_sid_768d_${dsl}_${tag}_seed${S}.pth  (tag 防覆盖既有主实验模型)
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS="${1:?need DS e.g. Toys}"; CODES="${2:?need codes basename e.g. Toys_ta_sid_768d_seed2025_sid.npy}"; TAG="${3:?need tag e.g. rq2025}"
shift 3
SEEDS=("$@"); [ ${#SEEDS[@]} -ge 1 ] || SEEDS=(2025 2026 2027)
DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"

CODE_PATH="data/${DS}/${CODES}"
[ -f "$CODE_PATH" ] || { echo "✗ 缺 codes: $CODE_PATH"; exit 1; }

echo "=================================================================="
echo "[$(date '+%F %T')] T5 3-seed: DS=${DS}  codes=${CODES}  tag=${TAG}  seeds=${SEEDS[*]}"
echo "=================================================================="

for S in "${SEEDS[@]}"; do
  TAG_FULL="tiger_ta_sid_768d_${DSL}_${TAG}_seed${S}"
  T5_CKPT="model/ckpt/${TAG_FULL}.pth"
  EVAL_LOG="model/logs/${TAG_FULL}_eval.log"
  if [ -f "$EVAL_LOG" ]; then
    echo "    $TAG_FULL eval 已存在, 跳过"
    continue
  fi
  if [ ! -f "$T5_CKPT" ]; then
    echo "[$(date '+%F %T')]    --- train T5 $TAG_FULL (seed=$S) ---"
    python model/main.py \
      --dataset_path "data/${DS}" \
      --code_path "$CODE_PATH" \
      --save_path "$T5_CKPT" \
      --log_path "model/logs/${TAG_FULL}.log" \
      --seed "$S" 2>&1 | tee "model/logs/${TAG_FULL}_t5_train.log"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ T5 训练失败 rc=$rc ($TAG_FULL)"; exit 1; }
  fi
  echo "[$(date '+%F %T')]    --- eval $TAG_FULL (seed=$S) ---"
  python3 tools/eval_trained_model.py \
    --dataset "$DS" \
    --ckpt_path "$T5_CKPT" \
    --code_path "$CODE_PATH" \
    --beam_size 30 --topk_list 5,10,20,50,100 --seed "$S" \
    2>&1 | tee "$EVAL_LOG"
done
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (DS=${DS}, tag=${TAG}, codes=${CODES})"
echo "    ckpts: model/ckpt/tiger_ta_sid_768d_${DSL}_${TAG}_seed*.pth"
echo "    evals: model/logs/tiger_ta_sid_768d_${DSL}_${TAG}_seed*_eval.log"
echo "=================================================================="
