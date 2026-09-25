#!/usr/bin/env bash
# run_k0_stress.sh — 受控压测：把粗层码本压到 K0(<256) 制造"容量不足"，验证 z-gap 判据 + 翻倍恢复闭环
#
# 背景：Toys 是唯一"默认 256 自然塌缩"的数据集（n=1 说服力弱）。本脚本在健康数据集上
#       人为压小 K0，制造与 Toys 同构的容量不足故障，验证 Adaptive(z-gap 判据 + 翻倍) 能识别并修复。
# 恢复臂 = 现有 {DS} K0=256 模型（RQ-VAE / codes / 3-seed T5 均已存在，零新增训练）。
# 本脚本只训"压测臂"，并给出 z-gap 判定与（可选）压测臂 T5。
#
# 用法:
#   bash rqvae/run_k0_stress.sh [DATASET] [K0]        # 默认 Sport 128
#   SKIP_T5=1 bash rqvae/run_k0_stress.sh Sport 128   # 只到量化端 gate，不进 T5
#
# 注意: K0=128 若 z-gap<=0.01(未压到位) → 用 K0=64 重跑。
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS="${1:-Sport}"; K0="${2:-128}"
SKIP_T5="${SKIP_T5:-0}"
DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
EPS=0.01

CKPT_DIR="rqvae/ckpt/${DS}_TA_SID_L0${K0}"
CODE_NPY="data/${DS}/${DS}_ta_sid_L0${K0}_sid.npy"
FUSED="data/${DS}/fused_item_emb_768d.parquet"
DIAG_REPORT="doc/e4x_${DSL}_K0${K0}.md"
TRAIN_LOG="rqvae/logs/e4x_${DSL}_K0${K0}_train.log"

echo "=================================================================="
echo "[$(date '+%F %T')] run_k0_stress: DS=${DS}  K0=${K0}  SKIP_T5=${SKIP_T5}"
echo "=================================================================="

[ -f "$FUSED" ] || { echo "✗ 缺 fused 输入: $FUSED"; exit 1; }

# ---------- A. 训练 RQ-VAE（镜像 {DS}_TA_SID_768d 参数，仅 num_emb_list 的 L0 不同）----------
if [ ! -d "$CKPT_DIR" ]; then
  echo "[$(date '+%F %T')] [A] 训练 RQ-VAE  K0=${K0}"
  python rqvae/main.py \
    --data_path "$FUSED" \
    --ckpt_dir "$CKPT_DIR" \
    --num_emb_list ${K0} 256 256 \
    2>&1 | tee "$TRAIN_LOG"
  rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && { echo "✗ 训练失败 rc=$rc"; exit 1; }
else
  echo "[$(date '+%F %T')] [A] 跳过：已存在 $CKPT_DIR"
fi

CKPT_SUB="$(ls -d ${CKPT_DIR}/*/ 2>/dev/null | tail -1)"
CKPT="${CKPT_SUB}best_collision_model.pth"
[ -f "$CKPT" ] || { echo "✗ 缺 ckpt: $CKPT"; exit 1; }
echo "    ckpt = $CKPT"

# ---------- B. 生成 SID codes ----------
if [ ! -f "$CODE_NPY" ]; then
  echo "[$(date '+%F %T')] [B] 生成 SID codes -> $CODE_NPY"
  python rqvae/generate_sid_codes_param.py \
    --ckpt_path "$CKPT" \
    --data_path "$FUSED" \
    --output_file "$CODE_NPY" \
    --device cuda:0 || { echo "✗ codegen 失败"; exit 1; }
else
  echo "[$(date '+%F %T')] [B] 跳过：$CODE_NPY 已存在"
fi

# ---------- C. 容量诊断（z-gap gate）----------
echo "[$(date '+%F %T')] [C] 容量诊断 -> $DIAG_REPORT"
python tools/sid_capacity_diagnostic.py --datasets "$DS" --topk 1000 --n_random 3000 \
  --ckpt-tag "${DS}_TA_SID_L0${K0}" \
  --code-file "$(basename "$CODE_NPY")" \
  --report "$DIAG_REPORT" \
  2>&1 | grep -E '真实模型码本|xq\[L0\]|xq\[full\]|z \(|3-token|碰撞|判定|L0 用' | head -24

GAP="$(python - "$DIAG_REPORT" <<'PY'
import re, sys
try:
    t = open(sys.argv[1]).read()
    mz = re.search(r"z \(编码器输出, 量化前\):\s*AUC=([\d.]+)", t)
    mx = re.search(re.escape("xq[full]") + r":\s*AUC=([\d.]+)", t)
    if mz and mx:
        print(f"{float(mz.group(1)) - float(mx.group(1)):.4f}")
    else:
        print("NA")
except Exception:
    print("NA")
PY
)"
echo "[$(date '+%F %T')] ★ z-gap@K0=${K0} = ${GAP}   (ε=${EPS})"

if [ "$GAP" = "NA" ]; then
  echo "✗ 无法解析 z-gap，请查看 $DIAG_REPORT"; exit 1
fi
python - "$GAP" "$EPS" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
if [ $? -eq 0 ]; then
  echo "[$(date '+%F %T')] [gate] 压测有效：容量不足(z-gap>ε)。恢复臂 = 现有 ${DS} K0=256（已存在，无需重训）"
else
  echo "[$(date '+%F %T')] [gate] z-gap<=ε：K0=${K0} 未压到位 → 建议 K0=$((K0/2)) 重跑；不进入 T5"
  exit 0
fi

# ---------- D. 压测臂 T5 3-seed + eval（256 臂复用现成）----------
if [ "$SKIP_T5" -eq 1 ]; then
  echo "[$(date '+%F %T')] SKIP_T5=1，量化端 gate 完成，退出。"; exit 0
fi

SEEDS=(2025 2026 2027)
echo "[$(date '+%F %T')] [D] T5 3-seed 训练 + 评估 (${DS}, K0=${K0}, code=$CODE_NPY)"
for S in "${SEEDS[@]}"; do
  TAG="tiger_ta_sid_768d_${DSL}_L0${K0}_seed${S}"
  T5_CKPT="model/ckpt/${TAG}.pth"
  EVAL_LOG="model/logs/${TAG}_eval.log"
  if [ -f "$EVAL_LOG" ]; then
    echo "    $TAG eval 已存在, 跳过"
    continue
  fi
  if [ ! -f "$T5_CKPT" ]; then
    echo "[$(date '+%F %T')]    --- train T5 $TAG (seed=$S) ---"
    python model/main.py \
      --dataset_path "data/${DS}" \
      --code_path "$CODE_NPY" \
      --save_path "$T5_CKPT" \
      --log_path "model/logs/${TAG}.log" \
      --seed "$S" 2>&1 | tee "model/logs/${TAG}_t5_train.log"
  fi
  echo "[$(date '+%F %T')]    --- eval $TAG (seed=$S) ---"
  python3 tools/eval_trained_model.py \
    --dataset "$DS" \
    --ckpt_path "$T5_CKPT" \
    --code_path "$CODE_NPY" \
    --beam_size 30 --topk_list 5,10,20,50,100 --seed "$S" \
    2>&1 | tee "$EVAL_LOG"
done
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE  (DS=${DS}, K0=${K0})"
echo "    压测臂: ${CODE_NPY} / $DIAG_REPORT"
echo "    恢复臂(现成): data/${DS}/${DS}_ta_sid_768d_sid.npy + model/logs/${DS}_ta_sid_768d_seed*_eval.log"
echo "=================================================================="
