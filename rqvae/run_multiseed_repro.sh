#!/usr/bin/env bash
# run_multiseed_repro.sh — RQ-VAE 多种子可复现性复跑（量化端 gate，无 T5）
#
# 背景：既有 RQ-VAE 训练种子硬编码 2024（main.py L62-67 曾写死，现 --seed 默认 2024）。
#       故 Toys 62/256 L0 塌缩 + z-gap 0.093 只挂在【一次确定性训练】上，3-seed 仅指 T5。
#       本脚本用不同 RQ-VAE 种子重训同一数据集同配置，检验塌缩/大 z-gap 是否跨种子复现：
#         - 结果 A: 塌缩跨种子复现        → Toys 系统性容量不足（正例稳健）
#         - 结果 B: 部分种子不塌缩         → 分配崩坏对初始化敏感 ⇒ 事后 z-gap 测是唯一可靠保护
#       每个 seed 只跑量化端(A 训练+B codegen+C 诊断)，判 z-gap，不进 T5。
#
# 用法:
#   bash rqvae/run_multiseed_repro.sh [DS] [K0] [SEED...]   # 默认 Toys 256 2025 2026
#   bash rqvae/run_multiseed_repro.sh Toys 256 2025 2026 2027
#
# 与 {DS}_TA_SID_768d 完全同构（epochs3000/batch1024/lr1e-3/256³/main.py 全默认），仅 --seed 不同。
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS="${1:-Toys}"; K0="${2:-256}"
shift 2 2>/dev/null || true
SEEDS=("$@"); [ ${#SEEDS[@]} -ge 1 ] || SEEDS=(2025 2026)
DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
EPS=0.01

FUSED="data/${DS}/fused_item_emb_768d.parquet"
[ -f "$FUSED" ] || { echo "✗ 缺 fused 输入: $FUSED"; exit 1; }

for S in "${SEEDS[@]}"; do
  CKPT_DIR="rqvae/ckpt/${DS}_TA_SID_seed${S}"
  CODE_NPY="data/${DS}/${DS}_ta_sid_768d_seed${S}_sid.npy"
  DIAG_REPORT="doc/e4x_${DSL}_seed${S}.md"
  TRAIN_LOG="rqvae/logs/e4x_${DSL}_seed${S}_train.log"

  echo "=================================================================="
  echo "[$(date '+%F %T')] seed=${S}  DS=${DS}  K0=${K0}  (num_emb_list=${K0} 256 256)"
  echo "=================================================================="

  # ---------- A. 训练 RQ-VAE（--seed=$S，余全默认）----------
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] [A] 训练 RQ-VAE  seed=${S}"
    python rqvae/main.py \
      --data_path "$FUSED" \
      --ckpt_dir "$CKPT_DIR" \
      --num_emb_list ${K0} 256 256 \
      --seed "$S" \
      2>&1 | tee "$TRAIN_LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ 训练失败 rc=$rc (seed=${S})"; exit 1; }
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
      --device cuda:0 || { echo "✗ codegen 失败 (seed=${S})"; exit 1; }
  else
    echo "[$(date '+%F %T')] [B] 跳过：$CODE_NPY 已存在"
  fi

  # ---------- C. 容量诊断（z-gap gate，含 L0 用码 = 塌缩判定）----------
  echo "[$(date '+%F %T')] [C] 容量诊断 -> $DIAG_REPORT"
  python tools/sid_capacity_diagnostic.py --datasets "$DS" --topk 1000 --n_random 3000 \
    --ckpt-tag "${DS}_TA_SID_seed${S}" \
    --code-file "$(basename "$CODE_NPY")" \
    --report "$DIAG_REPORT" \
    2>&1 | grep -E '真实模型码本|L0:|xq\[L0\]|xq\[full\]|z \(' | head -20

  GAP="$(python - "$DIAG_REPORT" <<'PY'
import re, sys
try:
    t = open(sys.argv[1]).read()
    mz = re.search(r"z \(编码器输出, 量化前\):\s*AUC=([\d.]+)", t)
    mx = re.search(re.escape("xq[full]") + r":\s*AUC=([\d.]+)", t)
    print(f"{float(mz.group(1)) - float(mx.group(1)):.4f}" if mz and mx else "NA")
except Exception:
    print("NA")
PY
)"
  # L0 用码数（塌缩判定：<128/256 即塌缩）
  L0USED="$(python - "$DIAG_REPORT" <<'PY'
import re, sys
try:
    m = re.search(r"L0: 用 (\d+)/256", open(sys.argv[1]).read())
    print(m.group(1) if m else "NA")
except Exception:
    print("NA")
PY
)"
  echo "[$(date '+%F %T')] ★ seed=${S}  L0=${L0USED}/256  z-gap=${GAP}  (ε=${EPS})"
done
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (DS=${DS}, seeds=${SEEDS[*]})"
echo "    判定: L0<128 且 z-gap>${EPS} → 塌缩复现(结果A); 反之 → 种子敏感(结果B)"
echo "=================================================================="
