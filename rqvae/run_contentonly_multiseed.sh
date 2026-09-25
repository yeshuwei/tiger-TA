#!/usr/bin/env bash
# run_contentonly_multiseed.sh — Content-only(768d) RQ-VAE 多种子可复现性（量化端 gate，无 T5）
#
# 目的：Q2 隔离实验。TA-SID(896d fused) 在 Toys seed2024 塌缩(62/256)，但 S0 已测
#       content-only(768d) seed2024 健康(148/256) → 需要 content-only 多种子，判断：
#         - 若 content-only 跨 seed 全部健康 → fusion/维度 提高了塌缩敏感度（Q2 第二种结局）
#         - 若 content-only 也跨 seed 摆动   → RQ-VAE 普遍敏感，fusion 无罪（Q2 第一种结局）
#       与 {DS}_Content768d_Baseline 同构（in_dim=768, 256³/3000ep/batch1024/lr1e-3），仅 --seed 不同。
#
# 用法:
#   bash rqvae/run_contentonly_multiseed.sh [DS] [SEED...]   # 默认 Toys 2025 2026
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS="${1:-Toys}"
shift 1 2>/dev/null || true
SEEDS=("$@"); [ ${#SEEDS[@]} -ge 1 ] || SEEDS=(2025 2026)
DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
EPS=0.01

CONTENT="data/${DS}/item_emb_768d_contentonly.parquet"
[ -f "$CONTENT" ] || { echo "✗ 缺 content-only 输入: $CONTENT"; exit 1; }

for S in "${SEEDS[@]}"; do
  CKPT_DIR="rqvae/ckpt/${DS}_Content768d_seed${S}"
  CODE_NPY="data/${DS}/${DS}_content768d_seed${S}_sid.npy"
  DIAG_REPORT="doc/e4x_${DSL}_contentonly_seed${S}.md"
  TRAIN_LOG="rqvae/logs/e4x_${DSL}_contentonly_seed${S}_train.log"

  echo "=================================================================="
  echo "[$(date '+%F %T')] seed=${S}  DS=${DS}  content-only(768d)  (num_emb_list=256 256 256)"
  echo "=================================================================="

  # ---------- A. 训练 RQ-VAE（--seed=$S，余全默认）----------
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] [A] 训练 RQ-VAE content-only  seed=${S}"
    python rqvae/main.py \
      --data_path "$CONTENT" \
      --ckpt_dir "$CKPT_DIR" \
      --num_emb_list 256 256 256 \
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
      --data_path "$CONTENT" \
      --output_file "$CODE_NPY" \
      --device cuda:0 || { echo "✗ codegen 失败 (seed=${S})"; exit 1; }
  else
    echo "[$(date '+%F %T')] [B] 跳过：$CODE_NPY 已存在"
  fi

  # ---------- C. 容量诊断（z-gap gate，含 L0 用码 = 塌缩判定）----------
  echo "[$(date '+%F %T')] [C] 容量诊断 -> $DIAG_REPORT"
  python tools/sid_capacity_diagnostic.py --datasets "$DS" --topk 1000 --n_random 3000 \
    --ckpt-tag "${DS}_Content768d_seed${S}" \
    --code-file "$(basename "$CODE_NPY")" \
    --ks 256 \
    --report "$DIAG_REPORT" \
    2>&1 | grep -E '真实模型码本|L0:|L1:|L2:|xq\[L0\]|xq\[L0\+L1\]|xq\[full\]|z \(' | head -20

  L0USED="$(python - "$DIAG_REPORT" <<'PY'
import re, sys
try:
    m = re.search(r"L0: 用 (\d+)/256", open(sys.argv[1]).read())
    print(m.group(1) if m else "NA")
except Exception:
    print("NA")
PY
)"
  echo "[$(date '+%F %T')] ★ seed=${S}  content-only L0=${L0USED}/256  (ε=${EPS}；TA-SID 参照 seed2024=62/256 塌缩)"
done
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (DS=${DS}, content-only seeds=${SEEDS[*]})"
echo "=================================================================="
