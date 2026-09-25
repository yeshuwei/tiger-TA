#!/usr/bin/env bash
# run_zeropad_multiseed.sh — Toys ZeroPad(896d = content 768 + 128 死维) RQ-VAE 多种子（量化端 gate，无 T5）
#
# 目的：Q2 维度-隔离实验。TA-SID(896d=content+behavior) seed2024 塌缩(62/256)，
#       content-only(768d) seed2024 健康(148/256) → 二择一：是"896 维度"还是"behavior 内容"？
#       ZeroPad(896d=content+128零) 把维度对齐到 896、内容保持 content-only：
#         - ZeroPad@seed2024 塌缩     → 维度(高维输入) 才是塌缩诱因
#         - ZeroPad@seed2024 健康     → behavior 内容 才是塌缩诱因（融合改变了几何）
# 与 TA-SID 同构（in_dim=896, 256³/3000ep/batch1024/lr1e-3），仅 --seed/输入 不同。
#
# 用法:
#   bash rqvae/run_zeropad_multiseed.sh [DS] [SEED...]   # 默认 Toys 2024 2025
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS="${1:-Toys}"
shift 1 2>/dev/null || true
SEEDS=("$@"); [ ${#SEEDS[@]} -ge 1 ] || SEEDS=(2024 2025)
DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"

ZEROPAD="data/${DS}/fused_item_emb_zeropad.parquet"
[ -f "$ZEROPAD" ] || { echo "✗ 缺 zeropad 输入: $ZEROPAD"; exit 1; }

for S in "${SEEDS[@]}"; do
  CKPT_DIR="rqvae/ckpt/${DS}_Zeropad_seed${S}"
  CODE_NPY="data/${DS}/${DS}_zeropad_seed${S}_sid.npy"
  DIAG_REPORT="doc/e4x_${DSL}_zeropad_seed${S}.md"
  TRAIN_LOG="rqvae/logs/e4x_${DSL}_zeropad_seed${S}_train.log"

  echo "=================================================================="
  echo "[$(date '+%F %T')] seed=${S}  DS=${DS}  ZeroPad(896d=content+0)  (num_emb_list=256 256 256)"
  echo "=================================================================="

  if [ ! -d "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] [A] 训练 RQ-VAE ZeroPad  seed=${S}"
    python rqvae/main.py \
      --data_path "$ZEROPAD" \
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

  if [ ! -f "$CODE_NPY" ]; then
    echo "[$(date '+%F %T')] [B] 生成 SID codes -> $CODE_NPY"
    python rqvae/generate_sid_codes_param.py \
      --ckpt_path "$CKPT" \
      --data_path "$ZEROPAD" \
      --output_file "$CODE_NPY" \
      --device cuda:0 || { echo "✗ codegen 失败 (seed=${S})"; exit 1; }
  else
    echo "[$(date '+%F %T')] [B] 跳过：$CODE_NPY 已存在"
  fi

  echo "[$(date '+%F %T')] [C] 容量诊断 -> $DIAG_REPORT"
  python tools/sid_capacity_diagnostic.py --datasets "$DS" --topk 1000 --n_random 3000 \
    --ckpt-tag "${DS}_Zeropad_seed${S}" \
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
  echo "[$(date '+%F %T')] ★ seed=${S}  ZeroPad L0=${L0USED}/256  (参照: TA-SID@2024=62 塌缩, content-only@2024=148 健康)"
done
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (DS=${DS}, ZeroPad seeds=${SEEDS[*]})"
echo "=================================================================="
