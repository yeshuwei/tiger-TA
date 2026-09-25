#!/usr/bin/env bash
# run_behavior_ablation.sh — E49 行为向量消融端到端驱动（幂等，可随时中断续跑）
#
# 协议（与主表逐位一致）：一个变体 = 一次 RQ-VAE（main.py 默认 seed=2024）
#   → 一个 codes.npy → T5 用 seed 2025/2026/2027 各训一次。
#
# 用法:
#   bash rqvae/run_behavior_ablation.sh <DS> <variant> [SEED...]
#       variant ∈ {rawcooc, itemcf, shuffled, zeropad}      SEED 默认 2025 2026 2027
#   bash rqvae/run_behavior_ablation.sh <DS> t5-only <codes_basename> <tag> [SEED...]
#       只补 T5（codes 已存在，例如 Toys zeropad 复用 Toys_zeropad_seed2024_sid.npy）
#
# 幂等：每个 stage `[ ! -e ... ]` 才执行；已有产物一律跳过、绝不覆盖。
# 无重命名/无复制：既有 legacy 产物（Sport shuffled/zeropad、Toys zeropad）按原名复用，
#   仅在本脚本输出里记录等价关系（e48 orphan 事故教训）。
#
# 环境：bare `python` = base env，与全部既有 RQ-VAE 日志一致
#   （rqvae/logs/*.log 均显示 /home/media601/anaconda3/lib/python3.10）。
#   T5 的模型初始化与前向在 base(tf4.57.6) 与 tiger_t5(tf4.30.2) 下已验逐字节相同，
#   故 T5 用哪个 env 不影响结果。
#
# 示例:
#   # P2 决策门：Sport 两变体各单 seed smoke test
#   bash rqvae/run_behavior_ablation.sh Sport rawcooc 2025
#   bash rqvae/run_behavior_ablation.sh Sport itemcf  2025
#   # P3：补另外两个 T5 seed
#   bash rqvae/run_behavior_ablation.sh Sport rawcooc 2026 2027
#   # P5：Toys zeropad 只补 T5
#   bash rqvae/run_behavior_ablation.sh Toys t5-only Toys_zeropad_seed2024_sid.npy zeropad_ablation 2025 2026 2027
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS="${1:?用法: $0 <DS> <variant|t5-only> [...]}"
VARIANT="${2:?用法: $0 <DS> <variant|t5-only> [...]}"
shift 2
DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
BEH_SUFFIX=768d

# ---------------------------------------------------------------- t5-only 模式
run_t5() {  # $1=codes_path(相对仓库根) $2=tag $3..=seeds
  local CODES="$1" TAG="$2"; shift 2
  local SEEDS=("$@"); [ ${#SEEDS[@]} -ge 1 ] || SEEDS=(2025 2026 2027)
  [ -f "$CODES" ] || { echo "✗ 缺 codes: $CODES"; exit 1; }

  echo "=================================================================="
  echo "[$(date '+%F %T')] T5  DS=${DS}  codes=${CODES}  tag=${TAG}  seeds=${SEEDS[*]}"
  echo "=================================================================="
  for S in "${SEEDS[@]}"; do
    local T5_CKPT="model/ckpt/tiger_${TAG}_${DSL}_seed${S}.pth"
    local EVAL_LOG="model/logs/${DS}_${TAG}_seed${S}_eval.log"
    # legacy 等价名（仅 seed2025，历史产物无 seed 后缀）——只读，不重命名
    [ -f "$T5_CKPT" ] || { [ "$S" = 2025 ] && [ -f "model/ckpt/tiger_${TAG}_${DSL}.pth" ] \
        && { T5_CKPT="model/ckpt/tiger_${TAG}_${DSL}.pth"; echo "    (复用 legacy ckpt 名: $T5_CKPT)"; }; }
    [ -f "$EVAL_LOG" ] || { [ "$S" = 2025 ] && [ -f "model/logs/${DS}_${TAG}_eval.log" ] \
        && { echo "    ✅ seed=${S} legacy eval 已存在: model/logs/${DS}_${TAG}_eval.log（跳过）"; continue; }; }

    if [ -f "$EVAL_LOG" ]; then
      echo "    ✅ seed=${S} eval 已存在: $EVAL_LOG（跳过）"; continue
    fi
    if [ ! -f "$T5_CKPT" ]; then
      echo "[$(date '+%F %T')]    [T5] train seed=${S} -> $T5_CKPT"
      python model/main.py \
        --dataset_path "data/${DS}" \
        --code_path "$CODES" \
        --save_path "$T5_CKPT" \
        --log_path "model/logs/tiger_${TAG}_${DSL}_seed${S}.log" \
        --seed "$S" 2>&1 | tee "model/logs/${DS}_${TAG}_seed${S}_t5_train.log"
      rc=${PIPESTATUS[0]}
      [ $rc -ne 0 ] && { echo "✗ T5 训练失败 rc=$rc (seed=$S)"; exit 1; }
    fi
    echo "[$(date '+%F %T')]    [EVAL] seed=${S} -> $EVAL_LOG"
    python3 tools/eval_trained_model.py \
      --dataset "$DS" --ckpt_path "$T5_CKPT" --code_path "$CODES" \
      --beam_size 30 --topk_list 5,10,20,50,100 --seed "$S" \
      2>&1 | tee "$EVAL_LOG"
    grep -E "NDCG@20|Recall@20" "$EVAL_LOG" | tail -2 || true
  done
}

# ---------------------------------------------------------------- t5-only 分派
if [ "$VARIANT" = "t5-only" ]; then
  CODES_ARG="${1:?t5-only 需要 <codes_basename>}"; TAG_ARG="${2:?t5-only 需要 <tag>}"
  shift 2
  run_t5 "data/${DS}/${CODES_ARG}" "$TAG_ARG" "$@"
  echo "=================================================================="
  echo "[$(date '+%F %T')] ALL DONE (t5-only DS=${DS} codes=${CODES_ARG} tag=${TAG_ARG})"
  echo "=================================================================="
  exit 0
fi

# ---------------------------------------------------------------- 全链路模式
SEEDS=("$@"); [ ${#SEEDS[@]} -ge 1 ] || SEEDS=(2025 2026 2027)

case "$VARIANT" in
  rawcooc) PASCAL=RawCoOcc; TAG=rawcooc_768d ;;
  itemcf)  PASCAL=ItemCF;   TAG=itemcf_768d  ;;
  shuffled) PASCAL=Shuffled; TAG=shuffled_768d ;;
  zeropad)  PASCAL=Zeropad;  TAG=zeropad_768d  ;;
  *) echo "✗ 未知 variant: $VARIANT (rawcooc|itemcf|shuffled|zeropad|t5-only)"; exit 1 ;;
esac

BEH="data/${DS}/behavior_emb_${VARIANT}.parquet"
FUSED="data/${DS}/fused_item_emb_${VARIANT}.parquet"
CKPT_DIR="rqvae/ckpt/${DS}_${PASCAL}_${BEH_SUFFIX}"
RQVAE_LOG="rqvae/logs/e49_${DSL}_${VARIANT}_rqvae.log"

echo "=================================================================="
echo "[$(date '+%F %T')] DS=${DS}  variant=${VARIANT}  tag=${TAG}  T5 seeds=${SEEDS[*]}"
echo "  RQ-VAE: seed=main.py 默认(2024)   in 896d → 3×256×32"
echo "=================================================================="

# ---------- A. behavior embedding ----------
if [ ! -f "$BEH" ]; then
  case "$VARIANT" in
    rawcooc|itemcf)
      echo "[$(date '+%F %T')] [A] behavior_variants.py --variant $VARIANT"
      python rqvae/behavior_variants.py --dataset "$DS" --variant "$VARIANT" \
        2>&1 | tee "rqvae/logs/e49_${DSL}_${VARIANT}_beh.log"
      rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { echo "✗ behavior_variants 失败 rc=$rc"; exit 1; } ;;
    shuffled|zeropad)
      echo "[$(date '+%F %T')] [A] ablation_data.py --variants $VARIANT（只写该变体，避免副作用）"
      python rqvae/ablation_data.py --dataset "$DS" --seed 2025 --variants "$VARIANT" ;;
  esac
else
  echo "[$(date '+%F %T')] [A] 跳过：$BEH 已存在"
fi

# ---------- B. fuse ----------
if [ "$VARIANT" = "rawcooc" ] || [ "$VARIANT" = "itemcf" ]; then
  if [ ! -f "$FUSED" ]; then
    echo "[$(date '+%F %T')] [B] fuse -> $FUSED"
    python rqvae/fuse_embeddings.py --dataset "$DS" \
      --content_file item_emb_${BEH_SUFFIX}.parquet \
      --behavior_file "behavior_emb_${VARIANT}.parquet" \
      --output_file "fused_item_emb_${VARIANT}.parquet" \
      2>&1 | tee "rqvae/logs/e49_${DSL}_${VARIANT}_fuse.log"
    rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { echo "✗ fuse 失败 rc=$rc"; exit 1; }
  else
    echo "[$(date '+%F %T')] [B] 跳过：$FUSED 已存在"
  fi
fi
[ -f "$FUSED" ] || { echo "✗ 缺 fused 输入: $FUSED"; exit 1; }

# ---------- B2. codes 解析：新名优先，其次既有 legacy 名（只读复用，不重命名）----------
CODES="data/${DS}/${DS}_${TAG}_sid.npy"
TAG_USED="$TAG"
if [ ! -f "$CODES" ]; then
  for cand in "data/${DS}/${DS}_${VARIANT}_ablation_sid.npy|${VARIANT}_ablation" \
              "data/${DS}/${DS}_${VARIANT}_seed2024_sid.npy|${VARIANT}_ablation" \
              "data/${DS}/${DS}_${TAG}_sid.npy|${TAG}"; do
    p="${cand%%|*}"; t="${cand##*|}"
    if [ -f "$p" ]; then
      CODES="$p"; TAG_USED="$t"
      echo "  ↪ 复用既有 codes（legacy 名，等价记录，不重命名）: $CODES  tag=$TAG_USED"
      break
    fi
  done
fi

# ---------- C. RQ-VAE + codegen（仅当 codes 尚不存在）----------
if [ ! -f "$CODES" ]; then
  mkdir -p rqvae/logs
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] [C1] 训练 RQ-VAE -> $CKPT_DIR"
    python rqvae/main.py \
      --data_path "$FUSED" \
      --ckpt_dir "$CKPT_DIR" \
      --num_emb_list 256 256 256 \
      2>&1 | tee "$RQVAE_LOG"
    rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { echo "✗ RQ-VAE 训练失败 rc=$rc"; exit 1; }
  else
    echo "[$(date '+%F %T')] [C1] 跳过：$CKPT_DIR 已存在"
  fi

  CKPT_SUB="$(ls -d ${CKPT_DIR}/*/ 2>/dev/null | tail -1)"
  CKPT="${CKPT_SUB}best_collision_model.pth"
  [ -f "$CKPT" ] || { echo "✗ 缺 ckpt: $CKPT"; exit 1; }
  echo "[$(date '+%F %T')] [C2] codegen -> $CODES   (ckpt=$CKPT)"
  python rqvae/generate_sid_codes_param.py \
    --ckpt_path "$CKPT" --data_path "$FUSED" --output_file "$CODES" \
    --device cuda:0 2>&1 | tee -a "$RQVAE_LOG"
  rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { echo "✗ codegen 失败 rc=$rc"; exit 1; }
else
  echo "[$(date '+%F %T')] [C] 跳过：$CODES 已存在"
fi

# ---------- D. 量化端容量诊断（gate 留档，失败不阻断）----------
DIAG="doc/e49_${DSL}_${TAG_USED}.md"
if [ ! -f "$DIAG" ]; then
  echo "[$(date '+%F %T')] [D] 容量诊断 -> $DIAG"
  python tools/sid_capacity_diagnostic.py --datasets "$DS" --topk 1000 --n_random 3000 \
    --ckpt-tag "$(basename "${CKPT_DIR}")" \
    --code-file "$(basename "$CODES")" --report "$DIAG" 2>&1 | tail -5 || true
fi

# ---------- E. T5 × seeds ----------
run_t5 "$CODES" "$TAG_USED" "${SEEDS[@]}"

echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (DS=${DS} variant=${VARIANT} tag=${TAG_USED})"
echo "    codes: $CODES"
echo "    ckpts: model/ckpt/tiger_${TAG_USED}_${DSL}_seed*.pth"
echo "    evals: model/logs/${DS}_${TAG_USED}_seed*_eval.log"
echo "=================================================================="
