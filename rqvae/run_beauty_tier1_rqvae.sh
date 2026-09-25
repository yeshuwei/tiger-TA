#!/usr/bin/env bash
# run_beauty_tier1_rqvae.sh — Beauty Tier-1：只训 RQ-VAE + 生成 codes
#
# 2026-09-23 建。目的：补齐 Beauty 缺的 4 个变体（RawCoOcc/ItemCF/Shuffled/Zeropad）
# 的 RQ-VAE checkpoint 与 SID codes，以便算 z / xq[full] 的 Behavior AUC。
#
# ⚠ 本脚本**不跑 T5、不跑推荐评测、不跑容量诊断**（那些属 Tier-2）。
#
# 配方与 rqvae/run_behavior_ablation.sh（E49 权威驱动）的 A/B/C 三步**逐字一致**：
#   同 RQ-VAE arch、同 codebook、同 latent dim、同 window=3 E 图、同 min_cooccur，
#   仅 dataset 换成 Beauty。main.py 全部走默认超参（epochs=3000/batch=1024/
#   num_emb=[256,256,256]/e_dim=32/layers=[512,256,128,64]/sk=[0,0,0.003]/seed=2024）。
# 变体嵌入（fused 896d）已于 2026-09-23 由 behavior_variants.py / ablation_data.py /
# fuse_embeddings.py 生成，本脚本**只读复用、不重生成**（避免覆盖既有产物）。
#
# 用法：bash rqvae/run_beauty_tier1_rqvae.sh
set -u
cd "$(dirname "$0")/.."

DS=Beauty
BEH_SUFFIX=768d
mkdir -p rqvae/logs
LOG=rqvae/logs/beauty_tier1_rqvae.log
: > "$LOG"

# variant  PASCAL     TAG
CELLS=(
  "rawcooc  RawCoOcc   rawcooc_768d"
  "itemcf   ItemCF     itemcf_768d"
  "shuffled Shuffled   shuffled_768d"
  "zeropad  Zeropad    zeropad_768d"
)

for cell in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $cell
  VARIANT="$1"; PASCAL="$2"; TAG="$3"
  FUSED="data/${DS}/fused_item_emb_${VARIANT}.parquet"
  CKPT_DIR="rqvae/ckpt/${DS}_${PASCAL}_${BEH_SUFFIX}"
  CODES="data/${DS}/${DS}_${TAG}_sid.npy"
  RQVAE_LOG="rqvae/logs/beauty_tier1_${VARIANT}_rqvae.log"

  echo "[$(date '+%F %T')] === ${DS} / ${VARIANT}  (ckpt=${CKPT_DIR}  codes=${CODES}) ===" | tee -a "$LOG"

  if [ ! -f "$FUSED" ]; then
    echo "  ✗ 缺 fused 输入: $FUSED（本脚本不生成，请先跑 behavior_variants/ablation_data/fuse）" | tee -a "$LOG"
    exit 1
  fi

  # ---------- C1: RQ-VAE ----------
  if [ ! -d "$CKPT_DIR" ]; then
    echo "  [C1] 训练 RQ-VAE -> ${CKPT_DIR}" | tee -a "$LOG"
    python rqvae/main.py \
      --data_path "$FUSED" \
      --ckpt_dir "$CKPT_DIR" \
      --num_emb_list 256 256 256 2>&1 | tee "$RQVAE_LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "  ✗ RQ-VAE 训练失败 rc=$rc" | tee -a "$LOG"; exit 1; }
  else
    echo "  [C1] 跳过：${CKPT_DIR} 已存在" | tee -a "$LOG"
  fi

  CKPT="$(ls -d ${CKPT_DIR}/*/ 2>/dev/null | tail -1)best_collision_model.pth"
  if [ ! -f "$CKPT" ]; then
    echo "  ✗ 缺 ckpt: $CKPT" | tee -a "$LOG"; exit 1
  fi

  # ---------- C2: codegen ----------
  echo "  [C2] codegen -> ${CODES}   (ckpt=${CKPT})" | tee -a "$LOG"
  python rqvae/generate_sid_codes_param.py \
    --ckpt_path "$CKPT" --data_path "$FUSED" --output_file "$CODES" \
    --device cuda:0 2>&1 | tee -a "$RQVAE_LOG"
  rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && { echo "  ✗ codegen 失败 rc=$rc" | tee -a "$LOG"; exit 1; }
done

echo "[$(date '+%F %T')] Tier-1 RQ-VAE + codes 全部完成（未跑 T5）" | tee -a "$LOG"
ls -la data/${DS}/Beauty_*_768d_sid.npy 2>/dev/null | tee -a "$LOG"
