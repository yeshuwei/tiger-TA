#!/usr/bin/env bash
# run_e49grid_diag.sh — 跑齐 6 变体 × 2 数据集 的量化端点诊断（统一口径）
#
# 2026-09-23 建。目的：把 xq[full] AUC 与 E49 的 NDCG@20 对齐到同一口径，
# 使"诊断指标 vs 下游性能"的相关性检验成为可能。
#
# 统一口径：--n_random 5000 --skip_sweep（只出 anchors；容量扫描是另一个问题）
# 配对均已用 rqvae/audit_main_codes_provenance.py 的 L0 一致率法验证（12/12 = 100%）
#
# 用法：bash rqvae/run_e49grid_diag.sh
set -u
cd "$(dirname "$0")/.."

N_RANDOM=5000
OUTDIR=doc/e49grid
mkdir -p "$OUTDIR"
LOG=rqvae/logs/e49grid_diag.log
: > "$LOG"

# DS  变体         ckpt_tag                 codes_file
CELLS=(
  "Sport content  Sport_Content768d_Baseline   Sport_content768d_baseline_sid.npy"
  "Sport shuffled Sport_Shuffled_Ablation      Sport_shuffled_ablation_sid.npy"
  "Sport zeropad  Sport_Zeropad_Ablation       Sport_zeropad_ablation_sid.npy"
  "Sport itemcf   Sport_ItemCF_768d            Sport_itemcf_768d_sid.npy"
  "Sport rawcooc  Sport_RawCoOcc_768d          Sport_rawcooc_768d_sid.npy"
  "Sport ta_sid   Sport_TA_SID_768d            Sport_ta_sid_768d_sid.npy"
  "Toys  content  Toys_Content768d_Baseline    Toys_content768d_baseline_sid.npy"
  "Toys  shuffled Toys_Shuffled_768d           Toys_shuffled_768d_sid.npy"
  "Toys  zeropad  Toys_Zeropad_seed2024        Toys_zeropad_seed2024_sid.npy"
  "Toys  itemcf   Toys_ItemCF_768d             Toys_itemcf_768d_sid.npy"
  "Toys  rawcooc  Toys_RawCoOcc_768d           Toys_rawcooc_768d_sid.npy"
  "Toys  ta_sid   Toys_TA_SID_768d             Toys_ta_sid_768d_seed2024_sid.npy"
)

for cell in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $cell
  DS="$1"; VAR="$2"; TAG="$3"; CODES="$4"
  REP="$OUTDIR/${DS}_${VAR}.md"
  echo "[$(date '+%F %T')] === $DS / $VAR  (ckpt=$TAG codes=$CODES) ===" | tee -a "$LOG"
  python3 tools/sid_capacity_diagnostic.py \
      --datasets "$DS" --topk 1000 --n_random "$N_RANDOM" --skip_sweep \
      --ckpt-tag "$TAG" --code-file "$CODES" \
      --report "$REP" >> "$LOG" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "  ✗ 失败 rc=$rc" | tee -a "$LOG"
done

echo "[$(date '+%F %T')] 全部完成" | tee -a "$LOG"

# ---- 汇总：从 12 份报告里抓 汇总表 行 ----
echo "" | tee -a "$LOG"
echo "================ 12 格汇总 ================" | tee -a "$LOG"
grep -h -A99 "汇总表（AUC + 码本利用）" "$OUTDIR"/*.md \
  | grep -E "^(Sport|Toys) " | sort -u | tee -a "$LOG"
