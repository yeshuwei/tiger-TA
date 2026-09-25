#!/usr/bin/env bash
# run_beauty_tier2_t5.sh — Beauty Tier-2：只补 4 个变体的 T5 生成式评测
#
# 2026-09-24 建。范围严格限定为用户 Tier-2 规格：
#   只补 Raw-CoOcc / ItemCF / Shuffled / ZeroPad 四个变体的 T5 NDCG@20（3 seeds）。
#
# ⚠ 本脚本**只调用 t5-only 模式**，因此：
#   - 不训练 RQ-VAE（t5-only 分支在 run_t5 之前就 exit 0）
#   - 不重新生成 SID codes
#   - 不修改任何超参（main.py / eval_trained_model.py 全走默认）
#   - 不重跑已有的 TA-SID / Content-only
#
# 逐字复用 E49 权威驱动 rqvae/run_behavior_ablation.sh 的 t5-only 分支 —— 与
# Sport/Toys E49 用的是同一段代码、同一 T5 配方（见 §四 检查报告）。
#
# 幂等：run_t5 对已存在的 eval 日志跳过，可随时中断续跑。
#
# 用法：nohup bash rqvae/run_beauty_tier2_t5.sh > logs/beauty_tier2_boot.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

DS=Beauty
LOG=logs/beauty_tier2_driver.log
mkdir -p logs
: > "$LOG"

# tag                 codes_basename
CELLS=(
  "rawcooc_768d   Beauty_rawcooc_768d_sid.npy"
  "itemcf_768d    Beauty_itemcf_768d_sid.npy"
  "shuffled_768d  Beauty_shuffled_768d_sid.npy"
  "zeropad_768d   Beauty_zeropad_768d_sid.npy"
)

for cell in "${CELLS[@]}"; do
  # shellcheck disable=SC2086
  set -- $cell
  TAG="$1"; CODES="$2"
  OUT="logs/beauty_tier2_${TAG}.log"

  echo "[$(date '+%F %T')] === Tier-2 ${DS} / ${TAG}   codes=${CODES}   seeds=2025 2026 2027 ===" | tee -a "$LOG"

  bash rqvae/run_behavior_ablation.sh "$DS" t5-only "$CODES" "$TAG" 2025 2026 2027 > "$OUT" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')]   ✗ ${TAG} 失败 rc=$rc  → 链路停止（详见 $OUT）" | tee -a "$LOG"
    exit 1
  fi

  for S in 2025 2026 2027; do
    E="model/logs/${DS}_${TAG}_seed${S}_eval.log"
    N=$(grep -oE "NDCG[[:space:]]+[0-9.]+[[:space:]]+[0-9.]+[[:space:]]+[0-9.]+" "$E" 2>/dev/null | awk '{print $4}')
    echo "[$(date '+%F %T')]   ✅ ${TAG} seed${S}  NDCG@20 = ${N:-解析失败}" | tee -a "$LOG"
  done
  echo "[$(date '+%F %T')]   → ${TAG} 完成（完整 stdout: $OUT）" | tee -a "$LOG"
done

echo "[$(date '+%F %T')] Tier-2 T5 全部完成（4 变体 × 3 seed = 12 次评测）" | tee -a "$LOG"
ls -la model/logs/Beauty_{rawcooc,itemcf,shuffled,zeropad}_768d_seed*_eval.log 2>/dev/null | tee -a "$LOG"
