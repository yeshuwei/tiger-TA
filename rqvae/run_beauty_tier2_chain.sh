#!/usr/bin/env bash
# run_beauty_tier2_chain.sh — Beauty Tier-2 自动接力：等 T5 跑完 → 跑汇总
#
# 幂等：若汇总报告已存在且晚于最后一份 eval 日志，则不重复跑。
#
# 用法：nohup bash rqvae/run_beauty_tier2_chain.sh > logs/beauty_tier2_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

DRIVER_LOG=logs/beauty_tier2_driver.log
SUM_LOG=logs/beauty_tier2_summary.log
MARK="Tier-2 T5 全部完成"

echo "[$(date '+%F %T')] chain 启动，等待 Tier-2 T5 完成标记（最长 16h）…"
for _ in $(seq 1 1920); do        # 16h = 30s × 1920
  if grep -q "$MARK" "$DRIVER_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] 检测到完成标记"
    break
  fi
  sleep 30
done

if ! grep -q "$MARK" "$DRIVER_LOG" 2>/dev/null; then
  echo "[$(date '+%F %T')] ✗ 超时未检测到完成标记。最后 10 行 driver 日志："
  tail -10 "$DRIVER_LOG"
  exit 1
fi

# 等 12 份 eval 日志全部落盘
for _ in $(seq 1 40); do
  n=$(ls model/logs/Beauty_{rawcooc,itemcf,shuffled,zeropad}_768d_seed*_eval.log 2>/dev/null | wc -l)
  [ "$n" -eq 12 ] && break
  echo "  … 已落盘 eval 日志: $n/12"
  sleep 15
done

echo "[$(date '+%F %T')] 运行 Tier-2 汇总"
{
  echo "================= Beauty Tier-2 汇总 ================="
  echo "执行时间: $(date '+%F %T')"
  python3 tools/beauty_tier2_summary.py
  echo "SUMMARY_EXIT=$?"
} > "$SUM_LOG" 2>&1

echo "[$(date '+%F %T')] 完成。结果："
echo "  报告 doc/beauty_tier2_conclusion.md"
echo "  CSV  doc/Beauty_Tier2_NDCG_results.csv"
echo "  三数据集表 doc/beauty_tier2_three_dataset_table.csv"
echo "  日志 $SUM_LOG"
tail -50 "$SUM_LOG"
