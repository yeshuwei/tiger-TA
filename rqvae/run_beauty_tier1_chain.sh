#!/usr/bin/env bash
# run_beauty_tier1_chain.sh — Beauty Tier-1 自动链路尾部
#
# 等待 rqvae/run_beauty_tier1_rqvae.sh（RQ-VAE + codegen）跑完，随即自动执行
# tools/beauty_tier1_probe.py，把结果落到指定路径。**不含 T5、不含 Tier-2。**
#
# 幂等：若 probe 产物已存在且晚于 codes，则不重复跑。
#
# 用法：nohup bash rqvae/run_beauty_tier1_chain.sh > logs/beauty_tier1_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

DRIVER_LOG=logs/beauty_tier1_driver.log
PROBE_LOG=logs/beauty_tier1_probe.log
MARK="Tier-1 RQ-VAE + codes 全部完成"

echo "[$(date '+%F %T')] chain 启动，等待训练完成标记…"
for _ in $(seq 1 480); do        # 最长等 4 小时（30s × 480）
  if grep -q "$MARK" "$DRIVER_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] 检测到训练完成标记"
    break
  fi
  sleep 30
done

if ! grep -q "$MARK" "$DRIVER_LOG" 2>/dev/null; then
  echo "[$(date '+%F %T')] ✗ 超时未检测到完成标记，放弃。"
  exit 1
fi

# 等 3 个 codes 全部落盘（防 race）
for _ in $(seq 1 40); do
  n=$(ls data/Beauty/Beauty_{rawcooc,itemcf,shuffled,zeropad}_768d_sid.npy 2>/dev/null | wc -l)
  [ "$n" -eq 4 ] && break
  echo "  … 已落盘 codes: $n/4"
  sleep 15
done

echo "[$(date '+%F %T')] 运行 Tier-1 probe"
{
  echo "================= Beauty Tier-1 probe ================="
  echo "执行时间: $(date '+%F %T')"
  python3 tools/beauty_tier1_probe.py
  echo "PROBE_EXIT=$?"
} > "$PROBE_LOG" 2>&1

echo "[$(date '+%F %T')] 完成。结果："
echo "  报告 doc/beauty_tier1_probe.md"
echo "  CSV  doc/beauty_tier1_results.csv"
echo "  日志 $PROBE_LOG"
tail -40 "$PROBE_LOG"
