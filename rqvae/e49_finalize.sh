#!/usr/bin/env bash
# e49_finalize.sh — E49 收尾检查 + 结果汇总（供 crontab 调度，**不依赖 Claude 会话**）
#
# 用途：后台链 `rqvae/run_e49_p3_p6.sh` 跑完后，自动产出「命名陷阱安全版」结果表。
#       由 `tools/e49_summary.py` 完成解析（逐 lane 显式文件名，不用 tag glob）。
#
# 幂等：可重复运行。未跑完时也照常出**部分快照**并标注进度，不会破坏任何既有产物。
#       本脚本**不修改** `doc/e49_behavior_ablation_results.md`（P6 段产物），只写自己的
#       `doc/e49_results_corrected.md`，避免覆盖。
#
# 退出码：0 = 36 路全齐；1 = 尚未跑完（属正常中间态，非错误）
#
# 手动运行：
#   bash rqvae/e49_finalize.sh

set -uo pipefail

# cron 的默认环境极简（PATH 常只有 /usr/bin:/bin），显式补全，避免 date/tee/grep 找不到
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

ROOT="/data/gtx/project/code/TIGER_minilm"
PYTHON="/home/media601/anaconda3/bin/python"
CHAIN_LOG="$ROOT/rqvae/logs/e49_p3_p6_bg.log"
FIN_LOG="$ROOT/rqvae/logs/e49_finalize.log"
OUT_DOC="$ROOT/doc/e49_results_corrected.md"
DONE_MARK="E49 全链路完成"

cd "$ROOT" || { echo "✗ 无法进入 $ROOT"; exit 2; }
mkdir -p "$(dirname "$FIN_LOG")"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$FIN_LOG"; }

hr() { printf '%.0s=' {1..70}; echo; }

{
  hr
  log "E49 收尾检查启动（crontab 调度，独立于 Claude 会话）"
  hr

  # ---------- 1. 后台链状态 ----------
  if [ -f "$CHAIN_LOG" ]; then
    if grep -aq "$DONE_MARK" "$CHAIN_LOG" 2>/dev/null; then
      log "✅ 后台链已完成（检测到标记：$DONE_MARK）"
      CHAIN_DONE=1
    else
      log "⏳ 后台链**尚未完成**（未检测到标记：$DONE_MARK）"
      log "   最后进度："
      grep -aE '^\[2026' "$CHAIN_LOG" | tail -3 | sed 's/^/     /' | tee -a "$FIN_LOG"
      CHAIN_DONE=0
    fi
  else
    log "⚠ 找不到链路日志 $CHAIN_LOG"
    CHAIN_DONE=0
  fi

  # 链路进程是否还活着
  PIDF="$ROOT/rqvae/logs/e49_p3_p6.pid"
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    log "   链路进程存活：pid=$(cat "$PIDF")"
  else
    log "   链路进程**已不在**（可能已结束或被中断）"
  fi

  # ---------- 2. 产出结果表 ----------
  log "→ 运行 tools/e49_summary.py"
  if [ ! -x "$PYTHON" ]; then
    log "✗ 找不到 python：$PYTHON"
    exit 2
  fi
  "$PYTHON" "$ROOT/tools/e49_summary.py" --out "$OUT_DOC" --quiet >>"$FIN_LOG" 2>&1
  RC=$?
  log "   summary 退出码=$RC（0=36路全齐 / 1=未跑完）"

  # ---------- 3. 摘要 ----------
  if [ -f "$OUT_DOC" ]; then
    log "✅ 结果表已写出：$OUT_DOC"
    log "   进度行：$(grep -a '^\*\*进度\*\*' "$OUT_DOC" | head -1)"
    log "   未完成 lane："
    sed -n '/^## 5\. 未完成 lane/,$p' "$OUT_DOC" | tail -n +3 | sed 's/^/     /' | tee -a "$FIN_LOG"
  else
    log "✗ 结果表未生成"
  fi

  # ---------- 4. P6 产物提示 ----------
  P6DOC="$ROOT/doc/e49_behavior_ablation_results.md"
  if [ -f "$P6DOC" ]; then
    log "⚠ P6 段产物存在于 $P6DOC —— 其 **Toys TA-SID** 与 **Toys Shuffled** 两行有命名陷阱，"
    log "  应作废，以 $OUT_DOC 为准（见 e48_toys_main_codes_provenance_fix.md）。"
  fi

  hr
  if [ "$CHAIN_DONE" = "1" ] && [ "$RC" = "0" ]; then
    log "🎉 E49 全部 36 路齐备，收尾完成。"
  else
    log "ℹ 本次收尾时尚未跑完（正常中间态）。下次 crontab 触发会自动重试。"
  fi
  hr
} 2>&1

exit $RC
