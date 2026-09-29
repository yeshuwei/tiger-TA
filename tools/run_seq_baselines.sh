#!/usr/bin/env bash
# run_seq_baselines.sh — GRU4Rec / BERT4Rec 三 seed 全量训练驱动
#
# 目的：补齐经典序列基线（回应"只挑了 SASRec 一个基线"）。
# 协议与 SASRec 完全一致：test.parquet 任务 / 全量用户 / 全量物品排序 /
# best-val NDCG@20 checkpoint / 3 seed（2025 2026 2027）。
#
# 纪律（沿用 rqvae/run_multiseed_repro.sh）：
#   - set -u、cd 到项目根、幂等跳过（产在即跳过）、2>&1 | tee、PIPESTATUS 检查退出码
#   - **单卡串行**：不并行两个方法（避免计时与 kernel 选择互相污染）
#   - 绝不写 sasrec_output/（那里的数字已进论文）
#
# 用法:
#   bash tools/run_seq_baselines.sh                 # 两方法 × 三数据集 × 三 seed
#   bash tools/run_seq_baselines.sh gru4rec         # 只跑 GRU4Rec
#   bash tools/run_seq_baselines.sh bert4rec Beauty # 只跑 BERT4Rec × Beauty
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

WHICH="${1:-all}"
ONLY_DS="${2:-}"
SEEDS=(2025 2026 2027)

declare -A SCRIPT=( [gru4rec]="tools/run_gru4rec.py" [bert4rec]="tools/run_bert4rec.py" )
declare -A OUTDIR=( [gru4rec]="gru4rec_output"       [bert4rec]="bert4rec_output" )

if [ "$WHICH" = "all" ]; then
  METHODS=(gru4rec bert4rec)
else
  METHODS=("$WHICH")
fi

if [ -n "$ONLY_DS" ]; then
  DATASETS=("$ONLY_DS")
else
  DATASETS=(Sport Toys Beauty)
fi

# 防御：绝不允许写进 SASRec 的权威产物目录
for m in "${METHODS[@]}"; do
  case "${OUTDIR[$m]}" in
    sasrec_output*) echo "✗ 输出目录指向 sasrec_output，拒绝执行"; exit 1;;
  esac
done

echo "=================================================================="
echo "[$(date '+%F %T')] 序列基线驱动启动"
echo "  方法   : ${METHODS[*]}"
echo "  数据集 : ${DATASETS[*]}"
echo "  seeds  : ${SEEDS[*]}"
echo "=================================================================="

for m in "${METHODS[@]}"; do
  SCRIPT_PATH="${SCRIPT[$m]}"
  OUT_DIR="${OUTDIR[$m]}"
  [ -f "$SCRIPT_PATH" ] || { echo "✗ 缺脚本 $SCRIPT_PATH"; exit 1; }

  for DS in "${DATASETS[@]}"; do
    DS_DIR="${OUT_DIR}/${DS}"
    mkdir -p "$DS_DIR"
    TOPLOG="${DS_DIR}/${m}_3seed_all.log"

    echo "" | tee -a "$TOPLOG"
    echo "##################################################################" | tee -a "$TOPLOG"
    echo "[$(date '+%F %T')] ${m} / ${DS}" | tee -a "$TOPLOG"
    echo "##################################################################" | tee -a "$TOPLOG"

    # ---------- 逐 seed 训练（幂等：best.txt 存在即跳过）----------
    for S in "${SEEDS[@]}"; do
      BEST="${DS_DIR}/seed${S}_best.txt"
      PERLOG="${DS_DIR}/seed${S}_train.log"

      if [ -f "$BEST" ]; then
        echo "[$(date '+%F %T')] 跳过 seed=${S}（已存在 $BEST）" | tee -a "$TOPLOG"
        continue
      fi

      echo "[$(date '+%F %T')] 训练 ${m} ${DS} seed=${S}" | tee -a "$TOPLOG"
      python3 "$SCRIPT_PATH" --dataset "$DS" --seeds "$S" \
              --output_dir "$OUT_DIR" 2>&1 | tee "$PERLOG"
      rc=${PIPESTATUS[0]}
      if [ $rc -ne 0 ]; then
        echo "✗ 训练失败 rc=$rc (${m} ${DS} seed=${S})，见 $PERLOG" | tee -a "$TOPLOG"
        exit 1
      fi
      [ -f "$BEST" ] || { echo "✗ 训练返回 0 但未产出 $BEST —— 疑似静默失败" | tee -a "$TOPLOG"; exit 1; }
    done

    # ---------- 三 seed 收齐后复建 summary ----------
    echo "[$(date '+%F %T')] 复建 summary_3seed.txt" | tee -a "$TOPLOG"
    python3 "$SCRIPT_PATH" --dataset "$DS" --output_dir "$OUT_DIR" --summary_only 2>&1 | tee -a "$TOPLOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ summary 复建失败 rc=$rc" | tee -a "$TOPLOG"; exit 1; }

    [ -f "${DS_DIR}/summary_3seed.txt" ] || {
      echo "✗ 缺 summary_3seed.txt (${m} ${DS})" | tee -a "$TOPLOG"; exit 1; }

    echo "--- ${DS_DIR}/summary_3seed.txt ---" | tee -a "$TOPLOG"
    cat "${DS_DIR}/summary_3seed.txt" | tee -a "$TOPLOG"
  done
done

echo ""
echo "##################################################################"
echo "[$(date '+%F %T')] ALL DONE"
echo "##################################################################"
echo "产物:"
for m in "${METHODS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    echo "  ${OUTDIR[$m]}/${DS}/summary_3seed.txt"
  done
done
echo ""
echo "下一步（零 GPU）："
echo "  python3 tools/paired_significance.py --extra-methods GRU4Rec BERT4Rec \\"
echo "      --out-md doc/seq_baselines_significance.md \\"
echo "      --out-csv doc/seq_baselines_significance.csv"
