#!/usr/bin/env bash
# run_checkpoint_trajectory.sh — Checkpoint Trajectory 实验驱动（训练 + codegen）
#
# 对应实验方案：doc/exp/checkpoint_trajectory/experiment_plan.md
#
# 做两件事，全部落在**全新根目录** `rqvae/checkpoint_trajectory/` 下：
#   A. 重训 9 个 TA-SID RQ-VAE run（Beauty/Sport/Toys × seed 2024/2025/2026），
#      配置与既有 run 逐字同构（epochs3000 / batch1024 / lr1e-3 / 256³ / sk[0,0,0.003]），
#      额外在完成 50/500/1000/2000 轮训练时各存一份 traj ckpt。
#   B. 对每份 traj ckpt 调 rqvae/generate_sid_codes_param.py 生成 SID codes
#      （与既有 codegen 同一脚本、同一 device，保证 quantization/code generation 口径不变）。
#
# ⚠ 隔离理由：既有工具（find_best_ckpt / find_ckpt_pair / sid_quality）按
#    `rqvae/ckpt/{tag}/*/` glob + mtime 取最新 run。若把重训结果写回 `rqvae/ckpt/{tag}/`，
#    会**污染既有审计路径**。故本实验全部产物落在 rqvae/checkpoint_trajectory/ 下。
#
# 幂等：已存在完整 4 份 traj ckpt 的 run 会跳过训练；已存在的 codes 会跳过 codegen。
#
# 用法：
#   bash rqvae/run_checkpoint_trajectory.sh                 # 全部 9 个 run
#   bash rqvae/run_checkpoint_trajectory.sh Toys_TA_SID_seed2026   # 只跑指定 tag（pilot）
#   bash rqvae/run_checkpoint_trajectory.sh TAG_A TAG_B     # 只跑指定若干 tag
#
# 并发安全：manifest 按 tag 拆成 manifest.d/{TAG}.tsv 片段（无共享文件写入），
#           故可安全地多进程并行跑不同数据集。

set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PYTHON:-/home/media601/anaconda3/bin/python}"

TRAJ_ROOT="rqvae/checkpoint_trajectory"
CKPT_ROOT="$TRAJ_ROOT/ckpt"
CODE_ROOT="$TRAJ_ROOT/codes"
LOG_ROOT="$TRAJ_ROOT/logs"
MANIFEST_DIR="$TRAJ_ROOT/manifest.d"
EPOCHS=(50 500 1000 2000)

mkdir -p "$CKPT_ROOT" "$CODE_ROOT" "$LOG_ROOT" "$MANIFEST_DIR"

# 污染审计基准：本实验开始的时间戳。probe 的 toys_audit 阶段用它证明
# 「实验开始后 rqvae/ckpt/ 与 data/ 下没有任何文件被改动」。
[ -f "$TRAJ_ROOT/.experiment_start" ] || touch "$TRAJ_ROOT/.experiment_start"

# DS | seed | tag | data_path  （data_path 逐格取自既有 ckpt 的 args.data_path）
RUNS=(
  "Beauty|2024|Beauty_TA_SID|data/Beauty/fused_item_emb.parquet"
  "Beauty|2025|Beauty_TA_SID_seed2025|data/Beauty/fused_item_emb_768d.parquet"
  "Beauty|2026|Beauty_TA_SID_seed2026|data/Beauty/fused_item_emb_768d.parquet"
  "Sport|2024|Sport_TA_SID_768d|data/Sport/fused_item_emb_768d.parquet"
  "Sport|2025|Sport_TA_SID_seed2025|data/Sport/fused_item_emb_768d.parquet"
  "Sport|2026|Sport_TA_SID_seed2026|data/Sport/fused_item_emb_768d.parquet"
  "Toys|2024|Toys_TA_SID_768d|data/Toys/fused_item_emb_768d.parquet"
  "Toys|2025|Toys_TA_SID_seed2025|data/Toys/fused_item_emb_768d.parquet"
  "Toys|2026|Toys_TA_SID_seed2026|data/Toys/fused_item_emb_768d.parquet"
)

ONLY_LIST=("$@")          # 空 = 全部
want_tag() {
  [ "${#ONLY_LIST[@]}" -eq 0 ] && return 0
  local t; for t in "${ONLY_LIST[@]}"; do [ "$t" = "$1" ] && return 0; done
  return 1
}
epfile() { printf "traj_epoch%05d.pth" "$1"; }
epname() { printf "ep%s" "$1"; }

# 返回该 tag 下含全部 4 份 traj ckpt 的 run 目录（无则空）
find_done_run() {
  local tag="$1" d n
  for d in $(ls -d "$CKPT_ROOT/$tag"/*/ 2>/dev/null | sort); do
    n=0
    for e in "${EPOCHS[@]}"; do [ -f "$d$(epfile "$e")" ] && n=$((n+1)); done
    [ "$n" -eq "${#EPOCHS[@]}" ] && { echo "$d"; return; }
  done
  echo ""
}

for spec in "${RUNS[@]}"; do
  IFS='|' read -r DS SEED TAG DATA_PATH <<< "$spec"
  want_tag "$TAG" || continue

  echo "=================================================================="
  echo "[$(date '+%F %T')] DS=$DS seed=$SEED tag=$TAG"
  echo "  data_path = $DATA_PATH"
  echo "=================================================================="

  [ -f "$DATA_PATH" ] || { echo "✗ 缺输入 $DATA_PATH"; exit 1; }

  RUN_DIR="$(find_done_run "$TAG")"

  # ---------- A. 训练 ----------
  if [ -n "$RUN_DIR" ]; then
    echo "[$(date '+%F %T')] [A] 跳过训练（已存在 4 份 traj ckpt）: $RUN_DIR"
  else
    echo "[$(date '+%F %T')] [A] 训练 RQ-VAE（epochs=3000, traj_epochs=${EPOCHS[*]}）"
    "$PY" rqvae/run_checkpoint_trajectory.py \
      --data_path "$DATA_PATH" \
      --ckpt_dir "$CKPT_ROOT/$TAG" \
      --seed "$SEED" \
      --traj_epochs "${EPOCHS[@]}" \
      > "$LOG_ROOT/${TAG}_train.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "✗ 训练失败 rc=$rc（日志 $LOG_ROOT/${TAG}_train.log）"; exit 1; }
    RUN_DIR="$(find_done_run "$TAG")"
    [ -n "$RUN_DIR" ] || { echo "✗ 训练结束但找不到 4 份 traj ckpt"; exit 1; }
    echo "[$(date '+%F %T')] [A] 完成: $RUN_DIR"
  fi

  # ---------- B. codegen ----------
  for e in "${EPOCHS[@]}"; do
    CK="$RUN_DIR$(epfile "$e")"
    OUT="$CODE_ROOT/${TAG}_$(epname "$e")_sid.npy"
    [ -f "$CK" ] || { echo "✗ 缺 ckpt $CK"; exit 1; }
    if [ -f "$OUT" ]; then
      echo "[$(date '+%F %T')] [B] 跳过 codegen: $(basename "$OUT")"
      continue
    fi
    echo "[$(date '+%F %T')] [B] codegen ep$e → $OUT"
    "$PY" rqvae/generate_sid_codes_param.py \
      --ckpt_path "$CK" --data_path "$DATA_PATH" \
      --output_file "$OUT" --device cuda:0 \
      > "$LOG_ROOT/${TAG}_$(epname "$e")_codegen.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "✗ codegen 失败 rc=$rc（日志 $LOG_ROOT/${TAG}_$(epname "$e")_codegen.log）"; exit 1; }
  done

  # ---------- C. manifest 片段（每 tag 一个文件，无共享写 → 并发安全）----------
  # 列：tag  ds  seed  data_path  run_dir
  printf "%s\t%s\t%s\t%s\t%s\n" "$TAG" "$DS" "$SEED" "$DATA_PATH" "$RUN_DIR" \
    > "$MANIFEST_DIR/${TAG}.tsv"
  echo "[$(date '+%F %T')] ✅ $TAG 完成"
done

echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE"
echo "  ckpt     : $CKPT_ROOT"
echo "  codes    : $CODE_ROOT"
echo "  manifest : $MANIFEST_DIR/*.tsv"
echo "=================================================================="
