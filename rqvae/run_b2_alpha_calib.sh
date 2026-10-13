#!/usr/bin/env bash
# run_b2_alpha_calib.sh — E50 b2 α 标定（Sport × T5-seed2025 × α∈{0.002,0.02,0.2}）
#
# 目的：按预注册规则（doc/exp/ab_decision_and_design.md §3.5）在**下游 val NDCG@20** 上选 α*，
#   选完冻结，供 step 10（三数据集 × 三 seed）正式跑使用。
#   **禁用潜空间诊断**选 α —— E49 已证量化端诊断不预测 NDCG@20。
#
# 协议（与主表逐位一致）：RQ-VAE 用 main.py 默认 seed=2024；T5 用 seed=2025。
#   b4(Content-only 768d) 同协议参照 = val NDCG@20 0.03226418297855821
#   （源 model/logs/tiger_content768d_baseline_sport.log，best 行非末行）。
#
# 预注册裁决规则（§3.5 第 3 条）：
#   若三个 α 的下游 val NDCG@20 **全部 ≤ b4** ⇒ 判 α*=0、b2 为负结果，
#   结论「损失级注入无增量」，**不必再跑其余数据集**。
#
# 码本健康门禁（§4.2-E，前置、只花 7 min）：RQ-VAE 训完先查 L0 unique 用码率，
#   < b4 的 90%（151×0.9=135）即记码本塌缩，**不进 T5**（省下 70 min/路）。
#
# 幂等：每 stage `[ ! -e ... ]` 才执行；已有产物一律跳过、绝不覆盖。
#
# 用法:
#   bash rqvae/run_b2_alpha_calib.sh              # 全量三个 α
#   bash rqvae/run_b2_alpha_calib.sh 0.02         # 只跑指定 α（补跑用）
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DS=Sport
DSL=sport
CONTENT="data/${DS}/item_emb_768d_contentonly.parquet"
ALIGN="data/${DS}/behavior_emb.parquet"
T5_SEED=2025

B4_VAL=0.03226418297855821   # b4 val NDCG@20（同协议）
B4_L0=151                    # b4 Sport L0 unique 用码数（满分 256）
GATE_MIN=135                 # int(151*0.9)

SUMMARY="rqvae/logs/e50_b2_alpha_summary.tsv"

[ -f "$CONTENT" ] || { echo "✗ 缺 content-only 输入: $CONTENT"; exit 1; }
[ -f "$ALIGN" ]   || { echo "✗ 缺对齐目标: $ALIGN"; exit 1; }

mkdir -p rqvae/logs rqvae/ckpt model/logs model/ckpt
if [ ! -f "$SUMMARY" ]; then
  printf "alpha\trqvae_epoch\tL0_unique\tgate\tval_ndcg20\ttest_ndcg5\ttest_ndcg10\ttest_ndcg20\tverdict\n" > "$SUMMARY"
fi

if [ $# -ge 1 ]; then ALPHAS=("$@"); else ALPHAS=(0.002 0.02 0.2); fi

echo "=================================================================="
echo "[$(date '+%F %T')] E50 b2 α 标定  DS=${DS}  T5_seed=${T5_SEED}"
echo "  α ∈ ${ALPHAS[*]}"
echo "  输入   : $CONTENT (768d, content-only)"
echo "  对齐目标: $ALIGN (128d behavior)"
echo "  b4 参照: val NDCG@20 = $B4_VAL   L0 unique = ${B4_L0}/256   gate_min = $GATE_MIN"
echo "=================================================================="

for A in "${ALPHAS[@]}"; do
  TAG="b2align${A}"
  CKPT_DIR="rqvae/ckpt/${DS}_B2_align${A}"
  CODES="data/${DS}/${DS}_${TAG}_sid.npy"
  RQVAE_LOG="rqvae/logs/e50_${DSL}_${TAG}_rqvae.log"
  T5_CKPT="model/ckpt/tiger_${TAG}_${DSL}_seed${T5_SEED}.pth"
  T5_LOG="model/logs/tiger_${TAG}_${DSL}_seed${T5_SEED}.log"
  EVAL_LOG="model/logs/${DS}_${TAG}_seed${T5_SEED}_eval.log"

  echo
  echo "──────────────────────────────────────────────────────────────"
  echo "[$(date '+%F %T')] α = $A   tag = $TAG"
  echo "──────────────────────────────────────────────────────────────"

  # ---------- A. 训练 RQ-VAE（content-only 768d 输入 + L_align）----------
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] [A] 训练 RQ-VAE  seed=2024  align_weight=$A"
    python rqvae/main.py \
      --data_path "$CONTENT" \
      --ckpt_dir "$CKPT_DIR" \
      --num_emb_list 256 256 256 \
      --align_weight "$A" \
      --align_data_path "$ALIGN" \
      2>&1 | tee "$RQVAE_LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ RQ-VAE 训练失败 rc=$rc (α=$A)"; exit 1; }
  else
    echo "[$(date '+%F %T')] [A] 跳过：$CKPT_DIR 已存在"
  fi

  CKPT_SUB="$(ls -d ${CKPT_DIR}/*/ 2>/dev/null | tail -1)"
  CKPT="${CKPT_SUB}best_collision_model.pth"
  [ -f "$CKPT" ] || { echo "✗ 缺 ckpt: $CKPT"; exit 1; }
  echo "    ckpt = $CKPT"
  # 记录 L_align 是否真的在收敛（防梯度断链复发）：首末两次 L_align。
  # 断链时 L_align 会卡在 ln(batch)≈6.9 附近不动；正常应单调下降到 4~5。
  python3 - "$RQVAE_LOG" <<'PY' || true
import re, sys
v = [float(x) for x in re.findall(r"L_align=([0-9.]+)", open(sys.argv[1]).read())]
if v:
    print(f"    L_align 轨迹: 首={v[0]:.4f}  末={v[-1]:.4f}  "
          f"{'✓ 在收敛' if v[-1] < v[0] - 0.5 else '⚠️ 几乎未动，疑似梯度断链'}")
PY

  # ---------- B. 生成 SID codes ----------
  if [ ! -f "$CODES" ]; then
    echo "[$(date '+%F %T')] [B] codegen -> $CODES"
    python rqvae/generate_sid_codes_param.py \
      --ckpt_path "$CKPT" --data_path "$CONTENT" --output_file "$CODES" \
      --device cuda:0 2>&1 | tee -a "$RQVAE_LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ codegen 失败 (α=$A)"; exit 1; }
  else
    echo "[$(date '+%F %T')] [B] 跳过：$CODES 已存在"
  fi

  # ---------- C. 码本健康门禁（§4.2-E）----------
  read -r L0U COMBO < <(python3 - "$CODES" <<'PY'
import numpy as np, sys
c = np.load(sys.argv[1])
print(len(np.unique(c[:, 0])), len(np.unique(c[:, :3], axis=0)))
PY
)
  echo "[$(date '+%F %T')] [C] 码本健康: L0 unique=${L0U}/256 (b4=${B4_L0}, gate_min=${GATE_MIN})  组合唯一=${COMBO}"
  if [ "$L0U" -lt "$GATE_MIN" ]; then
    echo "    ✗ 门禁不过：L0 用码率 < b4 的 90% ⇒ 记码本塌缩，**跳过 T5**"
    printf "%s\t%s\t%s\tFAIL\tNA\tNA\tNA\tNA\tcollapse\n" "$A" "$(basename "$CKPT_SUB")" "$L0U" >> "$SUMMARY"
    continue
  fi
  echo "    ✓ 门禁通过"

  # ---------- D. T5 训练（seed 2025）----------
  if [ ! -f "$T5_LOG" ]; then
    echo "[$(date '+%F %T')] [D] T5 训练 seed=${T5_SEED} -> $T5_CKPT"
    python model/main.py \
      --dataset_path "data/${DS}" \
      --code_path "$CODES" \
      --save_path "$T5_CKPT" \
      --log_path "$T5_LOG" \
      --seed "$T5_SEED" 2>&1 | tee "model/logs/${DS}_${TAG}_seed${T5_SEED}_t5_train.log"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ T5 训练失败 rc=$rc (α=$A)"; exit 1; }
  else
    echo "[$(date '+%F %T')] [D] 跳过：$T5_LOG 已存在"
  fi

  # ---------- E. test 评测 ----------
  if [ ! -f "$EVAL_LOG" ]; then
    echo "[$(date '+%F %T')] [E] test 评测 -> $EVAL_LOG"
    python3 tools/eval_trained_model.py \
      --dataset "$DS" --ckpt_path "$T5_CKPT" --code_path "$CODES" \
      --beam_size 30 --topk_list 5,10,20,50,100 --seed "$T5_SEED" \
      2>&1 | tee "$EVAL_LOG"
  else
    echo "[$(date '+%F %T')] [E] 跳过：$EVAL_LOG 已存在"
  fi

  # ---------- F. 取 best val NDCG@20（决策量）----------
  VAL="$(python3 - "$T5_LOG" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
v = [float(x) for x in re.findall(r"'NDCG@20':\s*([0-9.eE+-]+)", s)]
print(f"{max(v):.17g}" if v else "NA")
PY
)"
  T20="$(grep -E '^\s*NDCG' "$EVAL_LOG" 2>/dev/null | tail -1 | awk '{print $NF}')"
  T5V="$(grep -E '^\s*NDCG' "$EVAL_LOG" 2>/dev/null | tail -1 | awk '{print $(NF-2)}')"
  T10V="$(grep -E '^\s*NDCG' "$EVAL_LOG" 2>/dev/null | tail -1 | awk '{print $(NF-1)}')"

  VERDICT="$(python3 -c "print('>=b4' if float('$VAL')>$B4_VAL else '<b4')" 2>/dev/null || echo '?')"
  echo "[$(date '+%F %T')] [F] α=$A  val NDCG@20 = $VAL   (b4 = $B4_VAL)  ⇒ $VERDICT"
  echo "        test NDCG @5/@10/@20 = $T5V / $T10V / $T20"
  printf "%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\t%s\n" "$A" "$(basename "$CKPT_SUB")" "$L0U" "$VAL" "$T5V" "$T10V" "$T20" "$VERDICT" >> "$SUMMARY"
done

# ---------- G. 汇总 ----------
echo
echo "=================================================================="
echo "[$(date '+%F %T')] α 标定汇总  ($SUMMARY)"
echo "=================================================================="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

python3 - "$SUMMARY" "$B4_VAL" <<'PY'
import sys, csv
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')]
b4 = float(sys.argv[2])
ok = [r for r in rows if r['gate'] == 'PASS' and r['val_ndcg20'] != 'NA']
if not ok:
    print("\n⇒ 全部 α 未过门禁：b2 记码本塌缩")
elif max(float(r['val_ndcg20']) for r in ok) <= b4:
    print(f"\n⇒ 预注册裁决：三个 α 全部 ≤ b4({b4:.6f}) ⇒ α*=0，b2 为负结果，不再跑其余数据集")
else:
    b = max(ok, key=lambda r: float(r['val_ndcg20']))
    print(f"\n⇒ α* = {b['alpha']}（val NDCG@20 = {b['val_ndcg20']}, vs b4 {b4:.6f}）→ 冻结，进入 step 10 全量")
PY
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (E50 b2 α 标定)"
echo "=================================================================="
