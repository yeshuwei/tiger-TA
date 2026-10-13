#!/usr/bin/env bash
# run_b2_full.sh — E50 step 10：b2 臂正式实验（3 数据集 × 3 T5-seed × α*=0.02）
#
# 预注册依据：doc/exp/ab_decision_and_design.md（§3.5 α 标定已裁 α*=0.02；§3.6 执行顺序第 10 步）
# 标定结论：doc/e50_b2_alpha_conclusion.md
#
# 协议（与主表逐位一致，2026-09-30 用户裁定）：
#   **一次 RQ-VAE（main.py 默认 seed=2024，每数据集只训一次、只出一份 codes.npy）
#     → T5 用 seed 2025/2026/2027 各训一次。**
#   b0/b4/b6 每个数据集同样只有一份 codes，故 b2 必须同构，否则不可比。
#
# 固定配置：
#   输入     data/{D}/item_emb_768d_contentonly.parquet   (768d)
#   对齐目标 data/{D}/behavior_emb.parquet                (128d)
#   α* = 0.02（冻结，不再调）；τ = 0.1
#   RQ-VAE   --num_emb_list 256 256 256，其余全默认
#   T5       model/main.py 默认（200 ep / early_stop 10 / beam 30），--seed 2025|2026|2027
#   eval     --beam_size 30 --topk_list 5,10,20,50,100
#
# 门禁（§4.2-E，前置）：RQ-VAE 训完先查 L0 unique 用码率 ≥ b4 的 90%（逐数据集从
#   b4 codes 现算，不硬编码）。不达标 ⇒ 记码本塌缩、**跳过该数据集全部 T5**，省 ~1.5 h/数据集。
#
# 幂等：每 stage `[ ! -e ... ]` 才执行。**已有产物一律跳过、绝不覆盖。**
#   Sport × seed2025 是 α 标定产物 ⇒ 自动跳过。
#
# ⚠️ 取数纪律（见 memory `eval-log-ndcg-column-trap`）：eval log 的 NDCG 行列数 =
#   len(--topk_list)。必须按**表头**定位 @20，**禁用 `$NF`**（5 列下会取到 @100）。
#
# 用法:
#   bash rqvae/run_b2_full.sh                 # 全域 3 数据集 × 3 seed
#   bash rqvae/run_b2_full.sh Toys Beauty     # 只跑指定数据集（补跑用）
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

ALPHA=0.02
TAG="b2align${ALPHA}"
T5_SEEDS=(2025 2026 2027)
SUMMARY="rqvae/logs/e50_b2_summary.tsv"

if [ $# -ge 1 ]; then DATASETS=("$@"); else DATASETS=(Sport Toys Beauty); fi

mkdir -p rqvae/logs rqvae/ckpt model/logs model/ckpt
if [ ! -f "$SUMMARY" ]; then
  printf "dataset\tseed\trqvae_ckpt\tL0_unique\tb4_L0\tgate\tval_ndcg20\ttest_ndcg5\ttest_ndcg10\ttest_ndcg20\teval_log\n" > "$SUMMARY"
fi

echo "=================================================================="
echo "[$(date '+%F %T')] E50 step 10 — b2 正式  α*=${ALPHA}（冻结）"
echo "  数据集: ${DATASETS[*]}   T5 seeds: ${T5_SEEDS[*]}"
echo "  RQ-VAE : 每数据集 1 次（默认 seed=2024）"
echo "  汇总   : $SUMMARY"
echo "=================================================================="

for DS in "${DATASETS[@]}"; do
  DSL="$(echo "$DS" | tr '[:upper:]' '[:lower:]')"
  CONTENT="data/${DS}/item_emb_768d_contentonly.parquet"
  ALIGN="data/${DS}/behavior_emb.parquet"
  CKPT_DIR="rqvae/ckpt/${DS}_B2_align${ALPHA}"
  CODES="data/${DS}/${DS}_${TAG}_sid.npy"
  RQVAE_LOG="rqvae/logs/e50_${DSL}_${TAG}_rqvae.log"
  B4_CODES="data/${DS}/${DS}_content768d_baseline_sid.npy"

  echo
  echo "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"
  echo "[$(date '+%F %T')] 数据集 = $DS"
  echo "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"

  if [ ! -f "$CONTENT" ]; then echo "✗ 缺 content-only 输入: $CONTENT  ⇒ 跳过 $DS"; continue; fi
  if [ ! -f "$ALIGN" ];   then echo "✗ 缺对齐目标: $ALIGN  ⇒ 跳过 $DS"; continue; fi
  if [ ! -f "$B4_CODES" ]; then
    echo "✗ 缺 b4 参照 codes: $B4_CODES（门禁门限无法现算）⇒ 跳过 $DS"; continue
  fi

  # ---------- A. RQ-VAE（每数据集一次，默认 seed=2024）----------
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[$(date '+%F %T')] [A] 训练 RQ-VAE (seed=2024) align_weight=$ALPHA"
    python rqvae/main.py \
      --data_path "$CONTENT" \
      --ckpt_dir "$CKPT_DIR" \
      --num_emb_list 256 256 256 \
      --align_weight "$ALPHA" \
      --align_data_path "$ALIGN" \
      2>&1 | tee "$RQVAE_LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ RQ-VAE 训练失败 rc=$rc ($DS)"; exit 1; }
  else
    echo "[$(date '+%F %T')] [A] 跳过：$CKPT_DIR 已存在"
  fi

  CKPT_SUB="$(ls -d ${CKPT_DIR}/*/ 2>/dev/null | tail -1)"
  CKPT="${CKPT_SUB}best_collision_model.pth"
  [ -f "$CKPT" ] || { echo "✗ 缺 ckpt: $CKPT"; exit 1; }

  # 梯度断链哨兵：L_align 必须显著下降（断链时会卡在 ln(batch)≈6.9 附近）
  python3 - "$RQVAE_LOG" <<'PY' || true
import re, sys
v = [float(x) for x in re.findall(r"L_align=([0-9.]+)", open(sys.argv[1]).read())]
if v:
    print(f"    L_align 轨迹: 首={v[0]:.4f}  末={v[-1]:.4f}  "
          f"{'✓ 在收敛' if v[-1] < v[0] - 0.5 else '⚠️ 几乎未动，疑似梯度断链'}")
PY

  # ---------- B. codegen ----------
  if [ ! -f "$CODES" ]; then
    echo "[$(date '+%F %T')] [B] codegen -> $CODES"
    python rqvae/generate_sid_codes_param.py \
      --ckpt_path "$CKPT" --data_path "$CONTENT" --output_file "$CODES" \
      --device cuda:0 2>&1 | tee -a "$RQVAE_LOG"
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && { echo "✗ codegen 失败 ($DS)"; exit 1; }
  else
    echo "[$(date '+%F %T')] [B] 跳过：$CODES 已存在"
  fi

  # ---------- C. 门禁：L0 unique ≥ b4 的 90%（逐数据集现算门限）----------
  read -r L0U B4L0 GATE < <(python3 - "$CODES" "$B4_CODES" <<'PY'
import numpy as np, sys
l0 = lambda f: len(np.unique(np.load(f)[:, 0]))
me, b4 = l0(sys.argv[1]), l0(sys.argv[2])
gate = int(b4 * 0.9)          # 向下取整，留一点余量
print(me, b4, gate)
PY
)
  echo "[$(date '+%F %T')] [C] 码本健康: L0 unique=${L0U}/256  (b4=${B4L0}, gate≥${GATE})"
  if [ "$L0U" -lt "$GATE" ]; then
    echo "    ✗ 门禁不过 ⇒ 记码本塌缩，**跳过 $DS 的全部 T5**"
    for S in "${T5_SEEDS[@]}"; do
      printf "%s\t%s\t%s\t%s\t%s\tFAIL\tNA\tNA\tNA\tNA\tcollapse\n" \
        "$DS" "$S" "$(basename "$CKPT_SUB")" "$L0U" "$B4L0" >> "$SUMMARY"
    done
    continue
  fi
  echo "    ✓ 门禁通过"

  # ---------- D/E/F. 逐 T5 seed ----------
  for S in "${T5_SEEDS[@]}"; do
    T5_CKPT="model/ckpt/tiger_${TAG}_${DSL}_seed${S}.pth"
    T5_LOG="model/logs/tiger_${TAG}_${DSL}_seed${S}.log"
    T5_TEE="model/logs/${DS}_${TAG}_seed${S}_t5_train.log"
    EVAL_LOG="model/logs/${DS}_${TAG}_seed${S}_eval.log"

    echo
    echo "  ──────────────────────────────────────────────────────────"
    echo "  [$(date '+%F %T')] $DS  seed=$S"
    echo "  ──────────────────────────────────────────────────────────"

    # D. T5 训练
    if [ ! -f "$T5_LOG" ]; then
      echo "  [$(date '+%F %T')] [D] T5 训练 -> $T5_CKPT"
      python model/main.py \
        --dataset_path "data/${DS}" \
        --code_path "$CODES" \
        --save_path "$T5_CKPT" \
        --log_path "$T5_LOG" \
        --seed "$S" 2>&1 | tee "$T5_TEE"
      rc=${PIPESTATUS[0]}
      [ $rc -ne 0 ] && { echo "✗ T5 训练失败 rc=$rc ($DS seed=$S)"; exit 1; }
    else
      echo "  [$(date '+%F %T')] [D] 跳过：$T5_LOG 已存在"
    fi

    # E. test 评测
    if [ ! -f "$EVAL_LOG" ]; then
      echo "  [$(date '+%F %T')] [E] test 评测 -> $EVAL_LOG"
      python3 tools/eval_trained_model.py \
        --dataset "$DS" --ckpt_path "$T5_CKPT" --code_path "$CODES" \
        --beam_size 30 --topk_list 5,10,20,50,100 --seed "$S" \
        2>&1 | tee "$EVAL_LOG"
    else
      echo "  [$(date '+%F %T')] [E] 跳过：$EVAL_LOG 已存在"
    fi

    # F. 取数：val（决策量，T5 日志 max） + test（eval log，**按表头定位**）
    read -r VAL T5V T10V T20 < <(python3 - "$T5_LOG" "$EVAL_LOG" <<'PY'
import re, sys
# --- val: T5 训练日志里的 'NDCG@20': x，取 max（= best-val checkpoint）---
s = open(sys.argv[1], errors='replace').read()
v = [float(x) for x in re.findall(r"'NDCG@20':\s*([0-9.eE+-]+)", s)]
val = f"{max(v):.17g}" if v else "NA"

# --- test: eval log，先抓表头 @k 序列，再按列号取，**绝不用 $NF** ---
out = {"5": "NA", "10": "NA", "20": "NA"}
lines = open(sys.argv[2], errors='replace').read().splitlines()
hdr = None
for i, ln in enumerate(lines):
    if ln.strip().startswith("Metric") and "@" in ln:
        hdr = [t for t in ln.split() if t.startswith("@")]
        # NDCG 行是表头之后第一行以 NDCG 开头的
        for l2 in lines[i + 1:]:
            if l2.strip().startswith("NDCG"):
                cells = l2.split()[1:]          # 去掉 "NDCG" 标签
                for k, c in zip(hdr, cells):
                    kk = k.lstrip("@")
                    if kk in out:
                        out[kk] = c
                break
        break
if hdr is None:
    print(val, "NA", "NA", "NA")
else:
    print(val, out["5"], out["10"], out["20"])
PY
)
    echo "  [$(date '+%F %T')] [F] $DS seed=$S  val NDCG@20=$VAL   test @5/@10/@20 = $T5V / $T10V / $T20"
    printf "%s\t%s\t%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\t%s\n" \
      "$DS" "$S" "$(basename "$CKPT_SUB")" "$L0U" "$B4L0" "$VAL" "$T5V" "$T10V" "$T20" "$EVAL_LOG" >> "$SUMMARY"
  done
done

# ---------- G. 汇总 ----------
echo
echo "=================================================================="
echo "[$(date '+%F %T')] E50 step 10 汇总  ($SUMMARY)"
echo "=================================================================="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"

python3 - "$SUMMARY" <<'PY'
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')]
by = collections.defaultdict(list)
for r in rows:
    if r['gate'] == 'PASS' and r['test_ndcg20'] not in ('NA', ''):
        by[r['dataset']].append(float(r['test_ndcg20']))
print("\n--- test NDCG@20 每数据集 mean ± std (ddof=1) ---")
for d in ('Sport', 'Toys', 'Beauty'):
    v = by.get(d)
    if not v:
        print(f"  {d:8s} 无有效行")
        continue
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    print(f"  {d:8s} n={len(v)}  {m:.4f} ± {sd:.4f}   " + " ".join(f"{x:.4f}" for x in v))
print("\n判据（§3.3 / 问题 D 的两增量写法）：")
print("  b0−b4 = 输入级注入效应（配 b6 安慰剂）")
print("  b2−b4 = 损失级注入效应   ← 本实验直接给出")
print("  ⚠️ 不得用 b0 vs b2 作单变量结论（同时差输入与损失两个因素）")
PY
echo "=================================================================="
echo "[$(date '+%F %T')] ALL DONE (E50 step 10 — b2 正式)"
echo "=================================================================="
