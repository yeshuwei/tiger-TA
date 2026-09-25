#!/usr/bin/env bash
# run_e49_p3_p6.sh — E49 P3→P6 无人值守链（含 P2 决策门守卫）
#
# 设计目的：P2 跑完后若判据通过，自动接 P3/P4/P5/P6，无需人工盯守；
#           若判据不通过（管线坏 / 变体不可分），**主动停下**并留下证据，
#           不浪费后面 ~20h 的 GPU。
#
# 用法:
#   nohup bash rqvae/run_e49_p3_p6.sh > rqvae/logs/e49_p3_p6_bg.log 2>&1 &
#
# 幂等：底层 run_behavior_ablation.sh 逐 stage skip-if-exists，可随时中断续跑。
# 纪律：不重命名、不复制既有产物；仅调用既有脚本（见 doc/e49_behavior_ablation.md §7）。
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ABL="rqvae/run_behavior_ablation.sh"
GATE_DOC="doc/e49_p2_gate_verdict.md"

log() { echo "[$(date '+%F %T')] $*"; }
hr()  { echo "=================================================================="; }

# 读取评测日志的 NDCG@20；文件不存在或表格尚未写完则输出 NA。
# ⚠️ 必须用「可解析」而非「文件存在」判就绪：tee 在评测一开始就创建文件，
#    而 NDCG 表格要到评测结束才写入 —— 只判存在会导致门读到空文件而误判失败
#    （2026-09-22 18:33 实际发生过一次，白白停链）。
metric() {
  python - "$1" <<'PY'
import os, re, sys, time
p = sys.argv[1]
if not os.path.exists(p):
    print("NA"); raise SystemExit
m = re.findall(r"^\s*NDCG\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", open(p).read(), re.MULTILINE)
print(f"{float(m[-1][2]):.6f}" if m else "NA")
PY
}

hr; log "E49 P3→P6 无人值守链启动"; hr

# ---------------------------------------------------------------- 等待 P2 收尾
# 必须等 P2 链（另一个后台进程）把 itemcf 也跑完再动 GPU：
#   ① 避免两个 RQ-VAE/T5 抢同一张卡；② 决策门需要 itemcf 的数字。
# P2 链的顺序是 `... rawcooc 2025 && ... itemcf 2025`，故 itemcf 的 eval 日志出现
# 即等价于 P2 链已结束（它是最后一步）。用文件而非 pid，抗重启。
WAIT_MAX=480          # 最长等待 8h（实测单变体全链路 ~86min，留足余量）
i=0
while [ $i -lt $WAIT_MAX ]; do
  R=$(metric model/logs/Sport_rawcooc_768d_seed2025_eval.log)
  I=$(metric model/logs/Sport_itemcf_768d_seed2025_eval.log)
  if [ "$R" != "NA" ] && [ "$I" != "NA" ]; then
    log "[WAIT] P2 两变体 eval 均已【写完】，继续 (rawcooc=$R itemcf=$I)"
    break
  fi
  [ $((i % 15)) -eq 0 ] && log "[WAIT] 等待 P2 产物... (${i}min) rawcooc=${R} itemcf=${I}"
  sleep 60; i=$((i+1))
done
if [ $i -ge $WAIT_MAX ]; then
  hr; log "✗ 等待超时（${WAIT_MAX}min）：P2 产物未齐 → 链路停止，请人工排查 P2"; hr
  exit 1
fi

# ---------------------------------------------------------------- P2 决策门
# 判据（doc/e49_behavior_ablation.md §6）:
#   ① rawcooc / itemcf 的 Sport seed2025 NDCG@20 均落在合理带内（非 0.001 / 非 0.06 量级）
#   ② 两变体互相可分（不几乎重合）
log "[GATE] 检查 P2 决策门 ..."
python - <<'PY'
import os, re, sys, time

BAND = (0.0150, 0.0450)          # 合理带：content-only~TA-SID 邻域，防管线坏值
MIN_SEP = 0.0005                 # 两变体最小可分差距（NDCG@20 绝对值）

def parse(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    m = re.findall(r"^\s*NDCG\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", txt, re.MULTILINE)
    return float(m[-1][2]) if m else None

def parse_ref(p):
    return parse(p)

rows = {}
for tag, p in [("rawcooc", "model/logs/Sport_rawcooc_768d_seed2025_eval.log"),
               ("itemcf",  "model/logs/Sport_itemcf_768d_seed2025_eval.log")]:
    v = parse(p)
    # 纵深防御：文件已存在但表格未写完时，重试等待，而不是立刻判"训练失败"
    tries = 0
    while v is None and os.path.exists(p) and tries < 20:
        time.sleep(15); tries += 1
        v = parse(p)
    if tries:
        print(f"  (itemcf/rawcooc {tag}: 等待评测写完，重试 {tries} 次)")
    rows[tag] = v

refs = {
    "content-only": parse_ref("model/logs/Sport_content768d_baseline_seed2025_eval.log"),
    "TA-SID":       parse_ref("model/logs/Sport_ta_sid_768d_seed2025_eval.log"),
}

print("  ── P2 决策门证据（Sport seed2025, NDCG@20）──")
for k, v in refs.items():
    print(f"    {k:14s} {v if v is None else f'{v:.4f}'}  (参照)")
for k, v in rows.items():
    print(f"    {k:14s} {v if v is None else f'{v:.4f}'}  (新变体)")

fails = []
missing = [k for k, v in rows.items() if v is None]
if missing:
    fails.append(f"缺少评测结果: {', '.join(missing)}（训练可能未完成或失败）")
else:
    for k, v in rows.items():
        if not (BAND[0] <= v <= BAND[1]):
            fails.append(f"{k} NDCG@20={v:.4f} 落在合理带 {BAND} 之外 → 疑似管线故障（codegen/融合/评测）")
    sep = abs(rows["rawcooc"] - rows["itemcf"])
    if sep < MIN_SEP:
        fails.append(f"两变体几乎重合（Δ={sep:.4f} < {MIN_SEP}）→ 作用域/归一化在该数据上不起作用，需重审设计")

# 与行为空间诊断的对照预期（仅记录，不作判据）
if not missing and all(v is not None for v in refs.values()):
    order = sorted(rows.items(), key=lambda kv: -kv[1])
    print(f"  ── 排序: {' > '.join(f'{k}({v:.4f})' for k, v in order)} ──")
    print(f"     行为空间 cons_auc_local 预期序: TA-SID > Raw-CoOcc > ItemCF")

# 写判定文档
os.makedirs("doc", exist_ok=True)
lines = ["# E49 P2 决策门判定", "",
         f"**生成时间**：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}　"
         "**自动生成**：`rqvae/run_e49_p3_p6.sh`（勿手改，需改写则改脚本后重跑）", "",
         "## 证据（Sport seed2025，NDCG@20，同口径 `tools/eval_trained_model.py --beam_size 30`）", "",
         "| 行 | NDCG@20 | seed 数 |", "|:--|:--|:--|"]
for k, v in refs.items():
    lines.append(f"| {k}（参照） | {v if v is None else f'{v:.4f}'} | 1 |")
for k, v in rows.items():
    lines.append(f"| **{k}（E49 新增）** | {v if v is None else f'{v:.4f}'} | 1 |")
lines += ["", "## 判据", "",
          f"1. 两新变体 NDCG@20 ∈ {BAND}",
          f"2. 两变体可分：|Δ| ≥ {MIN_SEP}", "",
          "## 判定", ""]
if fails:
    lines.append("**❌ 未通过 — 链路已在此停止，未进入 P3**")
    lines.append("")
    for f in fails:
        lines.append(f"- {f}")
    lines += ["", "→ 须人工排查后再决定是否续跑。"]
else:
    lines.append("**✅ 通过 — 自动续跑 P3/P4/P5/P6**")
lines.append("")
open("doc/e49_p2_gate_verdict.md", "w").write("\n".join(lines))

if fails:
    print("\n  ❌ GATE FAIL:")
    for f in fails:
        print(f"     - {f}")
    sys.exit(1)
print("\n  ✅ GATE PASS")
PY
GATE_RC=$?
if [ $GATE_RC -ne 0 ]; then
  hr; log "✗ P2 决策门未通过 → 链路停止，不进入 P3。请查看 $GATE_DOC"; hr
  exit 1
fi
log "✅ P2 决策门通过 → 续跑 P3/P4/P5/P6（判定见 $GATE_DOC）"

# ---------------------------------------------------------------- 逐阶段执行
run() {   # $*: run_behavior_ablation.sh 的参数
  hr; log "▶ bash $ABL $*"; hr
  bash "$ABL" "$@" || { log "✗ 失败: $ABL $*  → 链路停止"; exit 1; }
}

# --- P2 itemcf（若 P2 的 && 链已跑完 itemcf，此处会自动全部跳过）---
log "[P2 收尾] 确保 Sport itemcf seed2025 完成"
run Sport itemcf 2025

# --- P3: Sport 两新变体补齐 3-seed ---
log "[P3] Sport rawcooc/itemcf seeds 2026 2027"
run Sport rawcooc 2026 2027
run Sport itemcf  2026 2027

# --- P4: Toys 两新变体全 3-seed（含各自 RQ-VAE）---
log "[P4] Toys rawcooc/itemcf 全 3-seed"
run Toys rawcooc 2025 2026 2027
run Toys itemcf  2025 2026 2027

# --- P5: backfill 已有对照行到 3-seed ---
log "[P5-a] Sport Shuffled/ZeroPad 补 T5 seeds 2026 2027（复用 legacy codes，不重跑 RQ-VAE）"
run Sport shuffled 2026 2027
run Sport zeropad  2026 2027

log "[P5-b] Toys Shuffled 全链路 seeds 2025 2026 2027（behavior/fused 已在盘）"
run Toys shuffled 2025 2026 2027

log "[P5-c] Toys ZeroPad 只补 T5 seeds 2025 2026 2027（复用 Toys_zeropad_seed2024_sid.npy）"
run Toys t5-only Toys_zeropad_seed2024_sid.npy zeropad_ablation 2025 2026 2027

# ---------------------------------------------------------------- P6: 汇总
log "[P6] 汇总所有 E49 评测结果 → doc/e49_behavior_ablation_results.md"
python - <<'PY'
import os, re, glob, statistics as st

REF = {"Sport": {"content-only": "Sport_content768d_baseline_seed2025_eval.log",
                 "TA-SID":       "Sport_ta_sid_768d_seed2025_eval.log",
                 "Shuffled":     "Sport_shuffled_ablation_eval.log",
                 "ZeroPad":      "Sport_zeropad_ablation_eval.log"},
       "Toys":  {"content-only": "Toys_content768d_baseline_seed2025_eval.log",
                 "TA-SID":       "Toys_ta_sid_768d_seed2025_eval.log"}}

def parse(p):
    if not os.path.exists(p):
        return None
    m = re.findall(r"^\s*NDCG\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", open(p).read(), re.MULTILINE)
    return float(m[-1][2]) if m else None

def collect(pattern):
    out = {}
    for p in sorted(glob.glob(f"model/logs/{pattern}")):
        m = re.search(r"seed(\d+)_eval\.log$", os.path.basename(p))
        if m and parse(p) is not None:
            out[int(m.group(1))] = parse(p)
    return out

VARIANTS = [("content-only", "Sport_content768d_baseline"), ("TA-SID",       "Sport_ta_sid_768d"),
            ("Raw-CoOcc",    "Sport_rawcooc_768d"),          ("ItemCF",       "Sport_itemcf_768d"),
            ("Shuffled",     "Sport_shuffled_ablation"),     ("ZeroPad",      "Sport_zeropad_ablation"),
            ("content-only", "Toys_content768d_baseline"),   ("TA-SID",       "Toys_ta_sid_768d"),
            ("Raw-CoOcc",    "Toys_rawcooc_768d"),           ("ItemCF",       "Toys_itemcf_768d"),
            ("Shuffled",     "Toys_shuffled_ablation"),      ("ZeroPad",      "Toys_zeropad_ablation")]
# legacy 单 seed 名（无 seed 后缀）作为 seed2025 来源之一
LEGACY_2025 = {"Sport_shuffled_ablation": "Sport_shuffled_ablation_eval.log",
               "Sport_zeropad_ablation":  "Sport_zeropad_ablation_eval.log",
               "Toys_zeropad_ablation":   "Toys_zeropad_ablation_eval.log"}

lines = ["# E49 消融结果汇总（自动生成）", "",
         f"**生成时间**：{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}　"
         "由 `rqvae/run_e49_p3_p6.sh` P6 段产出；数字经 `tools/extract_metrics.py` 同源解析器读取。", "",
         "指标：NDCG@20（beam_size=30）。3-seed 行报 mean±std；缺 seed 的行如实标注。", "",
         "| 数据集 | 变体 | NDCG@20 (mean±std) | 各 seed | n |", "|:--|:--|:--|:--|:--|"]
for ds in ("Sport", "Toys"):
    for label, tag in [v for v in VARIANTS if v[1].startswith(ds)]:
        d = collect(f"{tag}_seed*_eval.log")
        lg = LEGACY_2025.get(tag)
        if lg:
            v = parse(f"model/logs/{lg}")
            if v is not None:
                d.setdefault(2025, v)
        if not d:
            lines.append(f"| {ds} | {label} | — | 无产物 | 0 |"); continue
        vals = [d[k] for k in sorted(d)]
        s = "±%.4f" % st.stdev(vals) if len(vals) > 1 else "(单 seed)"
        lines.append(f"| {ds} | {label} | {st.mean(vals):.4f} {s} | "
                     + ", ".join(f"{k}:{d[k]:.4f}" for k in sorted(d)) + f" | {len(vals)} |")
lines += ["", "## 备注", "",
          "- Random 行为辅助消融维持单 seed 口径，见 doc/e49_behavior_ablation.md §2。",
          "- 旧单 seed 数字若被均值取代，按项目惯例标注为“历史参考”，不与新数字并列比较显著性。",
          ""]
open("doc/e49_behavior_ablation_results.md", "w").write("\n".join(lines))
print(open("doc/e49_behavior_ablation_results.md").read())
PY

hr; log "✅ E49 全链路完成（P3/P4/P5/P6）"; hr
log "结果表: doc/e49_behavior_ablation_results.md"
log "决策门: $GATE_DOC"
hr
