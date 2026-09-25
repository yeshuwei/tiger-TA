#!/bin/bash
# ============================================================
# B-3: L_trans v2 (x_q-based) + L_balance
# ============================================================
set -e

PROJECT_ROOT="/data/gtx/project/code/TIGER_minilm"
cd "$PROJECT_ROOT"

# 使用 conda env 的 python（不用 conda activate，直接全路径）
export PATH="/home/media601/anaconda3/envs/tiger_t5/bin:$PATH"
PYTHON="python3"

DS="Sport"
LAMBDA_TRANS=0.01
LAMBDA_BALANCE=0.01
TAU=0.1
TAG="v2_xq"

CKPT_DIR="rqvae/ckpt/${DS}_Trans_768d_${TAG}"
DATA_PATH="data/${DS}/fused_item_emb_768d.parquet"
TRANS_PATH="data/${DS}/transition_pairs_768d.npy"
LOG_DIR="rqvae/logs"
CODE_OUT="data/${DS}/${DS}_track_b_sid_codes_768d_${TAG}.npy"
T5_SAVE="model/ckpt/tiger_ta_sid_sport_track_b_${TAG}.pth"
T5_LOG="model/logs/tiger_ta_sid_sport_track_b_768d_${TAG}.log"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

echo "=========================================="
echo " B-3: L_trans v2 (x_q) + L_balance"
echo " lambda_trans=$LAMBDA_TRANS, lambda_balance=$LAMBDA_BALANCE, tau=$TAU"
echo "=========================================="

# ==========================================
# Step 1: RQ-VAE Training (v2)
# ==========================================
echo ""
echo "=== Step 1: RQ-VAE Training (v2: L_trans on x_q) ==="

$PYTHON rqvae/main.py \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --lambda_trans "$LAMBDA_TRANS" \
    --lambda_balance "$LAMBDA_BALANCE" \
    --tau "$TAU" \
    --trans_graph_path "$TRANS_PATH" \
    --epochs 2000 \
    --eval_step 50 \
    2>&1 | tee "${LOG_DIR}/${DS}_TrackB_${TAG}_rqvae.log"

echo "RQ-VAE training complete"

# ==========================================
# Step 2: Find best checkpoint
# ==========================================
echo ""
echo "=== Step 2: Find best checkpoint ==="

BEST_CKPT=$($PYTHON -c "
import os, re
ckpt_base = '$CKPT_DIR'
best_collision = float('inf')
best_path = None
for d in sorted(os.listdir(ckpt_base)):
    dp = os.path.join(ckpt_base, d)
    bp = os.path.join(dp, 'best_collision_model.pth')
    if not os.path.exists(bp): continue
    for f in os.listdir(dp):
        m = re.search(r'_collision_([0-9.]+)_', f)
        if m:
            c = float(m.group(1))
            if c < best_collision:
                best_collision = c
                best_path = bp
if best_path is None:
    ds2 = sorted(os.listdir(ckpt_base), key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
    latest = ds2[-1]
    best_path = os.path.join(ckpt_base, latest, 'best_collision_model.pth')
print(best_path)
")

echo "Best ckpt: $BEST_CKPT"
if [ ! -f "$BEST_CKPT" ]; then echo "ERROR: no ckpt found"; exit 1; fi

# ==========================================
# Step 3: Generate SID codes
# ==========================================
echo ""
echo "=== Step 3: Generate SID codes ==="

if [ -f "$CODE_OUT" ]; then
    echo "Codes exist, skipping"
else
    # Use the step45 pipeline style auto-generated script
    GEN_SCRIPT="rqvae/generate_code_${DS}_${TAG}.py"

    cat > "$GEN_SCRIPT" << PYEOF
import collections, numpy as np, torch, os, sys
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rqvae.datasets import EmbDataset
from rqvae.models.rqvae import RQVAE

DS = "${DS}"
ckpt_path = r"${BEST_CKPT}"
output_file = os.path.join("${PROJECT_ROOT}", "${CODE_OUT}")
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
args, sd = ckpt["args"], ckpt["state_dict"]
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(args.data_path):
    args.data_path = os.path.normpath(os.path.join(script_dir, "..", args.data_path))
print(f"Data: {args.data_path}")
data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim, num_emb_list=args.num_emb_list, e_dim=args.e_dim,
              layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn,
              loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
              kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
              sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters)
model.load_state_dict(sd)
model = model.to(device); model.eval()

loader = DataLoader(data, num_workers=args.num_workers, batch_size=64, shuffle=False, pin_memory=True)

all_indices, all_str = [], []
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]
for d in tqdm(loader):
    idx = model.get_indices(d.to(device), use_sk=False).view(-1, 3).cpu().numpy()
    for ix in idx:
        c = [prefix[i].format(int(ix[i])) for i in range(3)]
        all_indices.append(c); all_str.append(str(c))

ai, astr = np.array(all_indices), np.array(all_str)
for vq in model.rq.vq_layers[:-1]: vq.sk_epsilon = 0.0

for tt in range(30):
    if len(set(astr.tolist())) == len(astr): break
    from collections import Counter
    cnt = Counter(astr.tolist())
    groups = []
    seen = set()
    for idx, s in enumerate(astr):
        if idx in seen: continue
        if cnt[s] > 1:
            g = [i for i, x in enumerate(astr) if x == s]
            groups.append(g); seen.update(g)
    print(f"Round {tt}: {len(groups)} collision groups")
    for g in groups:
        d = data[g].to(device)
        ix2 = model.get_indices(d, use_sk=True).view(-1, 3).cpu().numpy()
        for item, i2 in zip(g, ix2):
            c2 = [prefix[i].format(int(i2[i])) for i in range(3)]
            ai[item], astr[item] = c2, str(c2)

print(f"Collision: {(len(astr)-len(set(astr.tolist())))/len(astr):.4f}")

codes = np.array([[int(x.split('_')[1].rstrip('>')) for x in row] for row in ai])
codes = np.hstack((codes, np.zeros((codes.shape[0],1), dtype=int)))
# final dedup dim
uc, cnt = np.unique(codes, axis=0, return_counts=True)
for dup in uc[cnt > 1]:
    di = np.where((codes == dup).all(axis=1))[0]
    for i, idx in enumerate(di): codes[idx, -1] = i
print(f"No duplicates: {len(np.unique(codes, axis=0)) == len(codes)}")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
np.save(output_file, codes)
print(f"Saved: {output_file}")
PYEOF

    echo "Running code generation..."
    $PYTHON "$GEN_SCRIPT" 2>&1 | tee "${LOG_DIR}/${DS}_TrackB_${TAG}_generate_code.log"
    echo "Code generation done"
fi

# ==========================================
# Step 4: T5 Training
# ==========================================
echo ""
echo "=== Step 4: T5 Training ==="

if [ -f "$T5_SAVE" ]; then
    echo "T5 model exists, skipping"
else
    $PYTHON model/main.py \
        --dataset_path "data/${DS}" \
        --code_path "$CODE_OUT" \
        --save_path "$T5_SAVE" \
        --log_path "$T5_LOG" \
        --seed 2025 \
        2>&1 | tee "${LOG_DIR}/${DS}_TrackB_${TAG}_t5_train.log"
    echo "T5 training done"
fi

echo ""
echo "B-3 complete!"
echo "Codes: $CODE_OUT"
echo "Model: $T5_SAVE"
echo "Log:   $T5_LOG"
