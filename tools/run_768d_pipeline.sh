#!/bin/bash
# ============================================================================
# 768d Content 全量实验流水线
# 自动：等待 RQ-VAE → 生成 SID codes → T5 训练
# 用法: bash tools/run_768d_pipeline.sh <dataset>
# ============================================================================
set -euo pipefail

DS=$1
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

echo "============================================"
echo "${DS} 768d Pipeline"
echo "开始时间: $(date)"
echo "============================================"

# ===== Step 1: 等待 RQ-VAE 完成 =====
CKPT_DIR="rqvae/ckpt/${DS}_TA_SID_768d"
echo "[Step 1] 等待 RQ-VAE 训练完成 (${CKPT_DIR})"

MAX_WAIT=7200  # 最多等 2 小时
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ls "${CKPT_DIR}"/*/best_collision_model.pth 2>/dev/null >/dev/null; then
        echo "  ✓ RQ-VAE checkpoint 就绪"
        BEST_CKPT=$(find "${CKPT_DIR}" -name "best_collision_model.pth" | head -1)
        break
    fi
    # 检查进程是否还在跑
    if ! pgrep -f "rqvae/main.py.*${DS}_TA_SID_768d" > /dev/null 2>&1; then
        # 进程已结束，检查是否有任何 checkpoint
        BEST_CKPT=$(find "${CKPT_DIR}" -name "*.pth" | sort | tail -1)
        if [ -n "$BEST_CKPT" ]; then
            echo "  ✓ RQ-VAE 进程结束，使用最后 checkpoint"
            break
        else
            echo "  ⚠ RQ-VAE 进程已结束但无 checkpoint"
            exit 1
        fi
    fi
    sleep 30
    WAITED=$((WAITED + 30))
done

echo "  使用: ${BEST_CKPT}"

# ===== Step 2: 生成 SID codes =====
echo ""
echo "[Step 2] 生成 SID codes"
CODE_FILE="data/${DS}/${DS}_sid_codes_768d.npy"

python3 -c "
import collections, numpy as np, torch, os, sys
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, 'rqvae')
from datasets import EmbDataset
from models.rqvae import RQVAE

ckpt_path = '${BEST_CKPT}'
output_file = '${CODE_FILE}'
device = torch.device('cuda:0')
DATA_PATH = '${PROJECT_ROOT}/data/${DS}/fused_item_emb_768d.parquet'

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt['args']
state_dict = ckpt['state_dict']

data = EmbDataset(DATA_PATH)
model = RQVAE(in_dim=data.dim,
              num_emb_list=args.num_emb_list,
              e_dim=args.e_dim,
              layers=args.layers,
              dropout_prob=args.dropout_prob,
              bn=args.bn,
              loss_type=args.loss_type,
              quant_loss_weight=args.quant_loss_weight,
              kmeans_init=args.kmeans_init,
              kmeans_iters=args.kmeans_iters,
              sk_epsilons=args.sk_epsilons,
              sk_iters=args.sk_iters)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

data_loader = DataLoader(data, num_workers=args.num_workers,
                         batch_size=64, shuffle=False, pin_memory=True)

all_indices = []
all_indices_str = []
prefix = ['<a_{}>','<b_{}>','<c_{}>','<d_{}>','<e_{}>']

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0

tt = 0
while True:
    if tt >= 30 or len(set(all_indices_str.tolist())) == len(all_indices_str):
        break
    indices_count = collections.Counter(all_indices_str.tolist())
    collision_item_groups = []
    processed = set()
    for idx, s in enumerate(all_indices_str):
        if idx in processed: continue
        if indices_count[s] > 1:
            group = [i for i, x in enumerate(all_indices_str) if x == s]
            collision_item_groups.append(group)
            processed.update(group)
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
collision = (tot_item - tot_indice) / tot_item
print(f'Collision Rate: {collision:.4f}')

all_indices_dict = {item: list(indices) for item, indices in enumerate(all_indices.tolist())}
codes = []
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

codes_array = np.array(codes)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
for dup in unique_codes[counts > 1]:
    dup_idx = np.where((codes_array == dup).all(axis=1))[0]
    for i, idx in enumerate(dup_idx):
        codes_array[idx, -1] = i

final_unique, final_counts = np.unique(codes_array, axis=0, return_counts=True)
dupes = final_unique[final_counts > 1]
if len(dupes) > 0:
    print(f'WARNING: {len(dupes)} codes still have duplicates!')
else:
    print('No duplicates after resolution.')
print(f'Saving codes to {output_file}')
np.save(output_file, codes_array)
print('✓ Codes saved!')
" 2>&1 | tee "rqvae/logs/${DS}_768d_generate_code.log"

echo ""
echo "[Step 2] SID codes 生成完成"

# ===== Step 3: T5 训练 =====
echo ""
echo "[Step 3] T5 训练 (200 epochs)"
mkdir -p model/ckpt model/logs

python3 model/main.py \
    --dataset_path "data/${DS}" \
    --code_path "data/${DS}/${DS}_sid_codes_768d.npy" \
    --save_path "model/ckpt/tiger_ta_sid_${DS}_768d.pth" \
    --log_path "model/logs/tiger_ta_sid_${DS}_768d.log" \
    --num_epochs 200 \
    --seed 2025 \
    2>&1 | tee "model/logs/tiger_ta_sid_${DS}_768d_train.log"

echo ""
echo "[Step 3] T5 训练完成"

# ===== Step 4: 输出结果 =====
echo ""
echo "============================================"
echo "${DS} 768d 结果"
echo "============================================"
NDCG=$(grep "Test Dataset.*NDCG@20" "model/logs/tiger_ta_sid_${DS}_768d.log" 2>/dev/null | tail -1 | awk '{print $NF}')
echo "Test NDCG@20: ${NDCG:-???}"
echo "完成时间: $(date)"