#!/bin/bash
# ============================================================================
# Beauty 4096d Content 最小验证脚本
# 目的：验证用 4096d Content 替代 768d 后，TA-SID 结果是否一致
# 用法: ./tools/validate_beauty_4096d.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

echo "============================================"
echo "Beauty 4096d Content 验证"
echo "开始时间: $(date)"
echo "============================================"

# 1. RQ-VAE 训练（500 epoch，完整是 3000）
echo ""
echo "[Step 1] RQ-VAE 训练 (500 epochs vs 3000)"
CKPT_DIR="rqvae/ckpt/Beauty_4096d"
mkdir -p "$CKPT_DIR"
python3 rqvae/main.py \
    --data_path "${PROJECT_ROOT}/data/Beauty/fused_item_emb_4096d.parquet" \
    --ckpt_dir "${PROJECT_ROOT}/${CKPT_DIR}" \
    --epochs 500 \
    --batch_size 1024 \
    --eval_step 50 \
    2>&1 | tee "rqvae/logs/Beauty_4096d_rqvae.log"

echo "[Step 1] RQ-VAE 完成"

# 2. 找到最佳 checkpoint
echo ""
echo "[Step 2] 生成 SID codes"
BEST_CKPT=$(find "${CKPT_DIR}" -name "best_collision_model.pth" | head -1)
if [ -z "$BEST_CKPT" ]; then
    # 如果 500 epoch 还没找到 best_collision，用最后一个
    BEST_CKPT=$(find "${CKPT_DIR}" -name "*.pth" | sort | tail -1)
fi
echo "使用 checkpoint: ${BEST_CKPT}"

CODE_FILE="data/Beauty/Beauty_ta_sid_4096d.npy"
ABS_BEST_CKPT=$(realpath "$BEST_CKPT")
ABS_CODE_FILE=$(realpath "$CODE_FILE")

cat > "rqvae/generate_code_beauty_4096d.py" << PYEOF
"""Generate Beauty 4096d SID codes"""
import collections, numpy as np, torch, os, sys
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import EmbDataset
from models.rqvae import RQVAE

ckpt_path = r"${ABS_BEST_CKPT}"
output_file = r"${ABS_CODE_FILE}"
device = torch.device("cuda:0")
DATA_PATH = r"${PROJECT_ROOT}/data/Beauty/fused_item_emb_4096d.parquet"

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

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
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
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
        if idx in processed:
            continue
        if indices_count[s] > 1:
            group = [i for i, x in enumerate(all_indices_str) if x == s]
            collision_item_groups.append(group)
            processed.update(group)
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1

print("All indices number:", len(all_indices))
collision_counts = collections.Counter(all_indices_str.tolist())
print("Max conflicts:", max(collision_counts.values()))
tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate:", (tot_item - tot_indice) / tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

codes = []
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

codes_array = np.array(codes)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]
if len(duplicates) > 0:
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(duplicate_indices):
            codes_array[idx, -1] = i

new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
dupes = new_unique_codes[new_counts > 1]
if len(dupes) > 0:
    print("Still have duplicates:", dupes)
else:
    print("No duplicates after resolution.")

print(f"Saving codes to {output_file}")
np.save(output_file, codes_array)
print("✓ Codes saved!")
PYEOF

cd rqvae
python3 generate_code_beauty_4096d.py 2>&1 | tee "${PROJECT_ROOT}/rqvae/logs/Beauty_4096d_generate_code.log"
cd "${PROJECT_ROOT}"

echo "[Step 2] SID codes 生成完成: ${CODE_FILE}"

# 3. T5 训练（100 epoch，完整是 200）
echo ""
echo "[Step 3] T5 训练 (100 epochs vs 200)"
python3 model/main.py \
    --dataset_path "data/Beauty" \
    --code_path "${CODE_FILE}" \
    --save_path "model/ckpt/beauty_ta_sid_4096d.pth" \
    --log_path "model/logs/beauty_ta_sid_4096d.log" \
    --num_epochs 100 \
    --seed 2025 \
    2>&1 | tee "model/logs/beauty_ta_sid_4096d_train.log"

echo "[Step 3] T5 训练完成"

# 4. 结果提取
echo ""
echo "============================================"
echo "结果对比"
echo "============================================"
echo ""
NDCG_4096=$(grep "Test Dataset.*NDCG@20" "model/logs/beauty_ta_sid_4096d.log" 2>/dev/null | tail -1 | awk '{print $NF}')
NDCG_768=$(grep "Test Dataset.*NDCG@20" "model/logs/tiger_ta_sid_beauty_seed42.log" 2>/dev/null | tail -1 | awk '{print $NF}')

echo "  ┌──────────────────────────────────────┬────────────┐"
echo "  │ Beauty TA-SID                        │ NDCG@20   │"
echo "  ├──────────────────────────────────────┼────────────┤"
echo "  │ 768d Content（原始，完整训练）       │ ${NDCG_768:-???}     │"
echo "  │ 4096d Content（本次验证，100 epoch） │ ${NDCG_4096:-???}     │"
echo "  └──────────────────────────────────────┴────────────┘"
echo ""
echo "解读:"
echo "  • 如果 4096d 结果 ≈ 768d → 维度切换不影响结论, 可直接全量重跑"
echo "  • 如果 4096d 结果 >> 768d → 4096d 更好, 值得全量重跑"
echo "  • 如果 4096d 结果 << 768d → 需要分析原因"
echo ""
echo "完成时间: $(date)"
