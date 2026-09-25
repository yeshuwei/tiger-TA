#!/usr/bin/env bash
# 4.1 后处理：为新训练的 K0 模型生成 SID codes + 跑容量诊断
# 用法: bash rqvae/run_e41_postprocess.sh 512 768 2048
cd "$(dirname "$0")/.."

for K0 in "$@"; do
  echo "=========================================================="
  echo "[$(date '+%F %T')] Postprocess K0=${K0}"
  # 找 best_collision_model.pth（该 tag 下最新 ckpt 目录）
  CKPT_DIR=$(ls -d rqvae/ckpt/Toys_TA_SID_L0${K0}/*/ 2>/dev/null | tail -1)
  CKPT="${CKPT_DIR}best_collision_model.pth"
  if [ ! -f "$CKPT" ]; then
    echo "⚠ 无 checkpoint: $CKPT，跳过"
    continue
  fi
  echo "ckpt: $CKPT"

  # 1) 生成 SID codes（纯 argmin + 碰撞消解，与原生成器一致）
  python rqvae/generate_sid_codes_param.py \
    --ckpt_path "$CKPT" \
    --data_path data/Toys/fused_item_emb_768d.parquet \
    --output_file "data/Toys/Toys_ta_sid_L0${K0}_sid.npy" \
    --device cuda:0 2>&1 | tail -6

  # 2) 容量诊断（AUC / z-gap / collision / recon）
  python tools/sid_capacity_diagnostic.py \
    --datasets Toys --topk 1000 --n_random 3000 \
    --ckpt-tag "Toys_TA_SID_L0${K0}" \
    --code-file "Toys_ta_sid_L0${K0}_sid.npy" \
    --report "doc/e41_K0${K0}.md" 2>&1 | grep -E "xq\[L0\]|xq\[full\]|z |L0:|L1:|L2:|3-token|碰撞|AUC" | head -12

  # 3) recon MSE（统一口径，全量 use_sk=True）
  python - <<PYEOF
import sys, torch
sys.path.insert(0, 'rqvae')
from torch.utils.data import DataLoader
from datasets import EmbDataset
from models.rqvae import RQVAE
from torch.nn import functional as F
ckpt = torch.load("$CKPT", map_location='cpu', weights_only=False)
a = ckpt["args"]
data = EmbDataset("data/Toys/fused_item_emb_768d.parquet")
model = RQVAE(in_dim=data.dim, num_emb_list=a.num_emb_list, e_dim=a.e_dim,
              layers=a.layers, dropout_prob=a.dropout_prob, bn=a.bn,
              loss_type=a.loss_type, quant_loss_weight=a.quant_loss_weight,
              kmeans_init=a.kmeans_init, kmeans_iters=a.kmeans_iters,
              sk_epsilons=a.sk_epsilons, sk_iters=a.sk_iters)
model.load_state_dict(ckpt["state_dict"]); model.eval()
loader = DataLoader(data, batch_size=512, shuffle=False)
tot=0.0; n=0
with torch.no_grad():
    for x in loader:
        out,_,_,_ = model(x, use_sk=True, return_z=True)
        tot += F.mse_loss(out,x,reduction='mean').item()*x.shape[0]; n+=x.shape[0]
print(f"K0=${K0} recon MSE = {tot/n:.4f}")
PYEOF
done
echo "[$(date '+%F %T')] ALL POSTPROCESS DONE"
