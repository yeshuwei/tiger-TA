import collections, numpy as np, torch, os, sys
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rqvae.datasets import EmbDataset
from rqvae.models.rqvae import RQVAE

DS = "Sport"
ckpt_path = r"rqvae/ckpt/Sport_Trans_768d_v2_xq/Jul-29-2026_11-03-03/best_collision_model.pth"
output_file = os.path.join("/data/gtx/project/code/TIGER_minilm", "data/Sport/Sport_track_b_sid_codes_768d_v2_xq.npy")
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
