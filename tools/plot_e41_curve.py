#!/usr/bin/env python3
"""
plot_e41_curve.py — 4.1 K0 sweep 最终图（v2，紧凑清晰）
横轴 K0（log2 刻度），三面板：
  Panel 1: transition AUC (xq[full])，y 收窄放大
  Panel 2: z-gap，半对数刻度（256 与 512-2048 相差 20 倍，bar 会压扁）
  Panel 3: L0 utilization，标注 used/total，错开防重叠
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# K0, xq[full] AUC, z AUC, L0 used, L0 total
data = [
    (256,  0.8719, 0.9652,  62, 256),
    (512,  0.9675, 0.9720, 305, 512),
    (768,  0.9670, 0.9715, 300, 768),
    (1024, 0.9629, 0.9632, 275, 1024),
    (1536, 0.9647, 0.9678, 304, 1536),
    (2048, 0.9740, 0.9755, 375, 2048),
]
k0 = np.array([d[0] for d in data])
xq = np.array([d[1] for d in data])
z  = np.array([d[2] for d in data])
used = np.array([d[3] for d in data])
tot  = np.array([d[4] for d in data])
gap = z - xq
util = used / tot * 100

fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.6), sharex=True,
                         gridspec_kw={'hspace': 0.18})
fig.suptitle('K0 sweep — RQ-VAE L0 codebook size → transition structure (Toys)',
             fontsize=12, y=0.99)

# ---------- Panel 1: transition AUC ----------
ax = axes[0]
ax.axhspan(0.965, 0.990, color='#ff7f0e', alpha=0.08, lw=0)
ax.plot(k0, xq, 'o-', color='#1f77b4', lw=2, ms=7, label='xq[full] AUC')
ax.plot(k0, z, 's--', color='#ff7f0e', lw=1.4, ms=5, alpha=0.75, label='z (ceiling)')
# 标注塌缩点与峰值
ax.annotate('L0 collapse\n62/256 codes', xy=(256, 0.872), xytext=(400, 0.882),
            fontsize=9, color='#d62728', ha='center',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1))
ax.annotate('peak AUC 0.974', xy=(2048, 0.974), xytext=(1400, 0.982),
            fontsize=9, color='#1f77b4', ha='center',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1))
ax.set_ylabel('Transition AUC')
ax.set_ylim(0.855, 0.990)
ax.grid(alpha=0.3, axis='y')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.text(0.012, 0.08, 'transition AUC = P(transfer cos > random cos)',
        transform=ax.transAxes, fontsize=7.5, color='#555')

# ---------- Panel 2: z-gap (log) ----------
ax = axes[1]
ax.semilogy(k0, gap, 'o-', color='#d62728', lw=2, ms=7)
ax.fill_between(k0, gap, 0.0002, color='#d62728', alpha=0.10)
for i, g in enumerate(gap):
    ax.annotate(f'{g:.4f}', xy=(k0[i], g), xytext=(0, 6),
                textcoords='offset points', ha='center', fontsize=8)
ax.axhline(0.01, color='gray', ls=':', lw=1)
ax.text(0.012, 0.015, 'target ≈ 0 (lossless quantization)', fontsize=7.5, color='#555')
ax.set_ylabel('z-gap = AUC(z) − AUC(xq)')
ax.set_ylim(0.0004, 0.2)
ax.grid(alpha=0.3, axis='y', which='both')
ax.text(0.012, 0.92, 'log scale — gap drops 20× from 256 to 512',
        transform=ax.transAxes, fontsize=7.5, color='#555')

# ---------- Panel 3: utilization ----------
ax = axes[2]
ax.axhline(60, color='#2ca02c', ls=':', lw=1)
ax.plot(k0, util, '^-', color='#2ca02c', lw=2, ms=7)
# 标注 used/total，错开防重叠
for i, (u, k) in enumerate(zip(util, k0)):
    dy = 6 if i % 2 == 0 else -9
    va = 'bottom' if dy > 0 else 'top'
    ax.annotate(f'{int(used[i])}/{tot[i]}', xy=(k, u), xytext=(0, dy),
                textcoords='offset points', ha='center', va=va, fontsize=8)
ax.set_ylabel('L0 utilization (%)')
ax.set_ylim(0, 72)
ax.grid(alpha=0.3, axis='y')
ax.text(0.012, 0.92, 'active codes / codebook size',
        transform=ax.transAxes, fontsize=7.5, color='#555')
ax.text(0.988, 0.08, 'util 60% at K0=512 (sweet spot)',
        transform=ax.transAxes, ha='right', fontsize=7.5, color='#2ca02c')

# ---------- x axis (shared) ----------
axes[2].set_xlabel('K0  (L0 codebook size, log2 scale)')
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_xticks(k0)
    ax.set_xticklabels([str(k) for k in k0])

out = 'doc/figures/e41_k0_sweep_curve.png'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150)
print(f'figure saved: {out}')
