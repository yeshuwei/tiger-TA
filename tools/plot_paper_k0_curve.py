#!/usr/bin/env python3
"""
plot_paper_k0_curve.py — 论文图5-x：不同粗层码本容量下量化结构损失变化（Toys）

单面板。K0 为离散实验配置 {256, 512, 1024}（Fixed-small / Adaptive / Fixed-large，
均为真实训练过的 RQ-VAE 模型，见 doc/e41_k0_sweep_results.md），横轴线性刻度。
纵轴 z-gap 半对数（0.0931 → 0.0003 跨 ~300×）。

图元素：ε=0.01 无损阈值虚线小标记 + 各点数值标注 + 512 处小箭头 K0*=512。
叙事：256 断崖（0.093 ≫ ε）→ 512 跌破 ε（0.0045）→ 1024 无进一步断崖（0.0003）。

数据源：Toys, 768d；转移对 top-1000，随机对 3000 (seed=0)（E44 统一口径，
与六数据集 z-gap 表一致；早期 e43 版本用 5000 随机对得 0.0933，已按 3000 规范值 0.0931 修正）。
用法：python3 tools/plot_paper_k0_curve.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 离散实验配置：(K0, z-gap)
data = [
    (256,  0.0931),   # Fixed-small（主实验 baseline，Toys_TA_SID_768d）
    (512,  0.0045),   # Adaptive（K0*=512，Toys_TA_SID_L0512）
    (1024, 0.0003),   # Fixed-large（Toys_TA_SID_L0Big）
]
k0 = np.array([d[0] for d in data])
gap = np.array([d[1] for d in data])
EPS = 0.01
C_GAP = '#d62728'

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.semilogy(k0, gap, 'o-', color=C_GAP, lw=1.8, ms=7)

# ε=0.01 无损阈值虚线（小标记）
ax.axhline(EPS, color='gray', ls=':', lw=1.2)
ax.text(0.98, EPS, ' $\\varepsilon$ = 0.01',
        transform=ax.get_yaxis_transform(), ha='right', va='bottom',
        fontsize=8, color='gray')

# 数据点数值标注
for (k, g) in zip(k0, gap):
    ax.annotate(f'{g:.4f}', xy=(k, g), xytext=(0, 8),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=8, color='#333')

# 512 处小箭头：K0*=512（自适应选择点）
ax.annotate('$K_0^*$ = 512', xy=(512, 0.0045), xytext=(690, 0.04),
            fontsize=9, color='#333', ha='center',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.1))

ax.set_xticks(k0)
ax.set_xticklabels(['256', '512', '1024'])
ax.set_xlabel('Coarse codebook capacity $K_0$')
ax.set_ylabel('z-gap (AUC(z) $-$ AUC($x_q$))')
ax.set_ylim(1e-4, 0.3)
ax.grid(alpha=0.3, axis='y', which='both')

out = os.path.join('doc', 'figures', 'paper_k0_sweep_curve.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'figure saved: {out}')
