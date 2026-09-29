实验方案：RQ-VAE Checkpoint Selection 对行为结构保持的影响

1. 实验目的

验证当前较小的量化行为结构损失是否依赖 best_collision_model.pth 的 checkpoint 选点策略。

现有结论：

B = RQ-VAE encoder 输出的连续 latent z

C = 量化后的表示

当前 C 使用 best_collision_model.pth

该 checkpoint 按三层离散码 collision rate 最小选择，而非 reconstruction loss 最小

因此需要比较：

best_collision_model.pth

best_loss_model.pth

唯一改变 checkpoint selection，其余实验条件全部保持一致。

核心问题：

z → C 的行为结构损失是否显著依赖 collision-based checkpoint selection？

2. 实验对象

数据集：

Beauty

Sport

Toys

Seed：

完全复用 doc/g2_stage_probe_results.csv 中已有 seed

固定：

A 输入

B = z

encoder

latent

codebook 配置

quantization 方法

pair sampling

positive/negative pair 定义

AUC 计算方法

数据预处理

不重新训练模型。

3. 对照实验

对每个 Dataset × Seed：

原方案

B=z → best_collision_model.pth → C_collision

计算：

AUC(z)
AUC(C_collision)
gap_collision = AUC(z) - AUC(C_collision)

对照方案

B=z → best_loss_model.pth → C_loss

计算：

AUC(z)
AUC(C_loss)
gap_loss = AUC(z) - AUC(C_loss)

比较：

delta_gap = gap_loss - gap_collision

4. 需要统计的指标

行为结构

AUC(z)

AUC(C_collision)

AUC(C_loss)

gap_collision

gap_loss

delta_gap

离散码结构

两个 checkpoint 分别统计：

L0/L1/L2 distinct

L0/L1/L2 utilization

depth-1/2/3 prefix collision rate

full SID collision rate

RQ-VAE

reconstruction loss

优先复用：
tools/sid_quality.py

compute_utilization

compute_collision_rate

5. 输出文件

生成：

doc/checkpoint_selection_behavior_auc.csv

字段：

dataset, seed, auc_z, auc_best_collision, gap_best_collision, auc_best_loss, gap_best_loss, delta_gap

生成：

doc/checkpoint_selection_behavior_auc_summary.md

内容：

三个数据集逐 seed 结果

mean ± std

两种 checkpoint 的 utilization / collision / reconstruction loss 对比

delta_gap 分析

6. 结果判定

情况 A：best_loss 的 gap 明显更大

说明当前较小的行为结构损失明显受到 checkpoint selection 影响。

后续研究重点转向：

离散码分配 / checkpoint selection 如何影响行为结构保持。

此时不能直接宣称“codebook capacity 不是瓶颈”。

情况 B：两者 gap 都较小

说明当前行为结构保持不主要依赖 collision-based selection。

可以继续研究：

当前量化设置下，行为结构损失是否确实较小。

但仍不能直接推出“codebook capacity 永远不是瓶颈”。

情况 C：不同数据集结论不同

按数据集分别报告，不强行形成统一结论。

7. Toys 数据版本同步检查

同时检查 E49 Toys ablation 中：

TA-SID

Raw-CoOcc

ItemCF

ZeroPad

Shuffled/Random（若存在）

是否全部使用当前可复现 checkpoint。

已知：

旧 TA-SID：0.0367 ± 0.0008，来自 orphan codes，作废

最新健康结果：0.0377 ± 0.0007

Raw-CoOcc：0.0340

需要重新计算 TA-SID 相对 Raw-CoOcc 的提升，并检查其他 variant 是否存在 orphan checkpoint。

8. 实验原则

不重新训练

不修改模型结构

不修改 AUC 定义

不修改 pair sampling

不修改数据

不提前假设结论

不根据结果自动提出新方法

本实验只回答一个问题：

checkpoint selection criterion 是否是当前行为结构保持结果的重要来源？