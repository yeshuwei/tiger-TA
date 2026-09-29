# Checkpoint Trajectory Experiment

## 1. 研究问题

RQ-VAE训练过程中，重建/量化优化是否会影响行为转移结构从连续表示 `z` 到离散表示 `xq` 的保持？

核心指标：

`z-gap = AUC(z) - AUC(xq)`

---

## 2. 实验动机

当前 TA-SID 在量化前加入了 transition structure，但尚未证明该结构能够稳定传递到最终 Semantic ID。

此前 `best_loss vs best_collision` 实验发现 checkpoint selection 存在训练目标偏离问题，因此该实验不再比较不同 checkpoint selection strategy，而采用固定训练轮次进行 trajectory 分析。

---

## 3. 实验设计

Datasets:

* Beauty
* Sport
* Toys

Seeds:

* 2024
* 2025
* 2026

固定训练轮次：

* 50
* 500
* 1000
* 2000

同一次 training run 中保存并评估各 checkpoint。

---

## 4. 控制变量

保持完全一致：

* TA-SID输入
* PPMI+SVD transition embedding
* RQ-VAE architecture
* codebook size
* latent dimension
* training hyperparameters
* pair sampling
* AUC计算方式
* quantization/code generation方式

唯一变化：

> training epoch

---

## 5. 记录指标

### Structure

* AUC(z)
* AUC(xq)
* z-gap

### Training

* reconstruction loss
* quantization loss
* total loss

### Codebook

* L0/L1/L2 utilization
* prefix collision
* full collision

---

## 6. 关键判断

重点观察：

`reconstruction loss ↓` 是否伴随 `z-gap ↑`

如果稳定出现，说明：

> RQ-VAE的训练优化目标与behavior structure preservation之间可能存在偏离。

如果不存在，则不继续强行构造该方向。

---

## 7. 实验定位

本实验不是第二个创新点，也不提出新方法。

作用是判断：

> 是否存在值得进一步研究的“behavior structure preservation during quantization”现象。

只有现象稳定后，才设计第二阶段方法。
