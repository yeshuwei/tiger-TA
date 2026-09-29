# `best_loss_model.pth` 固定于 epoch 49 的根因定位（2026-09-28）

> **一句话结论：不是代码 bug，也没有可修的保存逻辑。**
> `train_loss = recon + quant_loss_weight × quant` 中两项**反向演化**
> （recon 单调下降、quant 项单调上升），总损失在 **ep42–79** 触底后持续上升；
> 而评估网格 `eval_step=50` 使**最早评估点恰为 epoch 49**，
> 于是总损失的最小值在首次评估处即被捕获，此后 2950 个 epoch 再无更小者。
> `trainer.fit` 的比较与保存逻辑经逐行复刻验证**完全正确**。
> 依任务约定（"如果发现问题不是代码 bug…不要修改代码，直接报告证据"），
> **本轮未修改 `rqvae/trainer.py`**。

**关联**：[checkpoint_selection_behavior_auc_summary.md](checkpoint_selection_behavior_auc_summary.md)（该实验的完整结果）｜
设计 [exp/RQ-VAE checkpoint selection实验设计.md](exp/RQ-VAE%20checkpoint%20selection实验设计.md)

---

## 1. 逐项审计结果（对应任务第 1、2 项）

审计对象 [rqvae/trainer.py](../rqvae/trainer.py)（共 386 行）。

| # | 检查项 | 位置 | 结论 |
|:--|:--|:--|:--|
| 1 | `best_loss` 初始化 | L44 `self.best_loss = np.inf` | ✅ 正确 |
| 2 | `eval_step` / 评估时机 | L36 `eval_step = min(args.eval_step, epochs)`；L340 `if (epoch_idx + 1) % self.eval_step == 0` | ✅ 逻辑正确。`eval_step=50` ⇒ **最早评估点 = epoch 49**（0-indexed） |
| 3 | `_train_epoch()` 返回的 loss | L269 `return total_loss, total_recon_loss` | ⚠ 是**逐 batch 求和**（非均值），但 `n_batches` 逐 epoch 恒定 ⇒ 跨 epoch 可比，**不构成 bug**（详见下注） |
| 4 | loss 的 batch/epoch 累计方式 | L253 `total_loss += loss.item()`；L254 同 | ✅ 标量累加，无广播/张量残留 |
| 5 | `cur_loss < best_loss` 判断 | L344 `if train_loss < self.best_loss:` | ✅ 比较对象同口径（都是本次 epoch 的 train_loss），逻辑正确 |
| 6 | `best_loss_model.pth` 保存 | L345–346 | ✅ 条件成立即保存并同步 `self.best_loss` |
| 7 | `best_collision_model.pth` 保存 | L348–354 | ✅ 独立判据，互不干扰 |

**五项被怀疑的病因逐一排除**：

| 嫌疑 | 判定 | 依据 |
|:--|:--|:--|
| loss 累计/平均口径错误 | ❌ 不成立 | 求和口径逐 epoch 一致，仅影响数值尺度不影响排序（§3.1） |
| tensor / numpy 类型问题 | ❌ 不成立 | `train_loss` 为 python float，`best_loss` 为 `np.inf`/`np.float64`，`float < np.float64` 正常；日志与 ckpt 字段（2.826852）逐位对得上 |
| 比较条件错误 | ❌ 不成立 | L344 复刻后与训练日志完全一致（§3.2） |
| `best_loss` 没有正确更新 | ❌ 不成立 | 它**确实更新了，且只更新一次**（ep49），此后无更小值可更新 |
| **eval loss 实际没有变化** | ⚠ **方向反了，且暴露一个结构问题** | **训练器根本没有 eval loss**：`_valid_epoch` 只返回 `collision_rate`（L297），全日志 0 行 "eval loss"。选择信号只有训练过程中的 train_loss |
| checkpoint 保存逻辑错误 | ❌ 不成立 | 保存路径、字段、时机均正确 |

> 注（求和口径的一个细节）：`F.mse_loss(..., reduction='mean')` 对每个 batch 除以
> `B_b × 768`，再把这些 batch 均值相加 ⇒ **最后一个不满的 batch 与满 batch 等权**
> （Toys 为 11×1024 + 1×660，`N=11924`、`n_batches=12`）。因此该值并非严格的
> 全局逐元素均值（实测 0.2116 vs 全局值 0.2473），但**逐 epoch 口径恒定**，
> 不影响跨 epoch 排序，也与本结论无关——仅记录，便于日后对照日志数值。

**另一个观察（非病因，仅记录）**：L327/350/356 的 `cur_eval_step` 被赋值 3 处、
**从未被读取**（全仓库 grep 无其他引用）——是"耐心早停"类机制的废弃残骸。死代码，
不影响选择行为。

**再一个观察（非病因，超出本次范围）**：`RQVAE.forward` 的 `use_sk` 默认为 `True`，
而 `_train_epoch`（L226）与 L223 都未显式传参 ⇒ **训练时量化走 Sinkhorn**；
但 `get_indices`（L76，评估/码生成路径）默认 `use_sk=False`，codegen 主 pass 也是
`use_sk=False`。由于主表 `sk_epsilons=[0,0,0.003]` 只有 L2 受 `use_sk` 影响，
L0/L1 不受波及，故**不是 ep49 现象的成因**；但它是一个真实的训练/推理口径不一致，
若日后要查 L2 相关行为可从此处入手。本轮不改。

---

## 2. 根因：目标函数的两项反向演化

从 9 个 run 的训练日志解析逐 epoch 的 `train loss` 与 `reconstruction loss`
（`quant·w = train_loss − recon`）：

| dataset | seed | recon @首次 → @末次 | quant·w @首次 → @末次 | train_loss 真实 argmin | 该 argmin 是否落在 eval 网格 |
|:--|:--:|:--|:--|:--:|:--:|
| Beauty | 2024 | 11.9115 → 1.7311（↓85%） | 0.2426 → 3.9195（×16） | epoch **58** | ❌ 否 |
| Beauty | 2025 | 11.9259 → 1.7331（↓85%） | 0.2355 → 3.8302（×16） | epoch **79** | ❌ 否 |
| Beauty | 2026 | 11.8649 → 1.7471（↓85%） | 0.1470 → 4.1560（×28） | epoch **58** | ❌ 否 |
| Sport | 2024 | 17.8686 → 2.6498（↓85%） | 0.3865 → 6.6829（×17） | epoch **42** | ❌ 否 |
| Sport | 2025 | 17.8369 → 2.6928（↓85%） | 0.3660 → 6.5855（×18） | epoch **43** | ❌ 否 |
| Sport | 2026 | 17.7920 → 2.6720（↓85%） | 0.2591 → 6.8232（×26） | epoch **43** | ❌ 否 |
| Toys | 2024 | 11.9159 → 1.7132（↓86%） | 0.2256 → 3.6917（×16） | epoch **74** | ❌ 否 |
| Toys | 2025 | 11.9094 → 1.7121（↓86%） | 0.2278 → 3.8464（×17） | epoch **59** | ❌ 否 |
| Toys | 2026 | 11.8634 → 1.7026（↓86%） | 0.1504 → 3.9627（×26） | epoch **57** | ❌ 否 |

**9/9 一致**：重建损失下降 85–86%，承诺/码本项上升 16–28 倍。
由于 `quant_loss_weight = 1.0`，**总损失由 quant 项支配**，故在 ep42–79 触底后单调上升。

以 Toys 2024（完整 2654 epoch）为例的拐点前后：

| epoch | train_loss | recon | quant·w |
|--:|--:|--:|--:|
| 45 | 2.8204 | 2.5991 | 0.2213 |
| **49（首次 eval）** | **2.8269** | 2.5578 | 0.2691 |
| **74（真实 argmin）** | **2.7752** | — | — |
| 99 | 2.8766 | 2.2843 | 0.5923 |
| 249 | 3.4102 | 2.0795 | 1.3307 |
| 1049 | 4.7396 | 1.8126 | 2.9270 |
| 2049 | 5.2210 | 1.7365 | 3.4845 |
| 2649 | 5.4041 | 1.7148 | 3.6893 |

**成因**：这是 RQ-VAE / VQ-VAE 的经典现象——码本承诺（commitment）项随编码器输出
尺度增大而增长，与重建项的下降方向相反。`sk_epsilons=[0,0,0.003]`、
`beta=0.25`、`quant_loss_weight=1.0` 下，承诺项最终压倒重建项，
使**训练目标与重建质量解耦**：模型重建越来越好，目标值越来越差。

---

## 3. 验证（对应任务第 4、5 项）

### 3.1 保存逻辑复刻验证

按 `trainer.fit` 的判定规则（`eval_step=50`，比较 `train_loss < best_loss`）
在 Toys 2024 的真实日志轨迹上逐评估点复刻：

```
epoch  train_loss   best_loss_before   best_loss_after   保存 best_loss?
   49      2.8269               inf            2.8269   ✅ 保存
   99      2.8766            2.8269            2.8269   —
  149      3.0612            2.8269            2.8269   —
  199      3.2540            2.8269            2.8269   —
  249      3.4102            2.8269            2.8269   —
  299      3.5756            2.8269            2.8269   —
  449      3.9446            2.8269            2.8269   —
  999      4.7006            2.8269            2.8269   —
 2049      5.2210            2.8269            2.8269   —

共保存 1 次，事件 epoch = [49]；最终 best_loss = 2.8269 @ epoch 49
train_loss 真实 argmin = epoch 74 (2.7752)；eval 网格能否捕获该 epoch：否（网格跳过）
```

**复刻结果与磁盘上 checkpoint 的实际状态（`best_loss` 文件在 ep49）完全吻合** ——
`best_loss` 变量**正确地**在第一次评估时更新到 2.8269，此后**确实没有更小值**
可触发第二次更新。

### 3.2 两个 checkpoint 的直接重测（trainer 同口径，eval 全量）

| ckpt | Σ per-batch mean recon | Σ rq_loss | 总损失 | 与训练日志对照 |
|:--|--:|--:|--:|:--|
| `best_loss_model.pth`（ep49） | 2.5397 | 0.2735 | **2.8132** | 日志 ep49 = 2.8269 ✅ |
| `best_collision_model.pth`（ep2049） | 1.7277 | 3.4624 | **5.1901** | 日志 ep2049 = 5.2210 ✅ |

**9/9 单元同结论**：`best_loss_model.pth` 的总损失**确实**小于 `best_collision_model.pth`
（Beauty 2.79/2.85/2.73、Sport 4.35/4.26/4.21、Toys 2.81/2.73/2.67，对阵 5.37/5.30/5.77、
9.33/9.17/9.36、5.19/4.99/4.94）。
**选择准则被忠实执行**，`best_loss` 检查点就是这条轨迹上"总损失最小"的那个状态。

### 3.3 修复前后对比（对应任务第 5 项）

| 问题 | 修复前 | 修复后 |
|:--|:--|:--|
| best_loss 是否固定于 ep49 | 是 | **仍为 ep49 —— 因为这不是 bug** |
| eval loss 是否正常变化 | 训练器**不存在** eval loss | 同上 |
| best_loss 文件是否对应历史最低总损失 | **是**（已核实） | 是 |

---

## 4. 为什么不修改代码（对应任务第 3、6 项）

1. **没有可修的 bug。** 已核实的"异常"全部由训练动力学生成，修改比较/保存逻辑
   不改变任何行为，反而制造无谓 diff。
2. **"让 best_loss 能在后续 epoch 刷新"这一目标本身没有合法实现方式**：
   在这条轨迹上 ep42–79 之后总损失**再无下降**，任何不改目标函数的修改都刷不出新值。
3. 若强行修改（改用重建损失选择、加密评估网格、按固定 epoch 另存），
   就**改变了实验定义**——正是任务第 6 项明令禁止的
   （不添加 loss、不重新设计量化、不改数据/seed/codebook）。
4. 故本轮**未改动 `rqvae/trainer.py` 的任何一行**。

---

## 5. 附带的错误修正（本轮自查发现）

定位过程中发现**上一轮实验脚本自身**有一个量纲 bug，已修：

- `tools/checkpoint_selection_probe.py` 的 `z_and_losses()` 原先用
  `F.mse_loss(..., reduction='sum') / N` 计算 recon，而 `sum` 累加的是 `B × 768`
  个元素、却只除以 item 数 `N` ⇒ 该列是**逐元素均值的 768 倍**；
  同函数的 quant 列却是逐 item 均值（~1 倍）。**两列量纲不一致**，
  使 `total = recon + w × quant` 被 768 倍膨胀的 recon 主导。
- **后果**：`checkpoint_selection_behavior_auc_summary.md` §4 初版据此得出
  「best_loss 的 total loss 反而更大」的**错误结论**（真实结论相反）。
- **修正**：改为逐 batch `reduction='mean'` 后求和，与 `trainer._train_epoch`
  的 `total_recon_loss` / `total_quant` 完全同口径，数值可与训练日志逐行对照。
  已重算 18 个检查点的 5 个损失列并就地订正 summary §4（详见该节订正块）。
- **不影响**的部分：§4 的"best_loss 重建损失劣 35–47%"结论不变（同一量纲下比值不变）；
  §1–§3 的 AUC / 码结构结果全部未受影响（未触及这些代码路径）。

---

## 6. 结论：公平对照实验是否可重做

**当前不能直接用现有 checkpoint 重做**，但障碍的性质已明确：

- `best_loss_model.pth` **不合格不是因为逻辑错了，而是因为训练目标与下游解耦**。
  它执行了"总损失最小"这条准则，而该准则在本设置下恰好指向一个
  **重建更差、码本更塌缩**的早期状态（L0 utilization 0.06–0.31 vs 0.59–0.97）。
- **"另存一个总损失更低的更晚检查点"这条路已被证伪**——不存在这样的评估点。

**可行的替代判据（均需先确认是否改变实验定义，本轮不执行）**：

| 方案 | 说明 | 是否改变实验定义 |
|:--|:--|:--|
| ① 固定 epoch 对齐（如统一取 ep500/1000/2000） | 绕开"best"选择，直接比同一训练量的两个选点 | 改变（引入新选点规则） |
| ② 改用**重建损失**作选择准则 | recon 单调改善，"best recon"会落在晚期 | 改变（换判据） |
| ③ 改用下游行为选择 | 直接用 T5 效果选 | 改变（且引入 T5 训练开销） |

在此之前，设计 §6 的 A/B/C 三档**无法被干净判定**。
