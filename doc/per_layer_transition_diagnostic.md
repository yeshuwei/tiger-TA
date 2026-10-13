# Per-layer 转移结构保留诊断

> **⚠️ 修订记录（2026-10-13）：本文 Beauty 段的数值由错误的索引映射产生，勿引用。**
>
> **成因**：本报告由 `tools/sid_per_layer_diagnostic.py` 生成，其 `pair_ab` 用
> `[(a - 1, b - 1) for (a, b), c in pairs]` 从 ItemID 取矩阵行号。该写法隐含假设
> **`ItemID == 行号 + 1`** —— 此假设**只在 Sport / Toys 成立**：
> `data/Beauty/item_emb_768d.parquet` 的 ItemID 是 1..12101 的一个**排列**，
> 4158/12101 = **34.36% 的行错位**（例：row 8 存的是 ItemID 10）。
> 于是 Beauty 的每一对都被映射到**另一个商品**上（1000 对里仅 308 对正确）。
>
> **实测影响**（Beauty TA-SID，同一脚本口径重算）：`input` 0.9050→**0.9954**、
> `z` 0.8688→**0.9575**、`xq[L0]` 0.8485→**0.9225**、`xq[L0+L1]` 0.8675→**0.9454**、
> `xq[full]` 0.8687→**0.9495**。**三条结论标签均不翻转**（input→full 仍为"小幅度变化"、
> z→full 仍为"量化环节无丢失"、分层仍为"粗层部分保留，细层补充"），仅数值整体上移 ~0.07–0.09。
> 本文 Beauty 段的 Content-only 行同样受影响，但未重算。**Sport / Toys 两段的数值不受影响**
> （两数据集 ItemID 恰好连续，两法逐位等价，实测 Δ < 3e-5）。
>
> **附注**：脚本内的 `KNOWN_AUC` 自检锚点（Beauty TA-SID 0.869）也是按同一错值写死的，
> 故当时自检报"✓ sanity OK" —— 属自洽地错，不能作为该数正确的证据。
>
> **处置（2026-10-13）**：5 处 `a-1` 写法已全部改为显式 `pairs_to_rows()` 映射
> （`sid_consistency_contrast.py` / `sid_per_layer_diagnostic.py` / `sid_capacity_diagnostic.py`
> / `intrinsic_capacity.py` / `threshold_sensitivity.py`）。**本文不重跑**：本报告属已被
> E46/E47 推翻的"第二创新"线，无任何论文主张依赖它。正确口径的 Beauty 数字见
> [behavior_structure_probe_results.csv](behavior_structure_probe_results.csv)
> （TA-SID `xq[full]`/topk = 0.9560，其路径本就用显式 `id2row` 映射，未受影响）。

```text
====================================================================================================
Per-layer 转移结构保留诊断（transition-structure preservation across RQ-VAE layers）
运行时间: 2026-08-13 11:20:26
转移对: top-1000 高频相邻对 | 随机对: 5000 (seed=0)
AUC = P(转移对 cos > 随机对 cos)，0.5=随机，1=完美
====================================================================================================

### Beauty  |  top-1000 高频转移对

  ▸ TA-SID  (Beauty_TA_SID)
    codes=(12101, 4) | codebook 3 层, e_dim=32
    input(fused_item_emb.parquet): AUC=0.9050  转移cos=0.8487 随机cos=0.7370 Δ=+0.1116
    z (编码器输出): AUC=0.8688  转移cos=0.7166 随机cos=0.3486 Δ=+0.3679
    xq[L0]: AUC=0.8485  转移cos=0.8229 随机cos=0.4583 Δ=+0.3646
    xq[L0+L1]: AUC=0.8675  转移cos=0.7853 随机cos=0.4107 Δ=+0.3747
    xq[full]: AUC=0.8687  转移cos=0.7592 随机cos=0.3923 Δ=+0.3669

  ▸ Content-only  (Beauty_Content768d_Baseline)
    codes=(12101, 4) | codebook 3 层, e_dim=32
    input(item_emb_768d_contentonly.parquet): AUC=0.7616  转移cos=0.8933 随机cos=0.8565 Δ=+0.0367
    z (编码器输出): AUC=0.7433  转移cos=0.7532 随机cos=0.6184 Δ=+0.1348
    xq[L0]: AUC=0.7469  转移cos=0.8407 随机cos=0.7157 Δ=+0.1251
    xq[L0+L1]: AUC=0.7525  转移cos=0.8018 随机cos=0.6709 Δ=+0.1309
    xq[full]: AUC=0.7534  转移cos=0.7876 随机cos=0.6558 Δ=+0.1318

### Sport  |  top-1000 高频转移对

  ▸ TA-SID  (Sport_TA_SID_768d)
    codes=(18357, 4) | codebook 3 层, e_dim=32
    input(fused_item_emb_768d.parquet): AUC=0.9701  转移cos=0.8301 随机cos=0.7235 Δ=+0.1066
    z (编码器输出): AUC=0.9645  转移cos=0.7679 随机cos=0.2986 Δ=+0.4693
    xq[L0]: AUC=0.9459  转移cos=0.8761 随机cos=0.3946 Δ=+0.4816
    xq[L0+L1]: AUC=0.9607  转移cos=0.8460 随机cos=0.3561 Δ=+0.4899
    xq[full]: AUC=0.9621  转移cos=0.8240 随机cos=0.3418 Δ=+0.4822

  ▸ Content-only  (Sport_Content768d_Baseline)
    codes=(18357, 4) | codebook 3 层, e_dim=32
    input(item_emb_768d_contentonly.parquet): AUC=0.9027  转移cos=0.9048 随机cos=0.8404 Δ=+0.0644
    z (编码器输出): AUC=0.8983  转移cos=0.7545 随机cos=0.4367 Δ=+0.3178
    xq[L0]: AUC=0.8881  转移cos=0.8395 随机cos=0.5188 Δ=+0.3207
    xq[L0+L1]: AUC=0.8951  转移cos=0.8078 随机cos=0.4837 Δ=+0.3241
    xq[full]: AUC=0.8999  转移cos=0.7943 随机cos=0.4710 Δ=+0.3233

### Toys  |  top-1000 高频转移对

  ▸ TA-SID  (Toys_TA_SID_768d)
    codes=(11924, 4) | codebook 3 层, e_dim=32
    input(fused_item_emb_768d.parquet): AUC=0.9788  转移cos=0.8603 随机cos=0.7393 Δ=+0.1210
    z (编码器输出): AUC=0.9652  转移cos=0.8156 随机cos=0.3346 Δ=+0.4810
    xq[L0]: AUC=0.8535  转移cos=0.8007 随机cos=0.3934 Δ=+0.4073
    xq[L0+L1]: AUC=0.8739  转移cos=0.7686 随机cos=0.3584 Δ=+0.4102
    xq[full]: AUC=0.8719  转移cos=0.7386 随机cos=0.3422 Δ=+0.3964

  ▸ Content-only  (Toys_Content768d_Baseline)
    codes=(11924, 4) | codebook 3 层, e_dim=32
    input(item_emb_768d_contentonly.parquet): AUC=0.8519  转移cos=0.9100 随机cos=0.8594 Δ=+0.0506
    z (编码器输出): AUC=0.8601  转移cos=0.7598 随机cos=0.5139 Δ=+0.2459
    xq[L0]: AUC=0.8469  转移cos=0.8557 随机cos=0.6298 Δ=+0.2260
    xq[L0+L1]: AUC=0.8612  转移cos=0.8179 随机cos=0.5803 Δ=+0.2377
    xq[full]: AUC=0.8632  转移cos=0.7989 随机cos=0.5582 Δ=+0.2406

====================================================================================================
汇总表（AUC，越高越好；0.5=随机）
====================================================================================================
Dataset  Method         input       z  xq[L0] xq[L0+L1]  xq[full]  note
-----------------------------------------------------------------------
Beauty   TA-SID        0.9050  0.8688  0.8485    0.8675    0.8687  ✓ sanity OK
Beauty   Content-only  0.7616  0.7433  0.7469    0.7525    0.7534  ✓ sanity OK
Sport    TA-SID        0.9701  0.9645  0.9459    0.9607    0.9621  ✓ sanity OK
Sport    Content-only  0.9027  0.8983  0.8881    0.8951    0.8999  ✓ sanity OK
Toys     TA-SID        0.9788  0.9652  0.8535    0.8739    0.8719  ✓ sanity OK
Toys     Content-only  0.8519  0.8601  0.8469    0.8612    0.8632  ✓ sanity OK

====================================================================================================
诊断解读（判决第二创新方向是否值得继续）
====================================================================================================

── Beauty (TA-SID) ──
  input→xq[full]: 0.905 → 0.869  (Δ-0.036)  ~ 小幅度变化
  z→xq[full]: 0.869 → 0.869  (Δ-0.000)  ✓ 量化环节无丢失
  分层: L0=0.849  L0+L1=0.868  full=0.869  → 粗层部分保留，细层补充

── Sport (TA-SID) ──
  input→xq[full]: 0.970 → 0.962  (Δ-0.008)  ✓ 量化基本保留结构
  z→xq[full]: 0.965 → 0.962  (Δ-0.002)  ✓ 量化环节无丢失
  分层: L0=0.946  L0+L1=0.961  full=0.962  → 结构主要活在粗层

── Toys (TA-SID) ──
  input→xq[full]: 0.979 → 0.872  (Δ-0.107)  ⚠ 量化显著丢失结构
  z→xq[full]: 0.965 → 0.872  (Δ-0.093)  ⚠ 量化环节丢失结构
  分层: L0=0.853  L0+L1=0.874  full=0.872  → 结构主要活在粗层

判定规则：
  [1] 若 xq[L0] 即≈xq[full]（粗层已保留全部结构）→ 不存在'量化破坏'问题 → 增强损失无立足点
  [2] 若 z 高而 xq[full] 低 → 丢失发生在量化环节 → 增强方向有真实靶点
  [3] 若 input 就低 → 转移信号在输入表征里本来就弱 → 是信号/冗余问题，非量化问题
```
