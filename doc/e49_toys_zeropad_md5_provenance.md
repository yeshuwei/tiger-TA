# Toys ZeroPad 融合文件 md5 变更的 provenance 裁定（E49）

**日期**：2026-09-22
**结论**：`data/Toys/fused_item_emb_zeropad.parquet` 的 md5 变更 **是 parquet 写入器版本差异造成的字节级假象，数据未变**；下游两个 codes 产物经冻结 ckpt 复现验证 **零孤儿**。E47 结论不受影响。

---

## 1. 事件

E49 补 Toys Shuffled 消融数据时执行：

```bash
python rqvae/ablation_data.py --dataset Toys --seed 2025
```

该脚本除写出**目标产物** `behavior_emb_shuffled.parquet` / `fused_item_emb_shuffled.parquet` 外，还会重写 `fused_item_emb_zeropad.parquet`（E47 为 Toys 塌缩归因新建的输入）。md5 随之改变：

| | md5 | 写入器 |
|:--|:--|:--|
| 变更前（2026-09-04 建） | `b2239afa0a4b50b40e4d80f8f49c852a` | `parquet-cpp-arrow 22.0.0` |
| 变更后（2026-09-22） | `63892af79a4fdcf7c78a96e9dfbbcb37` | `parquet-cpp-arrow 24.0.0` |

原文件为 gitignore（`data/` 不入库），**已无备份**，无法按字节回滚。

## 2. 归因：写入器版本，而非数据

**直接证明**——对**未被触碰的** Sport 同名文件，用同一份数据在 `tiger_t5` 环境重写：

```
原始 data/Sport/fused_item_emb_zeropad.parquet : 03897499390f5587cc14473529948e12
同数据重写（当前环境）                          : 06457a4b8af1e9f30dca175786088198
```

数据逐元素 `0.000e+00` 差异，md5 仍然改变 → **md5 不是内容指纹**，随 `parquet-cpp-arrow` 版本漂移。

**根因**：环境里存在多个 conda env，parquet 写入器不同。`rqvae/logs/e4x_toys_zeropad_seed2024_train.log` 显示 09-04 的训练跑在 **base env**（`/home/media601/anaconda3/lib/python3.10/...`，arrow 22.0.0 + pandas 1.5.3），而本次用 `tiger_t5`（arrow 24.0.0 + pandas 2.3.3）。

**规范写入器已锚定**：base env 可**逐字节**复现 Sport 的 zeropad 文件（`03897499…` 完全一致）→ base env 是该项目 zeropad 文件的规范写入环境。

## 3. 数据同一性：冻结 ckpt 复现（决定性）

字节无法回滚，但可用**下游产物反证数据未变**。两个 codes 文件都是在该融合文件上训练的冻结 encoder 的输出，重算即可比对：

```bash
python rqvae/generate_sid_codes_param.py \
  --ckpt_path rqvae/ckpt/Toys_Zeropad_seed{2024,2025}/<ts>/best_collision_model.pth \
  --data_path data/Toys/fused_item_emb_zeropad.parquet \
  --output_file /tmp/... --device cuda:0
```

| codes 文件 | `np.array_equal` | L0 用码 | 碰撞率 |
|:--|:--|:--|:--|
| `Toys_zeropad_seed2024_sid.npy` | **True** | 163 / 163 | 0.00101 |
| `Toys_zeropad_seed2025_sid.npy` | **True** | 185 / 185 | 0.00034 |

全部 11924 行 × 4 列（含第 4 列冲突消解位）逐位相同 → 融合文件对**任何下游用途等价**。

**为什么这足以定案**：候选内容源在冻结 encoder 下的码分配互不相同（未 LN 的 768d 内容 norm=1.0，而 encoder 训练时见的是 norm≈27.6 的 LN 版；喂错源必然大规模翻码）。只有 `LN(item_emb_768d) ⊕ zeros` 能**完全**复现两套 codes，故原文件数据 = 该配方。

## 4. 配方一致性

- `item_emb_768d_contentonly.parquet` 本身就是 **LN 后的**内容（std≈0.996，row-norm≈27.6），故 `LN(item_emb_768d) ≡ contentonly`（数据级 md5 同为 `1fd98584…`）——e47 文档所述"**contentonly+128 零**"与本配方是同一个东西。
- 未被触碰的 Sport zeropad 内容段 = `LN(item_emb_768d)`（`0.000e+00`），与本次重生的内容段完全一致 → 两数据集同配方。
- `ablation_data.py` 只改行为向量，本脚本未触碰任何 behavior/fused 主产物。

## 5. 裁定与后续纪律

1. **不需要回滚、不需要重训**：`Toys_zeropad_seed{2024,2025}_sid.npy` 与 E47 结论（ZeroPad@2024=163 健康）全部有效，零孤儿。
2. **不重命名、不复制**（e48 orphan 事故教训）。
3. **md5 不作为数据指纹使用**：跨 env 比对必须用**数据级指纹**（float 数组 md5）或下游产物复现。本仓库已知写入器：base env arrow 22.0.0 / `tiger_t5` arrow 24.0.0。
4. **后续所有 E49 新产物的 md5 记录，须同时记录写入 env**。

## 6. 证据文件

- 本文档（裁定）
- `doc/e47_seed_health_isolation.md` §5（该文件的创建记录）
- `rqvae/logs/e4x_toys_zeropad_seed{2024,2025}_train.log`（base env 证据）
