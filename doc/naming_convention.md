# TA-SID 项目文件命名规范（Naming Convention）

> **生效日期：2026-08-03。** 所有新产生的实验文件必须遵循本规范。
> 背景：2026-07 期间文件命名存在多套旧规范混用（seed42 / 2025 / 768d_seed2025 / 768d_{ds}_seed{seed}，大小写与元素顺序不一致），导致文件难以定位。本规范统一之。
> **查找文件第一原则：先查训练日志里的 `save_path` / `code_path` 配置，再按实际命名找，勿只按新规范猜。**

---

## 1. 命名元素

文件名由 4 个元素按固定顺序组合：**`{前缀}_{方法tag}_{数据集}_{seed}`**。

- **前缀（prefix）：** 标识文件类型与框架（`tiger` = T5 生成模型，`{Ds}` = 数据集名）。
- **方法 tag（method tag）：** 标识实验配置，**必须包含内容维度**（项目存在多维度版本，显式标注防混淆）。

### 方法 tag 总表

| tag | 含义 | 输入 | 状态 |
|-----|------|------|:----:|
| `ta_sid_768d` | **TA-SID 正式版** | 768d content + 128d behavior（896d fused） | ✅ 主实验 |
| `ta_sid_4096d` | TA-SID 维度对照 | 4096d content + behavior（4224d fused） | 鲁棒性对照 |
| `content768d_baseline` | **Content-only baseline** | 768d content，无 behavior | ✅ 公平基线 |
| `random_sid` | Random SID 消融 | 随机码 | 码质量消融 |
| `zeropad_ablation` | ZeroPad 消融 | 768d content + 128 零（死维度） | 消融 |
| `shuffled_ablation` | Shuffled 消融 | 768d content + 错配 behavior | 消融 |
| `random_ablation` | Random noise 消融 | 768d + 谱匹配噪声 | ⚠️ 已弃用 |
| `track_b_768d` | L_trans 对比损失 | 768d + behavior + InfoNCE | ❌ 已证无效 |

- **seed：** 正式报告 = **2025 / 2026 / 2027** 三 seed。**禁止省略 seed 后缀**（省略 = 旧规范遗留，一律视为 seed2025 并应补全）。
- **数据集：** 路径目录用 `Sport/Beauty/Toys`；文件名内小写 `sport/beauty/toys`。

---

## 2. 文件类型规则

| 文件类型 | 路径规则 | 示例 |
|----------|----------|------|
| **T5 模型** | `model/ckpt/tiger_{tag}_{ds_lower}_seed{seed}.pth` | `model/ckpt/tiger_ta_sid_768d_beauty_seed2025.pth` |
| **T5 训练日志** | `model/logs/tiger_{tag}_{ds_lower}_seed{seed}.log` | `model/logs/tiger_ta_sid_768d_beauty_seed2025.log` |
| **T5 评测日志** | `model/logs/{Ds}_{tag}_seed{seed}_eval.log` | `model/logs/Beauty_ta_sid_768d_seed2025_eval.log` |
| **SID code** | `data/{Ds}/{Ds}_{tag}_sid.npy` | `data/Beauty/Beauty_ta_sid_768d_sid.npy` |
| **商品嵌入** | `data/{Ds}/{Ds}_{tag}_item_emb.parquet` | `data/Beauty/Beauty_content768d_baseline_item_emb.parquet` |
| **RQ-VAE 目录** | `rqvae/ckpt/{Ds}_{Tag}/`（Tag 用 tag 的 PascalCase） | `rqvae/ckpt/Beauty_TA_SID_768d/` |
| **RQ-VAE 训练日志** | `rqvae/logs/{Ds}_{tag}_rqvae.log` | `rqvae/logs/Beauty_ta_sid_768d_rqvae.log` |
| **code 生成脚本** | `rqvae/generate_code_{Ds}_{tag}.py` | `rqvae/generate_code_Beauty_ta_sid_768d.py` |

**约定：**
- 所有小写字母：tag、ds_lower、seed。
- 数据集目录名（`data/Sport/`）保持首字母大写（与 Amazon 数据集官方名一致）。
- RQ-VAE 目录内子目录是训练时间戳（如 `Jul-15-2026_23-07-40/`），由 trainer 自动生成，不手动改。
- 模型文件内部搜索用 `best_collision_model.pth` / `best_model.pth`，文件名不随实验变。

---

## 3. 新旧名对照（正式实验，2026-08-03 已执行）

> ✅ **本次重命名已于 2026-08-03 完成**，映射存档见 `doc/naming_mapping.md`。**严禁直接用旧名找文件。**

### 3.1 TA-SID 主实验（ta_sid_768d）—— ✅ 全部符合规范

| 类型 | 文件（新名） |
|------|------|
| T5 ckpt (3-seed × 3 数据集) | `model/ckpt/tiger_ta_sid_768d_{sport,beauty,toys}_seed{2025,2026,2027}.pth` |
| eval log (3-seed × 3) | `model/logs/{Sport,Beauty,Toys}_ta_sid_768d_seed{2025,2026,2027}_eval.log` |
| SID code | `data/{Sport,Beauty,Toys}/{Ds}_ta_sid_768d_sid.npy` |
| T5 训练日志 | `model/logs/tiger_ta_sid_768d_{ds}_seed{seed}.log`（seed2025 为历史名 `tiger_ta_sid{_ds}_768d.log`，未改） |

### 3.2 Content-only baseline（content768d_baseline）—— ✅ 全部符合规范

| 类型 | 文件（新名） |
|------|------|
| T5 ckpt (3-seed × 3) | `model/ckpt/tiger_content768d_baseline_{ds}_seed{2025,2026,2027}.pth` |
| eval log (3-seed × 3) | `model/logs/{Ds}_content768d_baseline_seed{2025,2026,2027}_eval.log` |
| SID code | `data/{Ds}/{Ds}_content768d_baseline_sid.npy`（未改动，已符合） |

### 3.3 其余（seed42 / 4096d / track_b / ablation / test）

> **历史归档，不重命名**，仅按 `doc/naming_mapping.md` 索引。它们是 seed42 旧口径或已弃用实验，不进论文；如需复现走对应脚本。

---

## 4. 禁止事项

1. **禁止**再出现无 seed 后缀 / 大小写混用 / 元素顺序乱序的文件名（`tiger_ta_sid_sport_768d.pth`、`tiger_ta_sid_Toys_768d.pth` 这类）。
2. **禁止**手工起名——新实验必须走统一脚本（`rqvae/run_multiseed.py` / `rqvae/run_content_baseline.py`）由代码生成路径。
3. **禁止**覆盖既有文件——每次实验用新 seed/新 tag，保留旧文件（早期教训：Beauty 768d content 被覆盖后无法再生成）。

---

## 5. 脚本引用现状（需同步）

- `rqvae/run_multiseed.py`：`METHODS` dict 引用 code 路径 → 需按新 code 名更新
- `rqvae/run_content_baseline.py`：已用规范路径（`tiger_content768d_baseline_{ds_lower}.pth`）→ 补 `_seed2025`
- `tools/sid_consistency_contrast.py`：`CONFIG` dict 引用 ckpt/code → 需按新名更新
- `doc/*.md`：9 个文档引用具体文件名 → 整理后统一更新
