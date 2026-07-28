"""
generate_content_embeddings.py — 从原始 Amazon 评论 JSON 生成 item content embeddings

用法:
    python tools/generate_content_embeddings.py --dataset Sport   # Sentence-T5-base (768d)
    python tools/generate_content_embeddings.py --dataset Toys

输出:
    {raw_data_dir}/{dataset}/content_embeddings_768d.pkl
    data/{dataset}/item_emb_768d.parquet
"""

import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 768d content embeddings from raw Amazon review JSON"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Beauty", "Sport", "Toys"],
                        help="Dataset name")
    parser.add_argument("--raw_data_dir", type=str,
                        default="/data/gtx/project/datasets/tiger_beauty",
                        help="Root dir containing raw dataset directories")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory (relative to project root)")
    parser.add_argument("--model_name", type=str,
                        default="sentence-transformers/sentence-t5-base",
                        help="Sentence-Transformer model (default: 768d)")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    raw_dir = os.path.join(args.raw_data_dir, ds)
    out_dir = os.path.join(args.output_dir, ds)
    os.makedirs(out_dir, exist_ok=True)

    # 确定 JSON 文件名
    json_name_map = {
        "Beauty": "Beauty_5.json",
        "Sport": "Sports_and_Outdoors_5.json",
        "Toys": "Toys_and_Games_5.json",
    }
    json_path = os.path.join(raw_dir, json_name_map.get(ds, f"{ds}.json"))
    if not os.path.exists(json_path):
        json_path = os.path.join(raw_dir, f"{ds}_5.json")

    print(f"Dataset: {ds}")
    print(f"Raw JSON: {json_path}")
    print(f"Model: {args.model_name}")
    print(f"Output: {out_dir}/")

    # ====== 1. 读取 JSON，提取唯一 ASIN 并聚合文本 ======
    print("\n[1/4] 读取评论数据，提取唯一商品文本...")

    # 用 OrderedDict 保持首次出现的顺序
    asin_texts = OrderedDict()

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
                asin = review["asin"]
                summary = review.get("summary", "")
                review_text = review.get("reviewText", "")
                # 聚合文本：summary + reviewText
                combined = f"{summary} {review_text}".strip()
                if combined:
                    if asin not in asin_texts:
                        asin_texts[asin] = []
                    asin_texts[asin].append(combined)
            except (KeyError, json.JSONDecodeError):
                continue

    num_items = len(asin_texts)
    print(f"  ✓ 共 {num_items} 个唯一商品")

    # ====== 2. 构建 ASIN → 整数 ID 映射 ======
    print("\n[2/4] 构建 ASIN → ItemID 映射...")
    asin_to_id = {}  # ASIN → 0-indexed ID (匹配 content_embeddings.pkl 格式)
    for i, asin in enumerate(asin_texts.keys()):
        asin_to_id[asin] = i  # 0-indexed

    # 每个商品的全部文本拼接
    item_texts = []
    for asin in asin_texts:
        full_text = " ".join(asin_texts[asin])
        # 限制最大长度（Sentence-T5 支持 512 token, 约 2000 字符）
        item_texts.append(full_text[:5000])

    print(f"  ✓ 文本准备完成，平均长度: {np.mean([len(t) for t in item_texts]):.0f} 字符")

    # ====== 3. 加载模型，生成 Embeddings ======
    print("\n[3/4] 加载 Sentence-T5-base (768d) 并生成嵌入...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    model = SentenceTransformer(args.model_name, device=device)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  模型维度: {emb_dim}d")

    # 批次生成
    batch_size = 256
    all_embeddings = []
    for i in range(0, len(item_texts), batch_size):
        batch = item_texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"  进度: {i}/{len(item_texts)} ({100*i//len(item_texts)}%)")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"  ✓ 嵌入完成: {embeddings.shape}")
    print(f"    均值={embeddings.mean():.4f}, 标准差={embeddings.std():.4f}")

    # ====== 4. 保存 ======
    print("\n[4/4] 保存文件...")

    # 4a. 保存 content_embeddings_768d.pkl
    pkl_path = os.path.join(raw_dir, "content_embeddings_768d.pkl")
    pkl_data = {
        "item_id": list(range(num_items)),  # 0-indexed
        "embedding": [embeddings[i] for i in range(num_items)],
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    print(f"  ✓ {pkl_path}")
    print(f"    shape: ({num_items}, {emb_dim})")

    # 4b. 保存 item_emb_768d.parquet
    item_ids_1based = list(range(1, num_items + 1))
    emb_list = [embeddings[i] for i in range(num_items)]
    item_emb_df = pd.DataFrame({
        "ItemID": item_ids_1based,
        "embedding": emb_list,
    })
    parquet_path = os.path.join(out_dir, "item_emb_768d.parquet")
    item_emb_df.to_parquet(parquet_path, index=False)
    print(f"  ✓ {parquet_path}")
    print(f"    shape: ({num_items}, {emb_dim})")

    # 4c. 验证
    verify_df = pd.read_parquet(parquet_path)
    print(f"\n  验证: item_emb_768d.parquet")
    print(f"    rows={len(verify_df)}, dim={len(verify_df['embedding'].iloc[0])}")

    print("\n[Done] 768d content embeddings 生成完成!")


if __name__ == "__main__":
    main()
