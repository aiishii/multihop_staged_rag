#!/usr/bin/env python
"""
FAISSインデックス作成スクリプト
"""

import os
import argparse
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Optional

def load_data(data_path: str) -> pd.DataFrame:
    """
    データをロード
    
    Args:
        data_path: データファイルのパス（TSV, CSV, JSON）
        
    Returns:
        データを格納したDataFrame
    """
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext == '.tsv':
        df = pd.read_csv(data_path, sep='\t', encoding='utf-8', on_bad_lines='skip')
    elif ext == '.csv':
        df = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')
    elif ext == '.json':
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"未対応のファイル形式: {ext}")
    
    return df

def prepare_sentences(df: pd.DataFrame, template: str = "{title}の{attribute}は{value}。") -> List[str]:
    """
    インデックス作成用の文を準備
    
    Args:
        df: データを格納したDataFrame
        template: 文生成のテンプレート
        
    Returns:
        生成された文のリスト
    """
    sentences = []
    for _, row in df.iterrows():
        try:
            # 値の引用符を除去
            if 'value' in row and isinstance(row['value'], str):
                value = row['value'].strip('"')
            else:
                value = row.get('value', '')
            
            # テンプレートに当てはめて文を生成
            sentence = template.format(
                title=row.get('title', ''),
                attribute=row.get('attribute', ''),
                value=value
            )
            sentences.append(sentence)
        except Exception as e:
            print(f"文の生成中にエラーが発生しました: {e}")
            sentences.append("")
    
    return sentences

def create_index(sentences: List[str], 
                output_dir: str, 
                model_name: str = "intfloat/multilingual-e5-small", 
                prefix: str = "passage: ") -> None:
    """
    FAISSインデックスを作成
    
    Args:
        sentences: インデックス化する文のリスト
        output_dir: 出力ディレクトリ
        model_name: 埋め込みモデル名
        prefix: 文のプレフィックス
    """
    # モデルのロード
    model = SentenceTransformer(model_name)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 埋め込みの計算
    print("文の埋め込みを計算中...")
    embeddings = []
    batch_size = 64
    
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        batch_with_prefix = [prefix + text for text in batch]
        batch_embeddings = model.encode(batch_with_prefix, normalize_embeddings=True)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # FAISSインデックスの作成
    print("FAISSインデックスを作成中...")
    d = embeddings.shape[1]  # 埋め込みの次元数
    
    # インデックスパラメータの設定
    nlist = min(1024, len(sentences) // 10)  # クラスタ数
    m = min(32, d // 4)  # PQの部分数
    
    # IVF + PQ インデックスの作成
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    
    # インデックスの学習
    print("インデックスを学習中...")
    index.train(embeddings)
    
    # ベクトルをインデックスに追加
    print("ベクトルをインデックスに追加中...")
    index.add(embeddings)
    
    # インデックスの保存
    index_dir = os.path.join(output_dir, f"{os.path.basename(model_name)}-passage")
    os.makedirs(index_dir, exist_ok=True)
    
    index_path = os.path.join(index_dir, f"index_IVF{nlist}_PQ{m}.faiss")
    faiss.write_index(index, index_path)
    
    print(f"インデックスを {index_path} に保存しました")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="FAISSインデックスを作成")
    
    parser.add_argument("--data", required=True, help="インデックス化するデータファイル")
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small", help="埋め込みモデル")
    parser.add_argument("--template", default="{title}の{attribute}は{value}。", help="文生成テンプレート")
    
    args = parser.parse_args()
    
    # データのロード
    print(f"データをロード中: {args.data}")
    df = load_data(args.data)
    
    # 文の準備
    print("インデックス用の文を準備中...")
    sentences = prepare_sentences(df, args.template)
    
    # インデックスの作成
    create_index(sentences, args.output, args.model)

if __name__ == "__main__":
    main()