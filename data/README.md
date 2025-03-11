# JEMHopQA データについて

このディレクトリには、段階的RAGシステムで使用するデータファイルを配置します。

## 必要なデータファイル

### データベースファイル

* `triple_database.db`: SQLiteデータベースファイル（構造化知識トリプルを格納）

### インデックスファイル

* `indices/shinra2022-attribute/multilingual-e5-small-passage/index_IVF1024_PQ32.faiss`: 森羅データのFAISSインデックス
* `indices/wikidata-20210823-all/multilingual-e5-small-passage/index_IVF1024_PQ32.faiss`: Wikidataの FAISSインデックス

### データセットファイル

* `shinra2022-attribute.tsv`: 森羅の属性データ（TSV形式）
* `wikidata-20210823-all.tsv`: Wikidataのトリプルデータ（TSV形式）
* `corpus_ver1.1/dev_ver1.1_fix20241017.json`: JEMHopQA開発セット
* `wikipedia_disambig_frequency.tsv`: Wikipedia曖昧性解消語の頻度データ

## データ準備手順

### 1. 森羅データのダウンロード

森羅プロジェクトデータは公式サイトからダウンロードできます：
http://shinra-project.info/

```bash
# 属性データの準備（公式サイトから）
wget -O shinra2022-attribute.zip http://shinra-project.info/download/latest/attribute-data.zip
unzip shinra2022-attribute.zip
```

### 2. Wikidataの準備

特定の日付のWikidataダンプをダウンロードし、処理する必要があります。

```bash
# Wikidataダンプのダウンロード例（20210823版）
wget https://dumps.wikimedia.org/wikidatawiki/20210823/wikidata-20210823-all.json.gz

# 処理スクリプト（別途用意が必要）を実行
python scripts/process_wikidata_dump.py wikidata-20210823-all.json.gz
```

### 3. インデックスの作成

```bash
# FAISSインデックスの作成
python scripts/create_indices.py --data shinra2022-attribute.tsv --output indices/shinra2022-attribute
python scripts/create_indices.py --data wikidata-20210823-all.tsv --output indices/wikidata-20210823-all
```

### 4. データベースの作成

```bash
# SQLiteデータベースの作成
python scripts/create_database.py --shinra shinra2022-attribute.tsv --wikidata wikidata-20210823-all.tsv --output triple_database.db
```

## JEMHopQAデータセット

JEMHopQAデータセットは、日本語の説明可能なマルチホップ質問応答データセットです。各質問に対して、回答と導出プロセス（複数ステップのトリプル）が含まれています。

データセットの詳細については、以下の論文を参照してください：

```
@inproceedings{ishii2024jemhopqa,
  title={JEMHopQA: Dataset for Japanese Explainable Multi-hop Question Answering},
  author={Ishii, Ai and Inoue, Naoya and Suzuki, Hisami and Sekine, Satoshi},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  pages={9515--9525},
  year={2024}
}
```

## サンプルデータ

各インデックスとデータセットの小規模サンプルが `sample/` ディレクトリに含まれています。完全な評価を行う場合は、上記の手順で完全なデータセットを準備する必要があります。
