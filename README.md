# 構造化知識・文書ベース RAG を段階的に利用したマルチホップQA(JEMHopQA)RAG

> **⚠️ 注意: このリポジトリは現在作成中です**  
> このプロジェクトは整備段階にあり、現時点では一部の機能やドキュメントが不完全である可能性があります。

このリポジトリは論文「構造化知識 RAG・文書ベース RAG を段階的に利用したマルチホップ QA に対する LLM の精度向上」のソースコードを提供します。

## 概要

LLM のハルシネーション（事実と矛盾する情報生成の課題）に対し、構造化・非構造化知識源を用いた Retrieval-Augmented Generation（RAG）の比較分析を行いました。マルチホップ QA データセットを用いて、構造化知識ベース、文書ベースの RAG および LLM のみによる回答を段階的に組み合わせる手法により、各知識源の特性を活かした回答生成の有効性を実証しています。

## 主な特徴

- 段階的 RAG アプローチ（構造化知識 → 文書ベース → LLM）
- マルチホップ質問の分解と処理
- 構造化知識（Wikidata、森羅）と文書ベース（Wikipedia）の活用
- 回答の正確性と根拠情報の評価

## インストール方法

```bash
git clone https://github.com/aiishii/JEMHopQA.git
cd JEMHopQA
pip install -r requirements.txt
```

## 使用方法

### 段階的 RAG パイプラインの実行

```bash
python scripts/run_pipeline.py --question "質問文" --mode staged
```

### 評価の実行

```bash
python scripts/evaluate_results.py --prediction_file output.tsv
```

## データセット

JEMHopQA データセット（日本語の説明可能なマルチホップ質問応答データセット）を使用しています。データセットの詳細とアクセス方法は [こちら](https://github.com/aiishii/JEMHopQA/data/README.md) を参照してください。

## 引用

本研究を引用する場合は、以下の形式を使用してください：

```
@inproceedings{ishii2025structured,
  title={構造化知識 RAG・文書ベース RAG を段階的に利用したマルチホップ QA に対する LLM の精度向上},
  author={石井愛 and 井之上直也 and 鈴木久美 and 関根聡},
  booktitle={言語処理学会第31回年次大会発表論文集},
  year={2025},
  month={3}
}
```

## ライセンス

本リポジトリの著作権は[理化学研究所](https://www.riken.jp/)に帰属し、 [クリエイティブ・コモンズ 表示 - 継承 4.0 国際 ライセンス (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt)*の条件のもとに、利用・再配布が許諾されます。*

  ![https://creativecommons.org/licenses/by-sa/4.0/legalcode](https://i.imgur.com/7HLJWMM.png)

本リポジトリの構築の一部はJSPS 科研費 19K20332 の助成を受けたものです。記して感謝いたします。
