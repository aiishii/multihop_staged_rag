#!/usr/bin/env python
"""
マルチホップQAの段階的RAGパイプライン実行スクリプト
"""

import argparse
import os
import json
import datetime
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union

from loguru import logger

from src.query_decomposition.query_splitter import QuerySplitter
from src.retrieval.process_db import DBEmbeddingSearch
from src.vector_search.process_faiss_shinra import faiss_search_shinra
from src.vector_search.process_faiss_wikidata import faiss_search_wikidata
from src.generation.answer_generator import AnswerGenerator


def get_shinra_results(query: str, question: str) -> List[Dict[str, Any]]:
    """
    森羅データに対する検索結果を取得
    
    Args:
        query: 検索クエリ
        question: 元の質問
        
    Returns:
        検索結果のリスト
    """
    # DBからの検索結果
    db_search = DBEmbeddingSearch(
        db_file='data/triple_database.db',
        model_name='hotchpotch/japanese-reranker-cross-encoder-small-v1'
    )
    
    # 主語と述語の抽出
    subject, attribute = extract_subject_attribute(query)
    
    # DB検索
    if subject and attribute:
        db_results = db_search.search(
            subject, 
            attribute, 
            table_name="shinra", 
            top_k=5, 
            use_template=True
        )
    else:
        db_results = pd.DataFrame()
    
    # FAISSベクトル検索
    faiss_results = faiss_search_shinra(query, top_k=5)
    
    # 結果を結合
    results = []
    
    if not db_results.empty:
        for _, row in db_results.iterrows():
            results.append((row['score'], f"db-{row['title']}", f"{row['title']}の{row['attribute']}は{row['value']}です"))
    
    results.extend(faiss_results)
    
    # スコアでソート
    results.sort(key=lambda x: x[0], reverse=True)
    
    # 上位5件を返す
    return results[:5]


def get_wikidata_results(query: str, question: str) -> List[Dict[str, Any]]:
    """
    Wikidataに対する検索結果を取得
    
    Args:
        query: 検索クエリ
        question: 元の質問
        
    Returns:
        検索結果のリスト
    """
    # DBからの検索結果
    db_search = DBEmbeddingSearch(
        db_file='data/triple_database.db',
        model_name='hotchpotch/japanese-reranker-cross-encoder-small-v1'
    )
    
    # 主語と述語の抽出
    subject, attribute = extract_subject_attribute(query)
    
    # DB検索
    if subject and attribute:
        db_results = db_search.search(
            subject, 
            attribute, 
            table_name="wikidata", 
            top_k=5, 
            use_template=True
        )
    else:
        db_results = pd.DataFrame()
    
    # FAISSベクトル検索
    faiss_results = faiss_search_wikidata(query, top_k=5)
    
    # 結果を結合
    results = []
    
    if not db_results.empty:
        for _, row in db_results.iterrows():
            results.append((row['score'], f"db-{row['title']}", f"{row['title']}の{row['attribute']}は{row['value']}です"))
    
    results.extend(faiss_results)
    
    # スコアでソート
    results.sort(key=lambda x: x[0], reverse=True)
    
    # 上位5件を返す
    return results[:5]


def extract_subject_attribute(query: str) -> Tuple[str, str]:
    """
    クエリから主語と属性を抽出
    
    Args:
        query: 検索クエリ
        
    Returns:
        (主語, 属性) の形式のタプル
    """
    # 「～の～は？」形式の場合
    parts = query.split('の')
    if len(parts) > 1 and '？' in parts[-1]:
        subject = parts[0]
        attribute = parts[-1].replace('？', '').replace('は', '').strip()
        return subject, attribute
    
    # それ以外の場合は空を返す
    return '', ''


def run_staged_rag(question: str) -> Dict[str, Any]:
    """
    段階的RAGによるマルチホップQAを実行
    
    Args:
        question: 質問
        
    Returns:
        処理結果を含む辞書
    """
    logger.info(f"質問: {question}")
    
    # 質問分解
    query_splitter = QuerySplitter()
    split_queries = query_splitter.split_question(question)
    logger.info(f"分解された質問: {split_queries}")
    
    # 各ステップの結果を格納するリスト
    step_results = []
    
    # 各クエリステップを処理
    for i, query_info in enumerate(split_queries):
        query = query_info["question"]
        logger.info(f"Step {i+1}: {query}")
        
        # Answerを含む質問は前の回答を使用するため、現段階ではスキップ
        if "Answer" in query:
            step_results.append({
                "question": query,
                "result": {"is_miss": True, "triple": "(TBD, TBD, TBD)"}
            })
            continue
        
        # Phase 1: 森羅RAG
        shinra_results = get_shinra_results(query, question)
        
        # トリプル生成
        generator = AnswerGenerator()
        shinra_triple = generator.generate_triple_from_search(query, shinra_results)
        
        # 森羅RAGで回答できない場合はPhase 2へ
        if shinra_triple["is_miss"]:
            logger.info("森羅RAGで回答できませんでした。Wikidata RAGを試行します。")
            
            # Phase 2: Wikidata RAG
            wikidata_results = get_wikidata_results(query, question)
            
            # トリプル生成
            wikidata_triple = generator.generate_triple_from_search(query, wikidata_results)
            
            # Wikidata RAGで回答できない場合はPhase 3へ
            if wikidata_triple["is_miss"]:
                logger.info("Wikidata RAGでも回答できませんでした。LLMのみの回答を生成します。")
                
                # Phase 3: LLM単体
                llm_result = generator.generate_staged_rag_answer(query, [])
                
                step_results.append({
                    "question": query,
                    "result": {
                        "triple": llm_result["derivations"][0] if llm_result["derivations"] else "(LLM, 回答, 生成)",
                        "is_miss": False,
                        "source": "llm"
                    }
                })
            else:
                # Phase 2で回答できた場合
                step_results.append({
                    "question": query,
                    "result": {
                        "triple": wikidata_triple["triple"],
                        "is_miss": wikidata_triple["is_miss"],
                        "source": "wikidata"
                    }
                })
        else:
            # Phase 1で回答できた場合
            step_results.append({
                "question": query,
                "result": {
                    "triple": shinra_triple["triple"],
                    "is_miss": shinra_triple["is_miss"],
                    "source": "shinra"
                }
            })
    
    # Answerを含む質問の処理（前の回答を代入）
    for i, step in enumerate(step_results):
        if "Answer" in step["question"]:
            # "Answer1", "Answer2" などを抽出
            answer_refs = [m for m in ["Answer1", "Answer2", "Answer3"] if m in step["question"]]
            
            if answer_refs:
                # 参照先を特定
                ref_idx = int(answer_refs[0][-1]) - 1
                if ref_idx < i and ref_idx >= 0:
                    # トリプルから目的語を抽出（3番目の要素）
                    ref_triple = step_results[ref_idx]["result"]["triple"]
                    value = extract_value_from_triple(ref_triple)
                    
                    # 質問内の参照を置換
                    new_question = step["question"]
                    for ref in answer_refs:
                        new_question = new_question.replace(ref, value)
                    
                    step_results[i]["question"] = new_question
    
    # 最終回答の生成
    final_result = generator.generate_final_answer(question, step_results)
    
    # 結果をまとめる
    return {
        "question": question,
        "split_queries": split_queries,
        "step_results": step_results,
        "answer": final_result["answer"],
        "derivations": final_result["derivations"]
    }


def extract_value_from_triple(triple_text: str) -> str:
    """
    トリプルから値（第3要素）を抽出
    
    Args:
        triple_text: トリプル文字列（例: "（エンティティ, 属性, 値）"）
        
    Returns:
        抽出された値
    """
    try:
        # 括弧を除去して分割
        content = triple_text.strip('（）()').split('，')
        if len(content) >= 3:
            return content[2].strip()
        return "参照エラー"
    except Exception:
        return "参照エラー"


def run_pipeline(questions: List[str], mode: str = "staged") -> List[Dict[str, Any]]:
    """
    複数の質問に対してパイプラインを実行
    
    Args:
        questions: 質問リスト
        mode: 処理モード（"staged", "shinra", "wikidata", "llm"）
        
    Returns:
        各質問の処理結果リスト
    """
    results = []
    
    for question in questions:
        if mode == "staged":
            result = run_staged_rag(question)
        else:
            logger.error(f"未対応のモード: {mode}")
            result = {"error": f"未対応のモード: {mode}"}
            
        results.append(result)
    
    return results


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="マルチホップQAのRAGパイプラインを実行")
    
    # 引数の設定
    parser.add_argument("--question", type=str, help="処理する質問")
    parser.add_argument("--input", type=str, help="質問リストファイル（TSVまたはJSON）")
    parser.add_argument("--mode", type=str, default="staged", 
                        choices=["staged", "shinra", "wikidata", "llm"],
                        help="実行モード（staged, shinra, wikidata, llm）")
    parser.add_argument("--output", type=str, help="出力ファイル（指定がなければ自動生成）")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの確認
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    
    questions = []
    
    # 質問の取得
    if args.question:
        questions = [args.question]
    elif args.input:
        ext = os.path.splitext(args.input)[1].lower()
        if ext == ".tsv":
            df = pd.read_csv(args.input, sep="\t")
            if "question" in df.columns:
                questions = df["question"].tolist()
            else:
                questions = df.iloc[:, 0].tolist()
        elif ext == ".json":
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], dict) and "question" in data[0]:
                    questions = [item["question"] for item in data]
                else:
                    questions = data
            elif isinstance(data, dict) and "questions" in data:
                questions = data["questions"]
        else:
            logger.error(f"未対応のファイル形式: {ext}")
            return
    else:
        logger.error("質問が指定されていません。--question または --input を指定してください。")
        return
    
    # パイプライン実行
    results = run_pipeline(questions, args.mode)
    
    # 出力ファイル名の決定
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output if args.output else f"{out_dir}/result_{args.mode}_{timestamp}.json"
    
    # 結果の保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"結果を {output_file} に保存しました。")
    
    # TSV形式でも保存
    tsv_output = os.path.splitext(output_file)[0] + ".tsv"
    
    with open(tsv_output, "w", encoding="utf-8") as f:
        f.write("qid\tquestion\tpredicted_answer\tpredicted_derivations\n")
        
        for i, result in enumerate(results):
            qid = f"q{i+1}"
            question = result["question"]
            answer = result.get("answer", "")
            derivations = ";".join(result.get("derivations", []))
            
            f.write(f"{qid}\t{question}\t{answer}\t{derivations}\n")
    
    logger.info(f"結果を {tsv_output} にもTSV形式で保存しました。")


if __name__ == "__main__":
    main()