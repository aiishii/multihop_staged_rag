"""
回答生成モジュール
"""

import json
import re
from typing import List, Dict, Tuple, Optional, Any, Union
import openai
from loguru import logger

from prompts import (
    get_simple_rag_template,
    get_final_answer_template,
    get_llm_only_template,
    get_staged_rag_template
)


class AnswerGenerator:
    """
    マルチホップQAの回答を生成するクラス
    """
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        """
        初期化
        
        Args:
            model_name: 使用するLLMのモデル名
        """
        self.model_name = model_name
        self.client = openai.OpenAI()
        
    def generate_triple_from_search(self, 
                                    question: str, 
                                    search_results: List[Any]) -> Dict[str, Any]:
        """
        検索結果からトリプルを生成
        
        Args:
            question: 質問
            search_results: 検索結果のリスト
            
        Returns:
            生成されたトリプル情報を含む辞書
        """
        try:
            # 検索結果を文字列に変換
            search_results_text = "\n".join([str(result) for result in search_results])
            
            # プロンプト作成
            prompt = get_simple_rag_template(question, search_results_text)
            
            # LLM呼び出し
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            response_text = response.choices[0].message.content
            
            # レスポンスからトリプルを抽出
            return self._parse_triple_response(response_text)
            
        except Exception as e:
            logger.error(f"トリプル生成中にエラーが発生しました: {e}")
            return {"triple": f"（エラー，エラー，{str(e)}）", "is_miss": True}
    
    def _parse_triple_response(self, response_text: str) -> Dict[str, Any]:
        """
        トリプル形式のレスポンスを解析
        
        Args:
            response_text: LLMからのレスポンステキスト
            
        Returns:
            解析したトリプル情報を含む辞書
        """
        cleaned_text = response_text.strip()
        
        # "=> " で分割して後ろの部分を取得
        if "=>" in cleaned_text:
            triple_text = cleaned_text.split("=>", 1)[1].strip()
        else:
            triple_text = cleaned_text
            
        # "NOTFOUND" を含む場合は"見つからない"と判定
        is_miss = "NOTFOUND" in triple_text
        
        return {
            "triple": triple_text,
            "is_miss": is_miss
        }
    
    def generate_final_answer(self, 
                              original_question: str, 
                              query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        クエリ結果から最終的な回答を生成
        
        Args:
            original_question: 元の質問
            query_results: クエリ結果のリスト。各要素は {"question": 質問, "result": トリプル} の形式
            
        Returns:
            {"answer": 回答, "derivations": 根拠となるトリプルのリスト} の形式の辞書
        """
        try:
            # 根拠トリプルの取得
            derivations = []
            for result in query_results:
                if not result.get("result", {}).get("is_miss", True):
                    derivations.append(result["result"]["triple"])
            
            # 最終質問を取得（通常は最後のクエリ）
            final_question = original_question
            if query_results and len(query_results) > 0:
                last_query = query_results[-1]
                if "Answer" not in last_query["question"]:
                    final_question = last_query["question"]
            
            # 根拠が1つも見つからない場合はLLMのみで回答
            if not derivations:
                return self.generate_answer_llm_only(original_question)
            
            # プロンプト作成
            prompt = get_final_answer_template(final_question, derivations)
            
            # LLM呼び出し
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "derivations": derivations
            }
            
        except Exception as e:
            logger.error(f"最終回答生成中にエラーが発生しました: {e}")
            return {
                "answer": f"回答生成エラー: {str(e)}",
                "derivations": []
            }
    
    def generate_answer_llm_only(self, question: str) -> Dict[str, Any]:
        """
        LLMのみを使用して回答を生成
        
        Args:
            question: 質問
            
        Returns:
            {"answer": 回答, "derivations": 根拠となるトリプルのリスト} の形式の辞書
        """
        try:
            # プロンプト作成
            prompt = get_llm_only_template(question)
            
            # LLM呼び出し
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            response_text = response.choices[0].message.content
            
            # レスポンスを解析
            return self._parse_llm_only_response(response_text)
            
        except Exception as e:
            logger.error(f"LLMのみでの回答生成中にエラーが発生しました: {e}")
            return {
                "answer": f"回答生成エラー: {str(e)}",
                "derivations": []
            }
    
    def _parse_llm_only_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLMのみからの回答を解析
        
        Args:
            response_text: LLMからのレスポンステキスト
            
        Returns:
            {"answer": 回答, "derivations": 根拠となるトリプルのリスト} の形式の辞書
        """
        # "=>" で分割
        parts = response_text.strip().split("=>")
        
        if len(parts) < 2:
            # フォーマットが不正な場合
            return {
                "answer": response_text.strip(),
                "derivations": []
            }
            
        derivation_part = parts[0].strip()
        answer_part = parts[1].strip() if len(parts) > 1 else ""
        
        # 複数のトリプルがある場合は ";" で区切られている
        derivations = [d.strip() for d in derivation_part.split(";")]
        
        return {
            "answer": answer_part,
            "derivations": derivations
        }
    
    def generate_staged_rag_answer(self, 
                                   question: str, 
                                   search_results: List[Any]) -> Dict[str, Any]:
        """
        段階的RAGを使用して回答を生成
        
        Args:
            question: 質問
            search_results: 検索結果のリスト
            
        Returns:
            {"answer": 回答, "derivations": 根拠となるトリプルのリスト, "source": 回答ソース} の形式の辞書
        """
        try:
            # 検索結果を文字列に変換
            search_results_text = "\n".join([str(result) for result in search_results])
            
            # プロンプト作成
            prompt = get_staged_rag_template(question, search_results_text)
            
            # LLM呼び出し
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            response_text = response.choices[0].message.content
            
            # JSON形式を解析
            result = self._parse_staged_rag_response(response_text)
            
            return result
            
        except Exception as e:
            logger.error(f"段階的RAG回答生成中にエラーが発生しました: {e}")
            return {
                "answer": f"回答生成エラー: {str(e)}",
                "derivations": [],
                "source": "error"
            }
    
    def _parse_staged_rag_response(self, response_text: str) -> Dict[str, Any]:
        """
        段階的RAGからの応答を解析
        
        Args:
            response_text: LLMからのレスポンステキスト
            
        Returns:
            {"answer": 回答, "derivations": 根拠となるトリプルのリスト, "source": 回答ソース} の形式の辞書
        """
        try:
            # JSON部分を抽出
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                data = json.loads(json_text)
                
                # データを取得
                own_answer = data.get("own_answer", "")
                searched_answer = data.get("searched_answer", "")
                pred_answer = data.get("pred_answer", "")
                derivation = data.get("derivation", "")
                
                # 回答ソースを判定
                source = "knowledge" if searched_answer == "NOTFOUND" else "rag"
                
                # 導出を複数のトリプルに分割
                derivation_list = [d.strip() for d in derivation.split(";")] if derivation else []
                
                return {
                    "answer": pred_answer,
                    "derivations": derivation_list,
                    "source": source,
                    "own_answer": own_answer,
                    "searched_answer": searched_answer
                }
            else:
                # JSON形式でない場合は直接テキストを返す
                return {
                    "answer": response_text.strip(),
                    "derivations": [],
                    "source": "unknown"
                }
                
        except json.JSONDecodeError:
            # JSON解析に失敗した場合
            logger.error(f"JSON解析エラー: {response_text}")
            return {
                "answer": response_text.strip(),
                "derivations": [],
                "source": "parsing_error"
            }