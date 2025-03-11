"""
マルチホップ質問を分解するためのモジュール
"""

import re
from typing import List, Dict, Tuple, Optional
import openai
from loguru import logger

from ..prompts.templates import SPLIT_QUESTION_TEMPLATE


class QuerySplitter:
    """
    マルチホップ質問を単一の回答ステップに分解するクラス
    """
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        """
        初期化
        
        Args:
            model_name: 使用するLLMのモデル名
        """
        self.model_name = model_name
        self.client = openai.OpenAI()
        self.examples = SPLIT_QUESTION_TEMPLATE["examples3"]
        
    def split_question(self, question: str) -> List[Dict[str, str]]:
        """
        質問を段階的なステップに分解
        
        Args:
            question: 元の質問
            
        Returns:
            分解された質問のリスト。各要素は以下の形式:
            {
                "question": 分解された質問,
                "subject": 検索する主語,
                "predicate": 検索する述語
            }
        """
        try:
            # LLMを使って質問を分解
            prompt = SPLIT_QUESTION_TEMPLATE["split_q3"].format(self.examples, question)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            # 応答から分解された質問を抽出
            response_text = response.choices[0].message.content
            return self._parse_split_response(response_text)
            
        except Exception as e:
            logger.error(f"質問分解中にエラーが発生しました: {e}")
            # エラーが発生した場合は元の質問をそのまま返す
            return [{"question": question, "subject": "", "predicate": ""}]
    
    def _parse_split_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        LLMの応答を解析して分解された質問のリストを生成
        
        Args:
            response_text: LLMからのレスポンステキスト
            
        Returns:
            分解された質問のリスト
        """
        # タブで区切られた質問を分割
        parts = response_text.strip().split("\t")
        
        result = []
        for part in parts:
            # 質問と主語・述語のパターンを抽出
            match = re.search(r'(.*?)(?:\[(.*?),(.*?)\])?$', part.strip())
            if match:
                question = match.group(1).strip()
                subject = match.group(2).strip() if match.group(2) else ""
                predicate = match.group(3).strip() if match.group(3) else ""
                
                result.append({
                    "question": question,
                    "subject": subject,
                    "predicate": predicate
                })
            else:
                # パターンに一致しない場合はそのままの質問を追加
                result.append({
                    "question": part.strip(),
                    "subject": "",
                    "predicate": ""
                })
                
        return result
        
    def get_search_terms(self, query_parts: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """
        分解された質問から検索用の主語と述語のペアを抽出
        
        Args:
            query_parts: 分解された質問のリスト
            
        Returns:
            [(主語, 述語), ...] の形式の検索用語のリスト
        """
        search_terms = []
        
        for part in query_parts:
            subject = part.get("subject", "")
            predicate = part.get("predicate", "")
            
            # 主語が "Answer1" などの場合は前の回答を使用するため、現時点ではスキップ
            if subject and not subject.startswith("Answer"):
                search_terms.append((subject, predicate))
                
        return search_terms