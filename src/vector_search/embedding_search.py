"""
ベクトル検索のための共通モジュール
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple, Union


class EmbeddingSearch:
    """
    ベクトル検索のための基本クラス
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", max_seq_length: int = 512):
        """
        初期化
        
        Args:
            model_name: 埋め込みモデル名
            max_seq_length: 最大シーケンス長
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        
    def to_emb(self, text: Union[str, List[str]], prefix: str = "query: ") -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            text: 入力テキストまたはテキストのリスト
            prefix: プレフィックス文字列
            
        Returns:
            埋め込みベクトル
        """
        if isinstance(text, str):
            text = [prefix + text]
        else:
            text = [prefix + t for t in text]
            
        return self.model.encode(text, normalize_embeddings=True)
        
    def search(self, 
               index: faiss.Index, 
               query: str, 
               dataset: Any, 
               top_k: int = 5, 
               prefix: str = "query: ",
               template: Optional[str] = None) -> List[Tuple[float, str, str]]:
        """
        ベクトル検索の実行
        
        Args:
            index: FAISSインデックス
            query: 検索クエリ
            dataset: 検索対象のデータセット
            top_k: 上位k件の結果を返す
            prefix: プレフィックス文字列
            template: 結果フォーマットのテンプレート文字列
            
        Returns:
            [(スコア, ID, テキスト), ...] の形式の検索結果リスト
        """
        # クエリの埋め込み取得
        emb = self.to_emb(query, prefix=prefix)
        
        # 検索実行
        scores, indexes = index.search(emb, top_k)
        
        result = []
        for idx, (id, score) in enumerate(zip(indexes[0], scores[0])):
            if id < 0 or id >= len(dataset):
                continue
                
            data = dataset[int(id)]
            result.append((score, f"emb-{id}", self._format_result(data, template)))
            
        return result
    
    def _format_result(self, data: Dict[str, Any], template: Optional[str] = None) -> str:
        """
        データをフォーマットする
        
        Args:
            data: フォーマットするデータ
            template: テンプレート文字列
            
        Returns:
            フォーマットされた文字列
        """
        if template is None:
            # デフォルトのフォーマット
            if 'text' in data:
                return data['text']
            elif 'title' in data and 'value' in data:
                return f"{data['title']}の{data.get('attribute', '')}は{data['value']}。"
            else:
                return str(data)
        else:
            # テンプレートによるフォーマット
            return template.format(**data)