"""
Wikidataのベクトル検索
"""

from datasets import load_dataset
import faiss
from typing import List, Tuple, Optional

from .embedding_search import EmbeddingSearch


class WikidataFaissSearch(EmbeddingSearch):
    """
    WikidataのFAISSベクトル検索クラス
    """
    
    def __init__(self, 
                 model_name: str = "intfloat/multilingual-e5-small",
                 index_path: str = "data/indices/wikidata-20210823-all/multilingual-e5-small-passage/index_IVF1024_PQ32.faiss",
                 dataset_path: str = "data/wikidata-20210823-all.tsv"):
        """
        初期化
        
        Args:
            model_name: 埋め込みモデル名
            index_path: FAISSインデックスファイルのパス
            dataset_path: データセットファイルのパス
        """
        super().__init__(model_name)
        
        # FAISSインデックスのロード
        self.index = faiss.read_index(index_path)
        
        # データセットのロード
        try:
            self.dataset = load_dataset(
                "csv", 
                data_files=[dataset_path], 
                on_bad_lines="skip", 
                sep=None, 
                engine='python', 
                encoding_errors='ignore', 
                quoting=3
            )["train"]
            
            # None値のフィルタリング
            self.dataset = self.dataset.filter(self._filter_none)
        except Exception as e:
            print(f"データセットのロード中にエラーが発生しました: {e}")
            raise
        
        # 結果フォーマット用テンプレート
        self.template = "{title}の{attr}は{value}。"
        
    def _filter_none(self, example):
        """Noneの値を持つ例をフィルタリング"""
        return all(value is not None for value in example.values())
        
    def search(self, text: str, top_k: int = 5) -> List[Tuple[float, str, str]]:
        """
        Wikidataに対するベクトル検索
        
        Args:
            text: 検索テキスト
            top_k: 上位k件の結果を返す
            
        Returns:
            [(スコア, ID, フォーマット済みテキスト), ...] の形式の検索結果リスト
        """
        # 埋め込み取得と検索実行
        emb = self.to_emb(text)
        scores, indexes = self.index.search(emb, top_k)
        
        result = []
        for idx, (id, score) in enumerate(zip(indexes[0], scores[0])):
            if id < 0 or id >= len(self.dataset):
                continue
                
            data = self.dataset[int(id)]
            
            # 値の引用符処理
            value = data["value"]
            if isinstance(value, str) and len(value) > 1 and value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
                
            # 結果フォーマット
            formatted_text = self.template.format(
                title=data["title"], 
                attr=data["attribute"], 
                value=value
            )
            
            result.append((score, f"embW-{id}", formatted_text))
            
        return result


# シングルトンインスタンス
_wikidata_search = None

def get_wikidata_search(
    model_name: str = "intfloat/multilingual-e5-small",
    index_path: Optional[str] = None,
    dataset_path: Optional[str] = None
) -> WikidataFaissSearch:
    """
    WikidataFaissSearchのシングルトンインスタンスを取得
    
    Args:
        model_name: 埋め込みモデル名
        index_path: FAISSインデックスファイルのパス（Noneの場合はデフォルトを使用）
        dataset_path: データセットファイルのパス（Noneの場合はデフォルトを使用）
        
    Returns:
        WikidataFaissSearchインスタンス
    """
    global _wikidata_search
    if _wikidata_search is None:
        _wikidata_search = WikidataFaissSearch(
            model_name=model_name,
            index_path=index_path if index_path else "data/indices/wikidata-20210823-all/multilingual-e5-small-passage/index_IVF1024_PQ32.faiss",
            dataset_path=dataset_path if dataset_path else "data/wikidata-20210823-all.tsv"
        )
    return _wikidata_search

def faiss_search_wikidata(text: str, top_k: int = 5) -> List[Tuple[float, str, str]]:
    """
    Wikidataに対するベクトル検索の簡易関数
    
    Args:
        text: 検索テキスト
        top_k: 上位k件の結果を返す
        
    Returns:
        [(スコア, ID, フォーマット済みテキスト), ...] の形式の検索結果リスト
    """
    return get_wikidata_search().search(text, top_k)