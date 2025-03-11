"""
トリプルデータベースの検索処理モジュール
"""

import sqlite3
import pandas as pd
import re
from string import Formatter
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
from typing import List, Dict, Optional, Tuple, Union

# 相対インポート
from .title_searcher import TitleSearcher

class ProcessDB:
    """トリプルデータベース検索のためのクラス"""
    
    def __init__(self, db_file: str = 'data/triple_database.db'):
        """
        初期化
        
        Args:
            db_file: SQLiteデータベースファイルのパス
        """
        self.conn = sqlite3.connect(db_file)
        self.title_searcher = TitleSearcher()
        
    def wiki_title_search(self, title: str, table_name: str = "shinra", debug: bool = False) -> pd.DataFrame:
        """
        Wikipediaタイトルを検索する包括的な検索メソッド
        
        Args:
            title: 検索するタイトル
            table_name: 検索対象のテーブル名（"shinra"または"wikidata"）
            debug: デバッグ出力の有効化フラグ
            
        Returns:
            検索結果のDataFrame
        """
        # 完全一致
        if debug: 
            print('comprehensive_search: try 完全一致')
        df = self.select_by_title(title, table_name=table_name, nocase=False, like=False)
        if len(df) > 0: 
            return df
            
        # 大文字小文字許容
        if title != title.lower():
            if debug:
                print('comprehensive_search: try 大文字小文字許容')
            df = self.select_by_title(title, table_name=table_name, nocase=True, like=False)
            if len(df) > 0: 
                return df
                
        # カッコがなかったら後ろにカッコがつくものを全文検索
        if not '(' in title and not '（' in title:
            if debug:
                print('comprehensive_search: try カッコがつくやつのみ全文検索')
            df = self.select_by_title_fts(title, with_paren=True)
            if len(df) > 0: 
                return df
                
        # 全文検索
        if debug:
            print('comprehensive_search: try 全文検索')
        df = self.select_by_title_fts(title, with_paren=False)
        if len(df) > 0: 
            return df
            
        # カッコがあったらカッコを外して全文検索
        if '(' in title or '（' in title:
            if debug:
                print('comprehensive_search: try カッコを外して全文検索')
            df = self.select_by_title_fts(
                self.title_searcher.remove_trailing_parentheses(title), 
                with_paren=False
            )
            if len(df) > 0: 
                return df
        
        # 限定子を取得して検索
        qualifier, main_title = self.title_searcher.get_qualifier(title)
        if qualifier:
            if debug:
                print('comprehensive_search: try using wiki_qualifier', qualifier, main_title)
            df = self.select_by_title_fts(
                self.title_searcher.remove_trailing_parentheses(main_title + " " + qualifier), 
                with_paren=False
            )
            if len(df) > 0: 
                return df
            
        # 漢数字を含む限定子の変換
        if qualifier and self.title_searcher.contains_kanji_numbers(qualifier):
            qualifier = self.title_searcher.convert_kanji_to_number(qualifier)
            if debug:
                print('comprehensive_search: try using wiki_qualifier convert kanji number', qualifier, main_title)
            df = self.select_by_title_fts(
                self.title_searcher.remove_trailing_parentheses(main_title + " " + qualifier), 
                with_paren=False
            )
            if len(df) > 0: 
                return df
        
        # 限定子を除外して検索
        if qualifier:
            if debug:
                print('comprehensive_search: try without wiki_qualifier', qualifier, main_title)
            df = self.select_by_title_fts(
                self.title_searcher.remove_trailing_parentheses(main_title), 
                with_paren=False
            )
            if len(df) > 0: 
                return df

        return pd.DataFrame()    

    def select_by_title(self, title: str, table_name: str = "shinra", nocase: bool = False, like: bool = False) -> pd.DataFrame:
        """
        タイトルによるデータベース検索
        
        Args:
            title: 検索するタイトル
            table_name: 検索対象のテーブル名
            nocase: 大文字小文字を区別しない検索を行うかどうか
            like: LIKE演算子を使用した部分一致検索を行うかどうか
            
        Returns:
            検索結果のDataFrame
        """
        if nocase and like:
            query = f"SELECT * FROM {table_name} WHERE title LIKE ? || '%' COLLATE NOCASE"
            df = pd.read_sql_query(query, self.conn, params=(self.escape_fts_query(title),))
        elif nocase:
            query = f"SELECT * FROM {table_name} WHERE title COLLATE NOCASE = ?"
            df = pd.read_sql_query(query, self.conn, params=(self.escape_fts_query(title),))
        else:
            query = f"SELECT * FROM {table_name} WHERE title=\"{self.escape_fts_query(title)}\""
            df = pd.read_sql_query(query, self.conn)
        return df 

    def escape_fts_query(self, query: str) -> str:
        """
        FTS (Full Text Search) のクエリをエスケープする
        
        Args:
            query: エスケープする検索クエリ
            
        Returns:
            エスケープされたクエリ
        """
        # エスケープが必要な特殊文字
        special_chars = '"()[]{}*+-!?^:&|/\'' 
        # 特殊文字の前にバックスラッシュを追加
        for char in special_chars:
            query = query.replace(char, f'\\{char}')
        return query
        
    def select_by_title_fts(self, title: str, with_paren: bool = False) -> pd.DataFrame:
        """
        FTSを使用したタイトル検索
        
        Args:
            title: 検索するタイトル
            with_paren: カッコを含むタイトルのみを検索するかどうか
            
        Returns:
            検索結果のDataFrame
        """
        fts_table_name = "triples_fts"

        try:
            if with_paren and not '(' in title:  # LIFE (小沢健二のアルバム) に対応
                query = f"""
                    SELECT * 
                    FROM {fts_table_name} 
                    WHERE title MATCH ?
                    AND title LIKE ? || " (%"
                """
                df = pd.read_sql_query(
                    query, 
                    self.conn, 
                    params=('"'+self.escape_fts_query(title)+'"', '"'+title + " (")
                )
            else:
                query = f"""
                SELECT * FROM {fts_table_name} WHERE title MATCH ?
                """
                df = pd.read_sql_query(
                    query, 
                    self.conn, 
                    params=('"'+self.escape_fts_query(title)+'"',)
                )
        except Exception as e:
            print(e)
            print('title', title, self.escape_fts_query(title))
            print('query', query)
            raise(e)
            
        return df

    def get_titles_by_title_fts(self, title: str, with_paren: bool = False) -> pd.DataFrame:
        """
        FTSを使用したタイトル一覧の取得
        
        Args:
            title: 検索するタイトル
            with_paren: カッコを含むタイトルのみを検索するかどうか
            
        Returns:
            タイトル一覧のDataFrame
        """
        fts_table_name = "triples_fts"
        
        try:
            if with_paren and not '(' in title:  # カッコがくるものだけ検索
                query = f"""
                    SELECT title 
                    FROM {fts_table_name} 
                    WHERE title MATCH ? 
                    AND title LIKE ? || '%'
                """
                df = pd.read_sql_query(query, self.conn, params=(self.escape_fts_query(title), title + " ("))
            else:
                query = f"""
                    SELECT title FROM {fts_table_name} 
                    WHERE title MATCH ?
                """
                df = pd.read_sql_query(query, self.conn, params=(self.escape_fts_query(title),))
        except Exception as e:
            print(e)
            print('title', title, self.escape_fts_query(title))
            print('query', query)
            raise(e)
            
        # 重複を削除
        df = df.drop_duplicates(subset=['title'])
        return df
    
    def create_fts5_table(self, table_name: str) -> None:
        """
        FTS5テーブルを作成する（初回のみ実行）
        
        Args:
            table_name: 対象のテーブル名
        """
        cursor = self.conn.cursor()
        
        # FTS5テーブルの作成（日本語対応）
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_fts USING fts5(
                page_id UNINDEXED,  -- 検索対象外のカラムはUNINDEXEDを指定
                title,
                attribute,
                value UNINDEXED,
                tokenize='unicode61'  -- 日本語対応のトークナイザ
            )
        """)
        
        # 既存データの移行
        cursor.execute(f"""
            INSERT INTO {table_name}_fts(page_id, title, attribute, value)
            SELECT page_id, title, attribute, value
            FROM {table_name}
        """)
        
        self.conn.commit()

        # インデックスの作成
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_title ON {table_name}(title COLLATE NOCASE)")
        self.conn.commit() 
    
    def create_index_nocase(self, table_name: str) -> None:
        """
        大文字小文字を区別しないインデックスを作成（初回のみ実行）
        
        Args:
            table_name: 対象のテーブル名
        """
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_title ON {table_name}(title COLLATE NOCASE)")
        self.conn.commit()


def format_template_vectorized(df: pd.DataFrame, template: str) -> pd.Series:
    """
    DataFrameに対してテンプレートを適用
    
    Args:
        df: 入力のDataFrame
        template: フォーマットテンプレート文字列
        
    Returns:
        フォーマットされた文字列のSeries
    """
    formatter = Formatter()
    field_names = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name is not None]

    return df.apply(
        lambda row: template.format(**{field: row[field] for field in field_names}),
        axis=1
    )


class DBEmbeddingSearch:
    """
    データベースとベクトル検索を組み合わせるクラス
    """
    
    def __init__(self, db_file: str = 'data/triple_database.db', model_name: str = ''):
        """
        初期化
        
        Args:
            db_file: データベースファイルのパス
            model_name: 埋め込みモデル名
        """
        self.process_db = ProcessDB(db_file)
        self.model_name = model_name
        if 'multilingual-e5' in self.model_name:
            self.model = SentenceTransformer(self.model_name)
        if 'japanese-reranker-cross-encoder' in self.model_name:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CrossEncoder(self.model_name, max_length=512, device=device)
            if device == "cuda":
                self.model.model.half()

    def search(self, 
               title: str, 
               attr: str, 
               table_name: Optional[str] = None, 
               table_names: List[str] = [], 
               top_k: int = 2, 
               use_template: bool = False, 
               org_q: Optional[str] = None, 
               debug: bool = False) -> pd.DataFrame:
        """
        知識ベース検索
        
        Args:
            title: 検索するタイトル
            attr: 属性
            table_name: 検索対象のテーブル名
            table_names: 複数テーブルの場合はテーブル名のリスト
            top_k: 上位k件の結果を返す
            use_template: テンプレートを使用するかどうか
            org_q: 元のクエリ
            debug: デバッグ出力の有効化フラグ
            
        Returns:
            検索結果のDataFrame
        """
        if debug: 
            print('search_kb in', title, attr, table_name, use_template)
            
        # テーブル指定
        if table_name:
            df = self.process_db.wiki_title_search(title, table_name=table_name, debug=debug)
        elif table_names:
            df = self.process_db.wiki_title_search(title, table_name=table_names[0], debug=debug)
            for i in range(1, len(table_names)):
                df = pd.concat([df, self.process_db.wiki_title_search(title, table_name=table_names[i], debug=debug)])
                
        # 結果が見つからない場合は空のDataFrameを返す
        if df.empty:
            if debug: 
                print('failed select_by_title')
            return pd.DataFrame()
            
        # 検索クエリとテンプレートの準備
        if use_template:
            query_template = "{title}の{attribute}は？"
            query = query_template.format(title=title, attribute=attr)
            template = "{title}の{attribute}は{value}です"
        else:
            query = attr
            template = None

        # 埋め込み検索の実行
        result_df = self.embedding_search(df, query, top_k=top_k, template=template, org_q=org_q)
        if debug: 
            print('search_kb result', len(result_df))
        return result_df

    def to_emb_for_e5(self, text, prefix="query: "):
        """
        E5モデル用にテキストを埋め込みベクトルに変換
        
        Args:
            text: 入力テキストまたはテキストのリスト
            prefix: プレフィックス文字列
            
        Returns:
            埋め込みベクトル
        """
        # textが文字列の場合、リストに変換
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text 
        # 各テキストにprefixを追加
        texts_with_prefix = [prefix + t for t in texts]       
        # エンコード
        return self.model.encode(texts_with_prefix, normalize_embeddings=True)

    def embedding_search(self, 
                         df: pd.DataFrame, 
                         query: str, 
                         column: str = 'attribute', 
                         top_k: int = 2, 
                         template: Optional[str] = None, 
                         org_q: Optional[str] = None) -> pd.DataFrame:
        """
        埋め込みベクトル検索
        
        Args:
            df: 検索対象のDataFrame
            query: 検索クエリ
            column: 検索対象のカラム
            top_k: 上位k件の結果を返す
            template: テンプレート文字列
            org_q: 元のクエリ
            
        Returns:
            検索結果のDataFrame（スコア付き）
        """
        # 値の引用符を除去
        df['value'] = df['value'].str.strip('"')
        
        # テンプレート適用またはカラム値の取得
        if template:
            passages = format_template_vectorized(df, template).tolist()
        else:
            passages = df[column].tolist()

        # 元のクエリが指定されていれば結合
        if org_q:
            query = org_q + query

        # モデルによる類似度計算
        if 'multilingual-e5' in self.model_name:
            embeddings = self.to_emb_for_e5(passages, prefix="passage: ")
            query_embedding = self.to_emb_for_e5(query, prefix="query: ")
            similarities = cosine_similarity(query_embedding, embeddings).flatten()
        elif 'japanese-reranker-cross-encoder' in self.model_name:
            similarities = self.model.predict([(query, passage) for passage in passages])

        # 上位k件のインデックスを取得
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        
        # 結果のDataFrameを作成
        result_df = df.iloc[top_k_indices].assign(score=similarities[top_k_indices])

        return result_df