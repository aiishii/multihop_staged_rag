"""
Wikipedia および Wikidata のタイトル検索のための共通ユーティリティ
"""

import pandas as pd
from typing import List, Set, Tuple
import re

class NumberVariantGenerator:
    """数字表記のバリエーションを生成するクラス（漢数字変換など）"""
    
    # 漢数字変換用の定数
    KANJI_NUMBERS = {
        1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
        6: '六', 7: '七', 8: '八', 9: '九', 10: '十',
        100: '百', 1000: '千', 10000: '万'
    }
    
    # 数字を含む一般的な接尾辞パターン
    SUFFIX_PATTERNS = [
        '代目', '代', '世', '世代', '番目', '号', 'つ目', 'カ月',
        '歳', '才', '年', '月', '日', '時', '分', '秒', '円'
    ]
    
    @classmethod
    def number_to_kanji(cls, number: int) -> str:
        """アラビア数字を漢数字に変換"""
        if number == 0:
            return '零'
        
        result = ''
        units = [10000, 1000, 100, 10, 1]
        
        for unit in units:
            if number >= unit:
                digit = number // unit
                if digit > 1 or unit == 1:
                    result += cls.KANJI_NUMBERS[digit]
                if unit != 1:
                    result += cls.KANJI_NUMBERS[unit]
                number %= unit
        
        return result
    
    @classmethod
    def generate_variants(cls, text: str) -> Set[str]:
        """数字を含む文字列のバリエーションを生成"""
        variants = {text}
        
        # 数字+接尾辞のパターンを検出
        for suffix in cls.SUFFIX_PATTERNS:
            pattern = f'(\\d+)({suffix})'
            matches = re.finditer(pattern, text)
            
            for match in matches:
                num = int(match.group(1))
                if 1 <= num <= 99:  # 実用的な範囲に制限
                    # 漢数字バージョンを生成
                    kanji_num = cls.number_to_kanji(num)
                    new_text = text[:match.start()] + kanji_num + match.group(2) + text[match.end():]
                    variants.add(new_text)
        
        return variants

class TrieNode:
    """Trie木のノードを表すクラス"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None

class Trie:
    """文字列の高速検索のためのTrie木"""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        """単語をTrie木に挿入"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.value = word
    
    def find_longest_match(self, query: str) -> Tuple[str, bool]:
        """
        クエリ文字列に対して最長一致する文字列を探す
        
        Returns:
            (最長一致した文字列, 完全一致したかどうか)
            一致する文字列がない場合は ("", False) を返す
        """
        node = self.root
        last_match = ""
        
        # クエリの各文字について検索
        for i, char in enumerate(query):
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_end:
                last_match = node.value
        
        if not last_match:
            return "", False
            
        return last_match, last_match == query
    
class TitleSearcher:
    """Wikipediaやその他のデータソースのタイトル検索クラス"""
    
    def __init__(self, qualifier_words_file='data/wikipedia_disambig_frequency.tsv'):
        """
        初期化
        
        Args:
            qualifier_words_file: 曖昧性解消語のファイルパス
        """
        try:
            words_df = pd.read_table(qualifier_words_file)
            words_df['disambiguation'] = words_df['disambiguation'].astype(str) 
            qualifier_words_expanded = self._expand_word_list(list(words_df['disambiguation']))
            self.qualifier_matcher = self._build_matcher(qualifier_words_expanded)
        except FileNotFoundError:
            print(f"警告: {qualifier_words_file} が見つかりません。曖昧性解消機能は無効です。")
            self.qualifier_matcher = None
    
    def _expand_word_list(self, original_list: List[str]) -> List[str]:
        """単語リストを拡張して表記ゆれバリエーションを追加"""
        generator = NumberVariantGenerator()
        expanded_set = set(original_list)
        
        for word in original_list:
            variants = generator.generate_variants(word)
            expanded_set.update(variants)
        
        return sorted(list(expanded_set))
    
    def _build_matcher(self, word_list: List[str]) -> Trie:
        """Trie木を構築"""
        trie = Trie()
        for word in word_list:
            trie.insert(word)
        return trie
    
    def get_qualifier(self, title: str, debug: bool = False) -> Tuple[str, str]:
        """
        タイトルから限定子を抽出し、基本タイトルと分離する
        
        Args:
            title: 検索するタイトル
            debug: デバッグ出力の有効化フラグ
            
        Returns:
            (限定子, 基本タイトル)
        """
        if self.qualifier_matcher is None:
            return "", title
            
        match, is_exact = self.qualifier_matcher.find_longest_match(title)
        
        if debug:
            print(f"Query: {title}")
            print(f"Longest match: {match}")
            print(f"Is exact match: {is_exact}")
            
        return match, title[len(match):] if match else title

    def remove_trailing_parentheses(self, title: str) -> str:
        """タイトルの末尾の括弧を削除"""
        pattern = r'\s*[\(（].+?[\)）]\s*$'
        return re.sub(pattern, '', title)
        
    def contains_kanji_numbers(self, text: str) -> bool:
        """テキストに漢数字が含まれているか確認"""
        numbers = r'[一二三四五六七八九十百千万億]'
        return bool(re.search(numbers, text))

    def convert_kanji_to_number(self, text: str) -> str:
        """漢数字をアラビア数字に変換"""
        kanji_to_number = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
        }
        for kanji, number in kanji_to_number.items():
            if kanji in text:
                return text.replace(kanji, number)
        return text