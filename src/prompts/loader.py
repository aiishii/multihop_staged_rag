"""
プロンプトテンプレートのローダーモジュール
"""

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

# テンプレートエンジンの初期化
template_path = Path(__file__).parent / "templates"
env = Environment(
    loader=FileSystemLoader(template_path),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)

def render_template(template_name, **kwargs):
    """
    指定されたテンプレートをレンダリング
    
    Args:
        template_name: テンプレートのパス (categories/name.j2 形式)
        **kwargs: テンプレートに渡す変数
        
    Returns:
        レンダリングされたテンプレート文字列
    """
    template = env.get_template(f"{template_name}.j2")
    return template.render(**kwargs)

# よく使われるテンプレート用のヘルパー関数
def get_split_question_template(question, examples, with_search=True):
    """質問分解テンプレートをレンダリング"""
    return render_template(
        "query_decomposition/split_question", 
        question=question, 
        examples=examples,
        with_search=with_search
    )

def get_simple_rag_template(question, search_results):
    """シンプルRAGテンプレートをレンダリング"""
    return render_template(
        "retrieval/simple_rag", 
        question=question, 
        search_results=search_results
    )

def get_final_answer_template(question, derivations=None):
    """最終回答生成テンプレートをレンダリング"""
    return render_template(
        "generation/final_answer", 
        question=question,
        derivations=derivations
    )

def get_llm_only_template(question):
    """LLMのみで回答するテンプレートをレンダリング"""
    return render_template("generation/llm_only", question=question)

def get_staged_rag_template(question, search_results):
    """段階的RAGテンプレートをレンダリング"""
    return render_template(
        "generation/staged_rag", 
        question=question, 
        search_results=search_results
    )

def get_derivation_evaluation_template(question, ground_truth_derivation, ground_truth_answer, prediction_derivation):
    """導出評価テンプレートをレンダリング"""
    return render_template(
        "evaluation/derivation_evaluation", 
        question=question,
        ground_truth_derivation=ground_truth_derivation,
        ground_truth_answer=ground_truth_answer,
        prediction_derivation=prediction_derivation
    )

def get_answer_evaluation_template(ground_truth_answer, prediction_answer):
    """回答評価テンプレートをレンダリング"""
    return render_template(
        "evaluation/answer_evaluation", 
        ground_truth_answer=ground_truth_answer,
        prediction_answer=prediction_answer
    )