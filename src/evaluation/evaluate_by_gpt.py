"""
GPTを用いた評価モジュール
"""

import argparse
import logging
import datetime
import re
import pandas as pd
import json
from typing import List, Dict, Tuple, Optional, Any, Union

import openai
from loguru import logger

from ..prompts.templates import SCORING_BY_GPT


class GptEvaluator:
    """
    GPTを使用して予測結果を評価するクラス
    """
    
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        """
        初期化
        
        Args:
            model_name: 使用するLLMのモデル名
        """
        self.model_name = model_name
        self.client = openai.OpenAI()
        
        # 評価用プロンプトテンプレート
        self.system_message_derivation = SCORING_BY_GPT["inst-5-derivation"].format(
            SCORING_BY_GPT["examples-5_derivation"]
        )
        self.system_message_answer = SCORING_BY_GPT["inst-4-answer"].format(
            SCORING_BY_GPT["examples-4_answer"]
        )
        
    def evaluate_prediction(self, 
                           question: str, 
                           ground_truth_derivation: str, 
                           ground_truth_answer: str, 
                           prediction_derivation: str, 
                           prediction_answer: str) -> Dict[str, Any]:
        """
        予測結果を評価
        
        Args:
            question: 質問
            ground_truth_derivation: 正解の根拠
            ground_truth_answer: 正解の回答
            prediction_derivation: 予測された根拠
            prediction_answer: 予測された回答
            
        Returns:
            評価結果を含む辞書
        """
        # 回答の評価
        answer_score, answer_explanation = self.evaluate_answer(
            ground_truth_answer, 
            prediction_answer
        )
        
        # 根拠の評価
        derivation_score, derivation_score_list, derivation_explanation = self.evaluate_derivation(
            question,
            ground_truth_derivation,
            ground_truth_answer,
            prediction_derivation
        )
        
        # 評価結果をまとめる
        return {
            "score_answer": answer_score,
            "answer_explanation": answer_explanation,
            "score_derivation": derivation_score,
            "score_derivation_list": derivation_score_list,
            "derivation_explanation": derivation_explanation
        }
    
    def evaluate_answer(self, 
                       ground_truth_answer: str, 
                       prediction_answer: str) -> Tuple[float, str]:
        """
        回答を評価
        
        Args:
            ground_truth_answer: 正解の回答
            prediction_answer: 予測された回答
            
        Returns:
            (スコア, 説明) の形式のタプル
        """
        try:
            # 入力プロンプトの作成
            input_prompt = SCORING_BY_GPT["input_template_answer"].format(
                ground_truth_answer,
                prediction_answer
            )
            
            # GPT呼び出し
            messages = [
                {"role": "system", "content": self.system_message_answer},
                {"role": "user", "content": input_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            # 応答の解析
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            
            score = result.get("score-answer", 0)
            explanation = result.get("explanation", "説明なし")
            
            return float(score), explanation
            
        except Exception as e:
            logger.error(f"回答評価中にエラーが発生しました: {e}")
            return 0.0, f"評価エラー: {str(e)}"
    
    def evaluate_derivation(self, 
                           question: str, 
                           ground_truth_derivation: str, 
                           ground_truth_answer: str, 
                           prediction_derivation: str) -> Tuple[float, List[float], str]:
        """
        根拠を評価
        
        Args:
            question: 質問
            ground_truth_derivation: 正解の根拠
            ground_truth_answer: 正解の回答
            prediction_derivation: 予測された根拠
            
        Returns:
            (総合スコア, 各トリプルのスコアリスト, 説明) の形式のタプル
        """
        try:
            # 入力プロンプトの作成
            input_prompt = SCORING_BY_GPT["input_template_derivation"].format(
                question,
                ground_truth_derivation,
                ground_truth_answer,
                prediction_derivation
            )
            
            # GPT呼び出し
            messages = [
                {"role": "system", "content": self.system_message_derivation},
                {"role": "user", "content": input_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            # 応答の解析
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            
            score = result.get("score-derivations", 0)
            score_list = result.get("score-derivation-list", [])
            explanation = result.get("explanation", "説明なし")
            
            return float(score), score_list, explanation
            
        except Exception as e:
            logger.error(f"根拠評価中にエラーが発生しました: {e}")
            return 0.0, [], f"評価エラー: {str(e)}"


def evaluate_predictions(prediction_tsv: str, 
                        model_name: str = "gpt-4o-2024-08-06", 
                        evaluate_type: str = "ALL") -> Dict[str, Any]:
    """
    予測TSVファイルを評価
    
    Args:
        prediction_tsv: 予測結果TSVファイルのパス
        model_name: 使用するLLMのモデル名
        evaluate_type: 評価の種類（"ALL", "ANS", "DRV"）
        
    Returns:
        評価結果を含む辞書
    """
    # ファイル名から出力ファイル名を生成
    label = prediction_tsv.replace('.tsv', '')
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    out_file = label + '_score-gpt_' + now_str + '.tsv' 
    log_file = label + '_score-gpt-all' + now_str + '.tsv'
    json_file = label + '_score-gpt_' + now_str + '.json' 

    # 評価器の初期化
    evaluator = GptEvaluator(model_name)

    # 開発セットとテスト予測の読み込み
    dev_file = "data/corpus_ver1.1/dev_ver1.1_fix20241017.json"
    df_dev = pd.read_json(dev_file)

    df_pred = pd.read_table(prediction_tsv)
    df_pred = df_pred.fillna("")

    # 評価用カウンタの初期化
    n_correct = 0
    n_correct_derivations = 0
    n_correct_answer = 0
    n_miss = 0
    n_miss_derivations = 0
    n_miss_answer = 0
    n_spurious_correct = 0
    n_logic_error = 0

    # 結果を格納する辞書
    result_dic = {}
    
    # 出力ファイルを開く
    with open(out_file, mode='w') as out_f:
        # ヘッダーを書き込み
        out_f.write('\t'.join([
            "qid", "question", "gold_answer", "gold_derivations", 
            "predicted_answer", "predicted_derivations", 
            "score_answer", "score_derivations", "score_derivation_list", 
            "explanation", "miss_answer", "miss_derivations"
        ])+'\n')
        
        # 各データに対して評価を実行
        for (idx1, r_dev), (idx2, r_pred) in zip(df_dev.iterrows(), df_pred.iterrows()):
            # 初期値設定
            miss_answer = 0
            miss_derivations = 0
            score_derivations = 0
            score_answer = 0
            explanation = ""
            
            # デバッグ出力
            print('----------\n', r_dev.question)
            
            # 予測された根拠の解析
            pred_derivation_str_list = r_pred.predicted_derivations.split(';')
            pred_derivations_len = len(pred_derivation_str_list)
            score_derivation_list = [0] * pred_derivations_len
            is_miss_derivations_list = [None] * pred_derivations_len

            # 正解の根拠をフォーマット
            deriv_list = []
            for deriv in r_dev.derivations:
                deriv_list.append("（"+'，'.join([deriv[0], deriv[1], '、'.join(deriv[2])])+"）")
            gold_derivations = ';'.join(deriv_list)
            
            # 完全一致の場合
            if r_pred.predicted_derivations == gold_derivations and r_pred.predicted_answer == r_dev.answer:
                # 完全一致
                score_derivations = 1
                score_answer = 1
                score_derivation_list = [1] * pred_derivations_len
                explanation = "EM"
            else:
                # 根拠の評価
                if evaluate_type in ["ALL", "DRV"]:
                    # NOTFOUNDチェック
                    for i, pred_deriv_str in enumerate(pred_derivation_str_list):
                        if is_miss(pred_deriv_str):
                            miss_derivations = 1
                            is_miss_derivations_list[i] = True
                        else:
                            is_miss_derivations_list[i] = False
                    if miss_derivations == 1:
                        n_miss_derivations += 1

                    # 根拠の完全一致チェック
                    if r_pred.predicted_derivations == gold_derivations:
                        score_derivations = 1
                        explanation = "EM(derivation)"
                        score_derivation_list = [1] * pred_derivations_len
                    else:
                        # GPTによる根拠評価
                        eval_result = evaluator.evaluate_derivation(
                            r_dev.question,
                            gold_derivations,
                            r_dev.answer,
                            r_pred.predicted_derivations
                        )
                        score_derivations = eval_result[0]
                        score_derivation_list = eval_result[1]
                        explanation = eval_result[2]
                
                # 回答の評価
                if evaluate_type in ["ALL", "ANS"]:
                    if 'score_answer' in r_pred:
                        score_answer = r_pred['score_answer']
                    else:
                        # NOTFOUNDチェック
                        if is_miss(r_pred.predicted_answer): 
                            miss_answer = 1
                            score_answer = 0
                            n_miss_answer += 1
                        # 完全一致チェック
                        elif normalize_answer(r_pred.predicted_answer) == normalize_answer(r_dev.answer):
                            score_answer = 1
                        # YES/NOの場合
                        elif (normalize_answer(r_pred.predicted_answer) in ["yes", "no"] and 
                             normalize_answer(r_dev.answer) in ["yes", "no"]):
                            score_answer = 0
                        else:
                            # GPTによる回答評価
                            eval_result = evaluator.evaluate_answer(
                                r_dev.answer,
                                r_pred.predicted_answer
                            )
                            score_answer = eval_result[0]

            # 総合判定（ALL評価の場合）
            if evaluate_type == "ALL":
                if miss_answer == 1 and miss_derivations == 1:
                    n_miss += 1
                    explanation = "MISS"
                if score_derivations == 1 and score_answer == 1:
                    n_correct += 1
                    n_correct_answer += 1
                    n_correct_derivations += 1
                if score_derivations == 0 and score_answer == 1:
                    n_spurious_correct += 1
                    n_correct_answer += 1
                if score_derivations == 1 and score_answer == 0:
                    n_correct_derivations += 1
                    n_logic_error += 1
                    
            # 結果の出力
            output_list = [
                r_dev.qid, r_dev.question, r_dev.answer, gold_derivations, 
                r_pred.predicted_answer, r_pred.predicted_derivations, 
                str(score_answer), str(score_derivations), 
                '/'.join([str(_s) for _s in score_derivation_list]), 
                explanation, str(miss_answer), str(miss_derivations)
            ]
            
            # 結果辞書に格納
            result_dic[r_dev.qid] = {
                "qid": r_dev.qid,
                "question": r_dev.question,
                "gold_answer": r_dev.answer,
                "gold_derivations": r_dev.derivations,
                "predicted_answer": r_pred.predicted_answer,
                "predicted_derivations": r_pred.predicted_derivations,
                "score_answer": score_answer,
                "score_derivations": score_derivations,
                "score_derivation_list": score_derivation_list,
                "explanation": explanation,
                "miss_answer": miss_answer,
                "miss_derivations": miss_derivations,
            }
            
            # ファイルに出力
            out_f.write('\t'.join(output_list)+'\n')
            
            # デバッグ出力
            print('\t'.join([
                str(score_answer), str(score_derivations), 
                '/'.join([str(_s) for _s in score_derivation_list]), 
                explanation, str(miss_answer), str(miss_derivations)
            ]))
            print('----------')

    # JSON形式で結果を保存
    with open(json_file, mode='w') as out_f:
        out_f.write(json.dumps(result_dic, ensure_ascii=False, indent=2))

    # 全体評価のまとめ（ALL評価の場合）
    if evaluate_type == "ALL":
        n = len(df_pred)
        results = {
            "score": (2 * n_correct + n_miss) / n - 1,
            "accuracy": n_correct / n,
            "hallucination": (n - n_correct - n_miss) / n,
            "missing": n_miss / n,
            "spurious_correct": n_spurious_correct / n,
            "n_miss": n_miss,
            "n_miss_answer": n_miss_answer,
            "n_miss_derivations": n_miss_derivations,
            "n_correct": n_correct,
            "n_correct_answer": n_correct_answer,
            "n_correct_derivations": n_correct_derivations,
            "n_logic_error": n_logic_error,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
        }
    
        # 評価結果をTSVファイルに保存
        with open(log_file, mode='w') as out_f:
            key_list = list(results.keys())
            out_f.write('\t'.join(key_list)+'\n')
            out_f.write('\t'.join([str(results[k]) for k in key_list])+'\n')

        return results, json_file
    else:
        return {}, json_file


def is_miss(text: str, 
           miss_words: List[str] = ["NOTFOUND", "不明", "判断でき", "情報不足", "見つかり", 
                               "わからない", "わかりません", "公表されていない", "情報非公開", 
                               "公に知られていない", "情報なし", "未確認"]) -> bool:
    """
    テキストが「見つからない」を意味するかチェック
    
    Args:
        text: チェックするテキスト
        miss_words: 「見つからない」を示す単語リスト
        
    Returns:
        「見つからない」を意味する場合はTrue
    """
    if len(str(text).strip()) == 0 or str(text) == 'nan':
        return True
    for w in miss_words:
        if w in str(text): 
            return True
    return False


def white_space_fix(text: str) -> str:
    """空白の正規化"""
    return ' '.join(text.split())


def remove_brackets(text: str) -> str:
    """括弧の除去"""
    text = re.sub(r'\s*[\(（].+?[\)）]\s*', '', text)
    return re.sub(r'[『』「」]', '', text)


def normalize_answer(s: str) -> str:
    """回答の正規化"""
    if s == 'はい': 
        s = 'YES'
    elif s == 'いいえ': 
        s = 'NO'

    return white_space_fix(remove_brackets(s.lower()))


def main():
    """メイン関数"""
    logging.basicConfig(format='%(asctime)s- %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-pred_tsv', '--prediction_tsv', required=True,
        help="Model prediction tsv file name.")
    parser.add_argument(
        '-model', '--model_name', default="gpt-4o-2024-08-06",
        help="model name")
    parser.add_argument(
        '-ea', '--eval_answer', action="store_true",
        help="evaluate answer")    
    parser.add_argument(
        '-ed', '--eval_deriv', action="store_true",
        help="evaluate derivation")    
    
    args = parser.parse_args()
    
    # 評価タイプの設定
    if args.eval_answer and args.eval_deriv:
        evaluate_type = "ALL"
    elif args.eval_answer:
        evaluate_type = "ANS"
    elif args.eval_deriv:
        evaluate_type = "DRV"
    else:
        print('Specify either --eval_answer, --eval_deriv, or both options.')
        exit()

    # 評価の実行
    evaluation_results, json_file = evaluate_predictions(
        args.prediction_tsv, args.model_name, evaluate_type=evaluate_type
    )
    print(json_file)


if __name__ == "__main__":
    main()