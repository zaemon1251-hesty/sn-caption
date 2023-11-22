from __future__ import annotations
import pandas as pd
from typing_extensions import TypedDict
import ast
from loguru import logger

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


category_name = "大分類"
subcategory_name = "小分類"

random_seed = 42
half_number = 1
model_type = "gpt-3.5-turbo-1106"


LLM_ANOTATION_CSV_PATH = (
    Config.base_dir / f"{random_seed}_{half_number}_llm_annotation.csv"
)
HUMAN_ANOTATION_CSV_PATH = (
    Config.base_dir / f"{random_seed}_{half_number}_moriy_annotation_preprocessed.csv"
)


class LlmAnnotationResult(TypedDict):
    """
    Meta information for LLM annotation evaluation
    """

    model_type: str
    random_seed: int
    half_number: int
    comment_ids: int
    category: EvalIndicator
    subcategory: EvalIndicator


class EvalIndicator(TypedDict):
    """
    Mohammad S Sorower,
    A Literature Survey on Algorithms for Multi-label Learning, 2010.

    Exact Match Ratio (EMR) = mean^{n}_{i=1} (Yi = Zi)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Precision = mean^{n}_{i=1} (Yi \bigcap Zi) / Zi
    Recall = mean^{n}_{i=1} (Yi \bigcap Zi) / Yi
    Accuracy = mean^{n}_{i=1} (Yi \bigcap Zi) / (Yi \bigcup Zi)
    """

    exact_match: float
    f1: float
    precision: float
    recall: float
    accuracy: float


class EvaluateAnnotation:
    """calculate evaluation metrics for annotation"""

    def __init__(self) -> None:
        human_df = pd.read_csv(HUMAN_ANOTATION_CSV_PATH)
        llm_df = pd.read_csv(LLM_ANOTATION_CSV_PATH)

        # 文字列として格納されている配列を配列化
        for col_name in [category_name, subcategory_name]:
            human_df[col_name] = human_df[col_name].apply(lambda x: ast.literal_eval(x))
            llm_df[col_name] = llm_df[col_name].apply(lambda x: ast.literal_eval(x))

        assert human_df.shape == llm_df.shape
        assert human_df["id"].equals(llm_df["id"])

        self.llm_human_df = pd.merge(
            llm_df,
            human_df,
            how="inner",
            on="id",
            suffixes=("_llm", "_human"),
        )

    def evaluate(self) -> LlmAnnotationResult:
        category_result = self._calculate(category_name)
        subcategory_result = self._calculate(subcategory_name)
        result = LlmAnnotationResult(
            model_type=model_type,
            random_seed=random_seed,
            half_number=half_number,
            comment_ids=self.llm_human_df["id"].tolist(),
            category=category_result,
            subcategory=subcategory_result,
        )
        return result

    def _calculate(self, col_name) -> EvalIndicator:
        er = self.llm_human_df.apply(
            lambda row: row[col_name + "_llm"] == row[col_name + "_human"],
            axis=1,
        ).mean()
        accuracy = self.llm_human_df.apply(
            lambda row: len(set(row[col_name + "_llm"]) & set(row[col_name + "_human"]))
            / len(set(row[col_name + "_llm"]) | set(row[col_name + "_human"])),
            axis=1,
        ).mean()
        precision = self.llm_human_df.apply(
            lambda row: len(set(row[col_name + "_llm"]) & set(row[col_name + "_human"]))
            / len(set(row[col_name + "_llm"])),
            axis=1,
        ).mean()
        recall = self.llm_human_df.apply(
            lambda row: len(set(row[col_name + "_llm"]) & set(row[col_name + "_human"]))
            / len(set(row[col_name + "_human"])),
            axis=1,
        ).mean()

        f1 = 2 * (precision * recall) / (precision + recall)

        return EvalIndicator(
            exact_match=er, accuracy=accuracy, precision=precision, recall=recall, f1=f1
        )


def preprocess_human_annotation():
    human_df = pd.read_csv(HUMAN_ANOTATION_CSV_PATH)
    human_df[category_name] = human_df[category_name].str.split(" ")
    human_df[subcategory_name] = human_df[subcategory_name].astype(str).str.split(" ")

    # 文字列として格納されている配列を配列化
    human_df[category_name] = human_df[category_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    human_df[subcategory_name] = human_df[subcategory_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 配列化されていないものを配列化
    human_df[category_name] = human_df[category_name].apply(
        lambda x: [x] if not isinstance(x, list) and not pd.isna(x) else x
    )
    human_df[subcategory_name] = human_df[subcategory_name].apply(
        lambda x: [x] if not isinstance(x, list) and not pd.isna(x) else x
    )

    # 格納されている配列の中の文字列となっている数字を小数点に変換
    human_df[category_name] = human_df[category_name].apply(
        lambda x: [int(i) if i not in ["None", "nan"] else None for i in x]
        if isinstance(x, list)
        else x
    )
    human_df[subcategory_name] = human_df[subcategory_name].apply(
        lambda x: [float(i) if i not in ["None", "nan"] else None for i in x]
        if isinstance(x, list)
        else x
    )

    # 大分類に格納されてる配列の要素数に小分類を合わせる.
    # その際、大分類が2のindexと小分類のNoneではないindexが対応するようにpaddingする
    for i, row in human_df.iterrows():
        assert len(row[category_name]) >= len(row[subcategory_name])

        if len(row[category_name]) != len(row[subcategory_name]):
            renewal_small_category = [None] * len(row[category_name])
            if 2 in row[category_name]:
                two_index = list(row[category_name]).index(2)
                renewal_small_category[two_index] = row[subcategory_name][0]
            human_df.at[i, subcategory_name] = renewal_small_category

    human_df.to_csv(
        str(HUMAN_ANOTATION_CSV_PATH).replace(".csv", "_preprocessed.csv"),
        index=False,
    )


if __name__ == "__main__":
    time_str = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    logger.add(
        "logs/evaluate_llm_annotation_{time}.log".format(
            time=time_str,
        )
    )
    # preprocess_human_annotation()
    evaluator = EvaluateAnnotation()
    result = evaluator.evaluate()
    logger.info(result)
    # {'category': [1, 2], 'subcategory': [1.8, None]}
