from __future__ import annotations
import pandas as pd
from collections import defaultdict
import ast
import pprint
from loguru import logger
import json
from whisper.tokenizer import LANGUAGES

try:
    from sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        subcategory_name,
        random_seed,
        half_number,
        model_type,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import (
        Config,
        binary_category_name,
        category_name,
        subcategory_name,
        random_seed,
        half_number,
        model_type,
    )

RAW_JSON_TEMPLATE = f"{half_number}_224p.json"

ALL_CSV_PATH = (
    Config.target_base_dir / f"{model_type}_500game_{half_number}_llm_annotation.csv"
)

logger.debug("loading csv...")
all_game_df = pd.read_csv(ALL_CSV_PATH)
logger.debug("loaded csv")


def get_label_statistics(binary: bool = False):
    if binary:
        label_counts = get_binary_categories_ratio()
        return
    else:
        label_counts = get_categories_ratio()
    result = pprint.pformat(label_counts, depth=2, width=40, indent=2)
    return result


def get_binary_categories_ratio():
    label_counts = all_game_df[binary_category_name].value_counts(dropna=False)
    logger.info(f"binary label counts: {label_counts}")

    return label_counts


def get_categories_ratio():
    filled_category_df = all_game_df
    all_game_df[category_name] = all_game_df[category_name].apply(
        lambda r: ast.literal_eval(r)
    )
    all_game_df[subcategory_name] = all_game_df[subcategory_name].apply(
        lambda r: ast.literal_eval(r)
    )

    # 大分類はマルチラベルなので、大分類の数はデータ数よりも多い
    # 大分類のマルチラベルを分割して数える

    # ユニークラベルの数を数える
    label_counts = {
        category_name: defaultdict(int),
        subcategory_name: defaultdict(int),
    }

    # Iterate over the  column
    for column in [category_name, subcategory_name]:
        for values in filled_category_df[column]:
            # Split the value by space and strip extra spaces
            assert isinstance(values, list)
            for value in values:
                labels = value
                label_counts[column][labels] += 1

    return label_counts


def get_average_num_comments() -> int:
    result = all_game_df.groupby("game").count().mean()
    return result


def get_number_audio_launguages():
    language_counts = defaultdict(int)

    for target in Config.targets:
        # target: str = target.rstrip("/").split("/")[-1]
        json_path = Config.base_dir / target / RAW_JSON_TEMPLATE
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        language_counts[LANGUAGES[raw_data["language"]]] += 1
    resutl_str = pprint.pformat(dict(language_counts), depth=2, width=40, indent=2)
    return resutl_str


def get_num_with_audio_commentary():
    games = open(Config.target_file_path, "r").readlines()
    game_num = len(games)
    return game_num


if __name__ == "__main__":
    logger.info("500game")
    logger.info(f"half_number: {half_number}")
    logger.info(f"model_type: {model_type}")
    logger.info(f"total lenght: {len(all_game_df)}")
    logger.info(get_label_statistics(binary=True))
    logger.info(f"average number of comments per game: {get_average_num_comments()}")
    logger.info(f"language ratio: {get_number_audio_launguages()}")
    logger.info(
        f"number of games with audio commentary: {get_num_with_audio_commentary()}"
    )
