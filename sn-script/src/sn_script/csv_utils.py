import pandas as pd
from collections import defaultdict
from pathlib import Path
import ast
import pprint

try:
    from sn_script.config import Config
    from sn_script.json2csv import write_csv
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


category_name = "大分類"
subcategory_name = "小分類"

random_seed = 42
half_number = 1

HUMAN_ANOTATION_CSV_PATH = (
    Config.base_dir / f"{random_seed}_{half_number}_moriy_annotation_preprocessed.csv"
)


def output_label_statistics(csv_path: str | Path):
    label_counts = get_categories_ratio(csv_path)
    result = pprint.pformat(label_counts, depth=2, width=40, indent=2)
    print(result)


def get_categories_ratio(csv_path: str | Path):
    all_game_df = pd.read_csv(csv_path)
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


def get_average_num_comments(half_number: int) -> int:
    lens = []
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / f"{half_number}_224p.csv"
        tmp_df = pd.read_csv(csv_path)
        lens.append(len(tmp_df))
    return sum(lens) / len(lens)


def dump_filled_comments(half_number: int):
    DUMP_FILE_PATH = f"filled_big_class_{half_number}.csv"

    df_list = []
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / f"{half_number}_224p.csv"
        tmp_df = pd.read_csv(csv_path)
        tmp_df["game"] = target.replace("SoccerNet/", "")
        df_list.append(tmp_df)

    all_game_df = pd.concat(df_list)
    filled_big_class_df = all_game_df.loc[
        ~all_game_df[category_name].isnull()
    ].reset_index(drop=True)
    filled_big_class_df.to_csv(DUMP_FILE_PATH, index=False)


def clean():
    half_number = 1
    DUMP_FILE_PATH = f"filled_big_class_{half_number}.csv"
    df = pd.read_csv(DUMP_FILE_PATH)
    print(df.columns)
    df.drop(columns=["Unnamed: 6", "Unnamed: 7"], inplace=True)
    df.to_csv(DUMP_FILE_PATH, index=False)


def create_tokonized_all_csv():
    half_number = 1
    ALL_CSV_PATH = Config.base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
    DENISED_TOKENIZED_CSV_TEMPLATE = f"denoised_{half_number}_tokenized_224p.csv"

    df_list = []
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / DENISED_TOKENIZED_CSV_TEMPLATE
        tmp_df = pd.read_csv(csv_path)
        tmp_df["game"] = target.replace("SoccerNet/", "")
        df_list.append(tmp_df)

    all_game_df = pd.concat(df_list)
    all_game_df = (
        all_game_df.reindex(
            columns=[
                "id",
                "game",
                "start",
                "end",
                "text",
                category_name,
                subcategory_name,
                "備考",
            ]
        )
        .sort_values(by=["game", "start", "end"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    all_game_df["id"] = all_game_df.index
    all_game_df.to_csv(ALL_CSV_PATH, index=False, encoding="utf-8_sig")


def create_tokenized_annotation_csv():
    half_number = 1
    number_of_comments = 100
    random_seed = 42

    ALL_CSV_PATH = Config.base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
    ANNOTATION_CSV_PATH = (
        Config.base_dir / f"{random_seed}_{half_number}_tokenized_224p_annotation.csv"
    )
    all_game_df = pd.read_csv(ALL_CSV_PATH)

    annotation_df = all_game_df.sample(n=number_of_comments, random_state=random_seed)
    annotation_df.to_csv(ANNOTATION_CSV_PATH, index=False, encoding="utf-8_sig")


if __name__ == "__main__":
    # create_tokenized_annotation_csv()
    output_label_statistics(HUMAN_ANOTATION_CSV_PATH)
