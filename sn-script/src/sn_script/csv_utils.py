import pandas as pd
from collections import defaultdict

try:
    from sn_script.config import Config
    from sn_script.json2csv import write_csv
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


def get_gig_class_ratio(half_number: int):
    df_list = []

    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / f"{half_number}_224p.csv"
        tmp_df = pd.read_csv(csv_path)
        df_list.append(tmp_df)

    all_game_df = pd.concat(df_list)
    filled_big_class_df = all_game_df.loc[all_game_df["大分類"] != ""].reset_index(
        drop=True
    )

    # 大分類はマルチラベルなので、大分類の数はデータ数よりも多い
    # 大分類のマルチラベルを分割して数える
    filled_big_class_df["大分類"] = filled_big_class_df["大分類"].astype(str)

    # ユニークラベルを
    label_counts = defaultdict(int)

    # Iterate over the '大分類' column
    for value in filled_big_class_df["大分類"]:
        # Split the value by space and strip extra spaces
        labels = value
        label_counts[labels] += 1

    # Convert the dictionary to a sorted list of tuples for readability
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print(label_counts)


def get_average_num_comments(half_number: int) -> int:
    lens = []
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / f"{half_number}_224p.csv"
        tmp_df = pd.read_csv(csv_path)
        lens.append(len(tmp_df))
    return sum(lens) / len(lens)


def dump_filled_coments(half_number: int):
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
        all_game_df["大分類"].isnull() == False
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
        all_game_df
        .reindex(columns=["id", "game", "start", "end", "text", "大分類", "小分類", "備考"])
        .sort_values(by=["game", "start", "end"],ascending=[True, True, True])
        .reset_index(drop=True)
    )

    all_game_df["id"] = all_game_df.index
    all_game_df.to_csv(ALL_CSV_PATH, index=False, encoding="utf-8_sig")

def create_tokenized_annotation_csv():
    half_number = 1
    number_of_comments = 100
    random_seed = 42

    ALL_CSV_PATH = Config.base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
    ANNOTATION_CSV_PATH = Config.base_dir / f"denoised_{half_number}_tokenized_224p_annotation.csv"
    all_game_df = pd.read_csv(ALL_CSV_PATH)

    annotation_df = all_game_df.sample(n=number_of_comments, random_state=random_seed)
    annotation_df.to_csv(ANNOTATION_CSV_PATH, index=False, encoding="utf-8_sig")


if __name__ == "__main__":
    create_tokonized_all_csv()
    create_tokenized_annotation_csv()
