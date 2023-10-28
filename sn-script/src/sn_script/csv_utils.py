from sn_script.config import Config
import pandas as pd
from collections import defaultdict


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


if __name__ == "__main__":
    clean()
