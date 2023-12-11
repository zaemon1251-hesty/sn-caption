from pathlib import Path
import os
from SoccerNet.utils import getListGames


# 　分析対象のデータセットのパス
class Config:
    # base_dir = Path(__file__).parent.parent.parent.parent / "data"
    base_dir = Path("/raid_elmo/home/lr/moriy")
    target_base_dir = Path(__file__).parent.parent / "data"
    target_file_path = target_base_dir / "all_targets.txt"
    targets = [
        os.path.join("SoccerNet", target)
        for target in open(target_file_path, "r").read().strip().split("\n")
    ]


# 分析対象のカラム名
binary_category_name = "付加的情報か"
category_name = "大分類"
subcategory_name = "小分類"


# ターゲットの設定
random_seed = 42
half_number = 2

# 使用するLLMのモデル名
model_type = "gpt-3.5-turbo-1106"
# model_type = "gpt-4-1106-preview"

if __name__ == "__main__":
    print(len(Config.targets))
    print(*Config.targets, sep="\n")
