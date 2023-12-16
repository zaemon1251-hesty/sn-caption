from pathlib import Path
import os


# 　分析対象のデータセットのパス
class Config:
    # base_dir = Path(__file__).parent.parent.parent.parent / "data"
    base_dir = Path("/raid_elmo/home/lr/moriy")
    target_base_dir = Path(__file__).parent.parent / "data"
    target_file_path = target_base_dir / "exist_targets.txt"
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
# model_type = "meta-llama/Llama-2-70b-chat-hf"

if __name__ == "__main__":
    # print(len(Config.targets))
    listdir = []
    for target in Config.targets:
        path = Config.base_dir / target
        if not Path(path).exists():
            listdir.append(target)

    all_gamedir = set(Config.targets)
    listdir = set(listdir)
    print(*(all_gamedir - listdir), sep="\n")
