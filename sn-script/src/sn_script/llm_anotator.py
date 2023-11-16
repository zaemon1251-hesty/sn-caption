from __future__ import annotations
from openai import OpenAI
import json
import os
import pandas as pd

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


# APIキーの設定
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

# 利用ファイルの設定
random_seed = 42
half_number = 1

ALL_CSV_PATH = Config.base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
ANNOTATION_CSV_PATH = (
    Config.base_dir
    / f"{random_seed}_denoised_{half_number}_tokenized_224p_annotation.csv"
)


def classify_comment(comment_id: int) -> dict:
    model = "gpt-3.5-turbo-1106"
    messages = get_messages(comment_id)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        stop=None,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    if response.choices[0].message.content is None:
        return {}
    else:
        return json.loads(response.choices[0].message.content)


def get_messages(comment_id: int) -> list[str]:
    messages = []
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        csv_path = Config.base_dir / target / f"{comment_id}_224p.csv"
        tmp_df = pd.read_csv(csv_path)
        messages.extend(tmp_df["コメント"].tolist())
    return messages


if __name__ == "__main__":
    target_comment_id = 833
    print(classify_comment(target_comment_id))
