from __future__ import annotations
from openai import OpenAI
import json
import os
import pandas as pd
from collections import namedtuple
import yaml
from loguru import logger
from datetime import datetime

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config

# プロンプト作成用の引数
PromptArgments = namedtuple("PromptArgments", ["comment", "game", "previous_comments"])

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
PROMPT_YAML_PATH = (
    Config.base_dir.parent / "sn-script" / "src" / "resources" / "classify_comment.yaml"
)

# load csv
all_comment_df = pd.read_csv(ALL_CSV_PATH)

# load yaml
prompt_config = yaml.safe_load(open(PROMPT_YAML_PATH, "r"))


def main():
    logger.add(
        "logs/llm_anotator_{time}.log".format(
            time=datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    target_comment_id = 833
    model_type = "gpt-3.5-turbo-1106"
    print(prompt_config)
    print(get_messages(target_comment_id))
    print(
        classify_comment(model_type, target_comment_id)
    )  # {'category': [1, 2], 'subcategory': [1.8, None]}


def classify_comment(model_type: str, comment_id: int) -> dict:
    messages = get_messages(comment_id)

    completion_params = {
        "model": model_type,
        "messages": messages,
        "n": 1,
        "stop": None,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    response = client.chat.completions.create(**completion_params)
    if response.choices[0].message.content is None:
        return {}
    else:
        return json.loads(response.choices[0].message.content)


def get_messages(comment_id: int) -> list[str]:
    messages = []

    target_comment = all_comment_df.iloc[comment_id]
    if target_comment.empty:
        raise ValueError(f"comment_id={comment_id} is not found.")

    description = prompt_config["description"]
    messages.append(
        {
            "role": "system",
            "content": description,
        }
    )

    shots = prompt_config["shots"]
    for shot in shots:
        messages.append(
            {
                "role": "user",
                "content": shot["user"],
            },
        )
        messages.append(
            {
                "role": "assistant",
                "content": shot["assistant"],
            }
        )

    # max_history = 5 & game == game
    previous_comments = (
        all_comment_df[
            (all_comment_df["game"] == target_comment["game"])
            & (all_comment_df.index < comment_id)
        ]
        .tail(5)["text"]
        .tolist()
    )

    target_prompt_args = PromptArgments(
        target_comment["text"], target_comment["game"], previous_comments
    )
    target_prompt = create_target_prompt(target_prompt_args)
    messages.append(
        {
            "role": "user",
            "content": target_prompt,
        }
    )
    return messages


def create_target_prompt(prompt_args: PromptArgments) -> str:
    """分類対象のコメントに関するプロンプトを作成する"""

    message = f"""
- comment: {prompt_args.comment}
- game: {prompt_args.game}
- previous_comments: {" ".join(prompt_args.previous_comments)}
"""

    return message


if __name__ == "__main__":
    main()
