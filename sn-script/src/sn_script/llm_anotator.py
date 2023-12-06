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

# プロンプト作成用の引数
PromptArgments = namedtuple("PromptArgments", ["comment", "game", "previous_comments"])

# APIキーの設定
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)


ALL_CSV_PATH = Config.base_dir / f"denoised_{half_number}_tokenized_224p_all.csv"
ANNOTATION_CSV_PATH = (
    Config.base_dir
    / f"{random_seed}_denoised_{half_number}_tokenized_224p_annotation.csv"
)
PROMPT_YAML_PATH = (
    Config.base_dir.parent / "sn-script" / "src" / "resources" / "classify_comment.yaml"
)

LLM_ANOTATION_CSV_PATH = (
    Config.base_dir / f"{model_type}_{random_seed}_{half_number}_llm_annotation.csv"
)


# load csv
all_comment_df = pd.read_csv(ALL_CSV_PATH)

# load yaml
prompt_config = yaml.safe_load(open(PROMPT_YAML_PATH, "r"))


def main():
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.add(
        "logs/llm_anotator_{time}.log".format(time=time_str),
    )

    annotation_df = pd.read_csv(ANNOTATION_CSV_PATH)

    def fill_category(row):
        comment_id = row["id"]
        if not pd.isnull(row[category_name]):
            return row[category_name], row[subcategory_name]
        try:
            result = classify_comment(model_type, comment_id)
            logger.info(f"comment_id:{comment_id},result:{result}")
            category = result.get("category")
            subcategory = result.get("subcategory")
            logger.info(f"comment_id={comment_id} is annotated.")
            return category, subcategory
        except Exception as e:
            logger.error(f"comment_id={comment_id} is not annotated.")
            logger.error(e)
            return None, None

    def fill_category_binary(row):
        comment_id = row["id"]
        if not pd.isnull(row[binary_category_name]):
            return row[binary_category_name]
        try:
            result = classify_comment(model_type, comment_id)
            logger.info(f"comment_id:{comment_id},result:{result}")
            category = result.get("category")
            logger.info(f"comment_id={comment_id} is annotated.")
            return category
        except Exception as e:
            logger.error(f"comment_id={comment_id} couldn't be annotated.")
            logger.error(e)
            return None

    # annotation_df[[category_name, subcategory_name]] = annotation_df.apply(
    #     lambda r: fill_category(r), axis=1, result_type="expand"
    # )
    annotation_df[binary_category_name] = annotation_df.apply(
        lambda r: fill_category_binary(r), axis=1
    )
    annotation_df.to_csv(LLM_ANOTATION_CSV_PATH, index=False)

    # print(classify_comment(model_type, 833))
    # # {'category': [1, 2], 'subcategory': [1.8, None]}


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
    content = response.choices[0].message.content
    if content is None:
        return {}
    else:
        return json.loads(content)


def get_messages(comment_id: int) -> list[str]:
    messages = []

    if comment_id not in all_comment_df.index:
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

    target_prompt = create_target_prompt(comment_id)
    messages.append(
        {
            "role": "user",
            "content": target_prompt,
        }
    )
    return messages


def create_target_prompt(comment_id: int) -> str:
    target_comment = all_comment_df.iloc[comment_id]

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

    message = _create_target_prompt(target_prompt_args)
    return message


def _create_target_prompt(prompt_args: PromptArgments) -> str:
    """分類対象のコメントに関するプロンプトを作成する"""

    message = f"""
- comment: {prompt_args.comment}
- game: {prompt_args.game}
- previous_comments: {" ".join(prompt_args.previous_comments)}
"""

    return message


if __name__ == "__main__":
    main()

    # ChatGPT用のプロンプトを作成する
    TARGET_PROMPT_CSV_PATH = (
        Config.base_dir.parent
        / "sn-script"
        / "src"
        / "resources"
        / f"{random_seed}_{half_number}_target_prompt.csv"
    )

    def output_target_prompt():
        annotation_df = pd.read_csv(ANNOTATION_CSV_PATH)
        targe_prompt_list = []
        for comment_id in annotation_df["id"]:
            targe_prompt_list.append(create_target_prompt(comment_id))

        with open(TARGET_PROMPT_CSV_PATH, "w") as f:
            f.write("\n".join(targe_prompt_list))

    # output_target_prompt()
