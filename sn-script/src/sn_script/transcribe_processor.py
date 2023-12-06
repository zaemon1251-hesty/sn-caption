import nltk
from typing import List, Dict
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

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
    from sn_script.json2csv import write_csv
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
    from src.sn_script.json2csv import write_csv


RAW_JSON_TEMPLATE = "{half_number}_224p.json"
DENOISED_TEXT_TEMPLATE = "denoised_{half_number}_224p.txt"
DENISED_JSONLINE_TEPMLATE = "denoised_{half_number}_224p.jsonl"
DENISED_TOKENIZED_TEPMLATE = "denoised_{half_number}_tokenized_224p.txt"
DENISED_TOKENIZED_JSONLINE_TEPMLATE = "denoised_{half_number}_tokenized_224p.jsonl"
DENISED_TOKENIZED_CSV_TEPMLATE = "denoised_{half_number}_tokenized_224p.csv"


def tokenize_sentense(half_number: int):
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        denoised_txt_path = (
            Config.base_dir
            / target
            / DENOISED_TEXT_TEMPLATE.format(half_number=half_number)
        )
        tokenized_txt_path = (
            Config.base_dir
            / target
            / DENISED_TOKENIZED_TEPMLATE.format(half_number=half_number)
        )
        with open(denoised_txt_path, "r") as f:
            text = f.read()
        with open(tokenized_txt_path, "w") as f:
            f.write("\n".join(nltk.sent_tokenize(text)))


def create_tokenized_data(half_number: int):
    def _run(
        tokenized_txt_path: str, denoised_jsonl_path: str, tokenized_jsonl_path: str
    ):
        result = []
        with open(tokenized_txt_path, "r") as f:
            tokenized_texts = f.readlines()

        with open(denoised_jsonl_path, "r") as f:
            denoised_data = [json.loads(line) for line in f.readlines()]
        tokenized_texts = [t.rstrip("\n") for t in tokenized_texts]

        current_segment_idx = 0
        end_segments_idx = len(denoised_data) - 1

        for tokenized_text in tokenized_texts:
            start_span_idx = current_segment_idx
            end_span_idx = current_segment_idx

            covering_text = denoised_data[start_span_idx]["text"]
            while end_span_idx <= end_segments_idx:
                if tokenized_text in covering_text:
                    break
                end_span_idx += 1
                covering_text += " " + denoised_data[end_span_idx]["text"]

            tokinized_data = {
                "text": tokenized_text,
                "start": denoised_data[start_span_idx]["start"],
                "end": denoised_data[end_span_idx]["end"],
            }
            result.append(tokinized_data)
            current_segment_idx = end_span_idx

        for tokenized_data in result:
            with open(tokenized_jsonl_path, "a") as f:
                json.dump(tokenized_data, f)
                f.write("\n")

    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        tokenized_txt_path = (
            Config.base_dir
            / target
            / DENISED_TOKENIZED_TEPMLATE.format(half_number=half_number)
        )
        denoised_jsonl_path = (
            Config.base_dir
            / target
            / DENISED_JSONLINE_TEPMLATE.format(half_number=half_number)
        )
        tokenized_jsonl_path = (
            Config.base_dir
            / target
            / DENISED_TOKENIZED_JSONLINE_TEPMLATE.format(half_number=half_number)
        )
        _run(tokenized_txt_path, denoised_jsonl_path, tokenized_jsonl_path)


def create_csv_tokenized_sentenses(half_number: int):
    for target in tqdm(Config.targets):
        target: str = target.rstrip("/").split("/")[-1]
        tokenized_jsonl_path = (
            Config.base_dir
            / target
            / DENISED_TOKENIZED_JSONLINE_TEPMLATE.format(half_number=half_number)
        )
        tokenized_csv_path = (
            Config.base_dir
            / target
            / DENISED_TOKENIZED_CSV_TEPMLATE.format(half_number=half_number)
        )
        data = [json.loads(line) for line in open(tokenized_jsonl_path, "r")]
        for i, d in enumerate(data, start=1):
            d["id"] = i
        write_csv(data, tokenized_csv_path)


def denoise_sentenses(half_number: int):
    def preprocess_data(sentenses_data: List[Dict[str, str]]):
        result = []
        for sts in sentenses_data:
            sts["text"] = sts["text"].replace("\n", "").strip()
            result.append(sts)
        return result

    def is_noise(sentense, prev_sentense):
        result = sentense == prev_sentense
        return result

    def remove_noise(sentenses_data: List[Dict[str, str]]):
        result = []
        prev_sentense = ""
        for sts in sentenses_data:
            if is_noise(sts["text"], prev_sentense):
                continue
            result.append(sts)
            prev_sentense = sts["text"]
        return result

    def dump_denoised_data(
        sentenses_data: List[Dict[str, str]],
        denoised_txt_path: str,
        denoised_jsonl_path: str,
    ):
        texts = [sts["text"] for sts in sentenses_data]
        with open(denoised_txt_path, "w") as f:
            f.write(" ".join(texts))
        for sts in sentenses_data:
            with open(denoised_jsonl_path, "a") as f:
                json.dump(sts, f)
                f.write("\n")

    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        json_txt_path = (
            Config.base_dir / target / RAW_JSON_TEMPLATE.format(half_number=half_number)
        )
        denoised_txt_path = (
            Config.base_dir
            / target
            / DENOISED_TEXT_TEMPLATE.format(half_number=half_number)
        )
        denoised_jsonl_path = (
            Config.base_dir
            / target
            / DENISED_JSONLINE_TEPMLATE.format(half_number=half_number)
        )
        raw_data = json.load(open(json_txt_path, "r"))["segments"]
        preprocessed_data = preprocess_data(raw_data)
        denoised_data = remove_noise(preprocessed_data)
        print(f"before: {len(preprocessed_data)}, after: {len(denoised_data)}")
        dump_denoised_data(denoised_data, denoised_txt_path, denoised_jsonl_path)


def round_down(half_number: int):
    """小数点を2桁に丸める"""
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        tokenized_jsonl_path = (
            Config.base_dir
            / target
            / DENISED_TOKENIZED_JSONLINE_TEPMLATE.format(half_number=half_number)
        )
        data = [json.loads(line) for line in open(tokenized_jsonl_path, "r")]
        for d in data:
            d["start"] = round(d["start"], 2)
            d["end"] = round(d["end"], 2)
        with open(tokenized_jsonl_path, "w") as f:
            for d in data:
                json.dump(d, f)
                f.write("\n")


def main():
    denoise_sentenses(half_number)
    tokenize_sentense(half_number)
    create_tokenized_data(half_number)
    round_down(half_number)
    create_csv_tokenized_sentenses(half_number)


if __name__ == "__main__":
    # nltk.download("punkt")
    main()
