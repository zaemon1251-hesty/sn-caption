from __future__ import annotations
import sys
from dataclasses import dataclass
from glob import glob
import json
from collections import defaultdict
import pandas as pd
import datetime
from logging import getLogger

logger = getLogger(__name__)

sys.path.append("/raid_elmo/home/lr/moriy/sn-caption")

import Benchmarks.TemporallyAwarePooling.src.dataset as dataset

tzinfo = datetime.timezone(datetime.timedelta(hours=9))


@dataclass(frozen=True)
class Config(object):
    SoccerNet_path = "/raid_elmo/home/lr/moriy/SoccerNet/"
    features = "baidu_soccer_embeddings.npy"
    output_file_pattern = "**/outputs/test/**/results_dense_captioning.json"
    split_test = ["test"]
    version = 2
    framerate = 2
    window_size_caption = 45
    results_csv = (
        datetime.datetime.utcnow().astimezone(tzinfo).strftime("%Y%m%d%H%m")
        + "word_counts.csv"
    )


def main(args: Config) -> int:
    args = Config()
    dataset_Test = dataset.SoccerNetCaptions(
        path=args.SoccerNet_path,
        features=args.features,
        split=args.split_test,
        version=args.version,
        framerate=args.framerate,
        window_size=args.window_size_caption,
    )
    texts = get_output_texts(args.output_file_pattern)
    data = process_texts(texts, dataset_Test.text_processor)

    counter = defaultdict(int)
    for processed_texts in data:
        for word in processed_texts:
            counter[word] += 1

    word_counts = [
        (dataset_Test.text_processor.detokenize([word]), count)
        for word, count in counter.items()
    ]

    save_word_counts(word_counts, args.results_csv)

    return 0


def get_output_texts(output_file_pattern: str):
    results = []
    pattern = output_file_pattern
    output_files = glob(pattern, recursive=True)
    print(output_files)
    for filepath in output_files:
        with open(filepath, "r") as f:
            data = json.load(f)
            if "predictions" not in data:
                continue
            for comment_data in data["predictions"]:
                target_text = comment_data["comment"]
                results.append(target_text)
    return results


def process_texts(
    texts: list[str], processor: dataset.SoccerNetTextProcessor
) -> list[list[int]]:
    results = []
    for text in texts:
        processed = processor(text)
        print(processed)
        results.append(processed)
    return results


def save_word_counts(word_counts: list[tuple[str, int]], filepath: str):
    df = pd.DataFrame(word_counts, columns=["word", "count"])
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    arg = Config()
    main(arg)
