import nltk
from typing import List, Dict
import json
from pathlib import Path
from loguru import logger

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    class Config:
        base_dir = Path(__file__).parent.parent.parent.parent / "data"
        targets = [
            "SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/",
            "SoccerNet/england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea/",
            "SoccerNet/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/",
            "SoccerNet/europe_uefa-champions-league/2014-2015/2014-11-04 - 20-00 Zenit Petersburg 1 - 2 Bayer Leverkusen/",
            "SoccerNet/europe_uefa-champions-league/2015-2016/2015-09-15 - 21-45 Galatasaray 0 - 2 Atl. Madrid/",
            "SoccerNet/europe_uefa-champions-league/2016-2017/2016-09-13 - 21-45 Barcelona 7 - 0 Celtic/",
            "SoccerNet/france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/",
            "SoccerNet/france_ligue-1/2016-2017/2017-01-21 - 19-00 Nantes 0 - 2 Paris SG/",
            "SoccerNet/germany_bundesliga/2014-2015/2015-02-21 - 17-30 Paderborn 0 - 6 Bayern Munich/",
            "SoccerNet/germany_bundesliga/2015-2016/2015-08-29 - 19-30 Bayern Munich 3 - 0 Bayer Leverkusen/",
            "SoccerNet/germany_bundesliga/2016-2017/2016-09-10 - 19-30 RB Leipzig 1 - 0 Dortmund/",
            "SoccerNet/italy_serie-a/2014-2015/2015-02-15 - 14-30 AC Milan 1 - 1 Empoli/",
            "SoccerNet/italy_serie-a/2016-2017/2016-08-20 - 18-00 Juventus 2 - 1 Fiorentina/",
            "SoccerNet/spain_laliga/2014-2015/2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna/",
            "SoccerNet/spain_laliga/2015-2016/2015-08-29 - 21-30 Barcelona 1 - 0 Malaga/",
            "SoccerNet/spain_laliga/2016-2017/2017-05-21 - 21-00 Malaga 0 - 2 Real Madrid/",
            "SoccerNet/spain_laliga/2019-2020/2019-08-17 - 18-00 Celta Vigo 1 - 3 Real Madrid/",
        ]

DENOISED_TEXT_TEMPATH = Config.base_dir / "denoised_text_term.txt"


def tokenize_sentense(half_number: int):
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        denoised_txt_path = (
            Config.base_dir / target / f"denoised_{half_number}_224p.txt"
        )
        tokenized_txt_path = (
            Config.base_dir / target / f"denoised_{half_number}_tokenized_224p.txt"
        )
        with open(denoised_txt_path, "r") as f:
            text = f.read()
        with open(tokenized_txt_path, "w") as f:
            f.write("\n".join(nltk.sent_tokenize(text)))


def create_tokenized_data(half_number: int):
    def _run(tokenized_txt_path: str, denoised_jsonl_path: str, tokenized_jsonl_path: str):
        result = []

        with open(tokenized_txt_path, "r") as f:
            tokenized_texts = f.readlines()

        with open(denoised_jsonl_path, "r") as f:
            denoised_data = [json.loads(line) for line in f.readlines()]
        tokenized_texts = [t.rstrip("\n") for t in tokenized_texts]

        current_start = 0
        current_end = 0
        current_data_index = 0

        start_offset = 0

        for tokenized_text in tokenized_texts:
            tokenized_data = {
                "text": tokenized_text,
            }

            if len(tokenized_text) <= len(denoised_data[current_data_index]["text"]) and \
                                    tokenized_text in denoised_data[current_data_index]["text"]:
                duration = denoised_data[current_data_index]["end"] - denoised_data[current_data_index]["start"]
                tokenized_data["start"] = denoised_data[current_data_index]["start"] + start_offset
                start_offset += duration * len(tokenized_text) / len(denoised_data[current_data_index]["text"])
                tokenized_data["end"] = denoised_data[current_data_index]["start"] + start_offset
                result.append(tokenized_data)
                continue

            # ここまでたどり着いた時、 tokenized_text が denoised_data[current_data_index]["text"] に含まれていないことが確定する
            # 以降、tokenized_text が denoised_data[current_data_index]["text"] より長いという話で進める
            start_offset = 0

            # start segment を探す
            try:
                while True:
                    if denoised_data[current_data_index]["text"] in tokenized_text:
                        current_start = denoised_data[current_data_index]["start"]
                        break
                    current_data_index += 1
                tokenized_data["start"] = current_start
            except IndexError:
                logger.error(f"IndexError: {tokenized_text}")
                logger.error(f"IndexError: {denoised_data[-1]}")
                continue
            # end segment を探す
            while True:
                if current_data_index >= len(tokenized_data) - 1 or denoised_data[current_data_index + 1]["text"] not in tokenized_text:
                    current_end = denoised_data[current_data_index]["end"]
                    break
                current_data_index += 1
            tokenized_data["end"] = current_end

            result.append(tokenized_data)

        for tokenized_data in result:
            with open(tokenized_jsonl_path, "a") as f:
                json.dump(tokenized_data, f)
                f.write("\n")

    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        tokenized_txt_path = (
            Config.base_dir / target / f"denoised_{half_number}_tokenized_224p.txt"
        )
        denoised_jsonl_path = (
            Config.base_dir / target / f"denoised_{half_number}_224p.jsonl"
        )
        tokenized_jsonl_path = (
            Config.base_dir / target / f"denoised_{half_number}_tokenized_224p.jsonl"
        )
        _run(tokenized_txt_path, denoised_jsonl_path, tokenized_jsonl_path)




def create_csv_tokenized_sentenses(half_number: int):
    pass


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

    def dump_denoised_data(sentenses_data: List[Dict[str, str]], denoised_txt_path: str, denoised_jsonl_path: str):
        texts = [sts["text"] for sts in sentenses_data]
        with open(denoised_txt_path, "w") as f:
            f.write(" ".join(texts))
        for sts in sentenses_data:
            with open(denoised_jsonl_path, "a") as f:
                json.dump(sts, f)
                f.write("\n")

    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        json_txt_path = Config.base_dir / target / f"{half_number}_224p.json"
        denoised_txt_path = (
            Config.base_dir / target / f"denoised_{half_number}_224p.txt"
        )
        denoised_jsonl_path = (
            Config.base_dir / target / f"denoised_{half_number}_224p.jsonl"
        )
        raw_data = json.load(open(json_txt_path, "r"))["segments"]
        preprocessed_data = preprocess_data(raw_data)
        denoised_data = remove_noise(preprocessed_data)
        print(f"before: {len(preprocessed_data)}, after: {len(denoised_data)}")
        dump_denoised_data(denoised_data, denoised_txt_path, denoised_jsonl_path)


def main():
    half_number = 1
    # denoise_sentenses(half_number)
    # tokenize_sentense(half_number)
    create_tokenized_data(half_number)
    # create_csv_tokenized_sentenses(half_number)


if __name__ == "__main__":
    # nltk.download("punkt")
    main()
