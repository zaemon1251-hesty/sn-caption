import whisper
from pathlib import Path
import json
from tqdm import tqdm
import os
import datetime
import logging

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys
    sys.path.append(".")
    from src.sn_script.config import Config


class Logger:
    """save log"""

    def __init__(self, path):
        self.general_logger = logging.getLogger(path)
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(os.path.join(path, "Experiment.log"))
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)

    def info(self, message):
        # display time
        self.general_logger.info("[{}] - {}".format(self.now_string(), message))

    @staticmethod
    def now_string():
        return str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


log_path = Path(__file__).parent.parent / "logs"
logger = Logger(log_path.__str__())


def run_transcribe(model, video_dir_path: Path):
    half_number = 1
    decode_lang = "ja"
    VIDEO_DIR = video_dir_path
    VIDEO_PATH = VIDEO_DIR / f"{half_number}_224p.mkv"
    TRANSCRIBE_TEXT_PATH = VIDEO_DIR / f"{half_number}_224p.{decode_lang}.txt"
    TRANSCRIBE_JSON_PATH = VIDEO_DIR / f"{half_number}_224p.{decode_lang}.json"

    result = model.transcribe(str(VIDEO_PATH), verbose=True)
    with open(TRANSCRIBE_TEXT_PATH, "w") as f:
        f.writelines(result["text"])
    with open(TRANSCRIBE_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main(cfg: Config):
    model = whisper.load_model("large")

    for target in tqdm(cfg.targets):
        target_dir_path = cfg.base_dir / target

        if not os.path.exists(target_dir_path):
            logger.info(f"Video not found: {target_dir_path}")
            continue

        run_transcribe(model, target_dir_path)

    return


def asr_comformer():
    from espnet2.bin.asr_inference import Speech2Text

    model = Speech2Text.from_pretrained(
        "espnet/YushiUeda_iemocap_sentiment_asr_train_asr_conformer"
    )

    speech, rate = soundfile.read("speech.wav")
    text, *_ = model(speech)[0]


def transcribe_reason():
    import reazonspeech as rs
    from espnet2.bin.asr_inference import Speech2Text

    half_number = 1

    target: str = "2014-11-04 - 20-00 Zenit Petersburg 1 - 2 Bayer Leverkusen"
    video_path = Config.base_dir / target / f"{half_number}_224p.oga"


    model = Speech2Text.from_pretrained(
        "espnet/kamo-naoyuki-mini_an4_asr_train_raw_bpe_valid.acc.best"
    )

    cap = rs.transcribe(str(video_path), model)

    # cap is generator
    while True:
        try:
            print(next(cap))
        except StopIteration:
            break


if __name__ == "__main__":
    transcribe_reason()
