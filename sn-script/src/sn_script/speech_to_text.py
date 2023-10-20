import whisper
from pathlib import Path
import json
from tqdm import tqdm
import os
import datetime
import logging
from sn_script.config import Config

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
    with open(TRANSCRIBE_TEXT_PATH, 'w') as f:
        f.writelines(result["text"])
    with open(TRANSCRIBE_JSON_PATH, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main(cfg: Config):
    model = whisper.load_model("large")

    for target in tqdm(cfg.targets):
        video_path = cfg.base_dir / target

        if not os.path.exists(video_path):
            logger.info(f"Video not found: {video_path}")
            continue

        run_transcribe(model, video_path)

    return


if __name__ == "__main__":
    cfg = Config()
    main(cfg)