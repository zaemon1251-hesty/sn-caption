import whisper
from pathlib import Path
import json
from tqdm import tqdm
import os
import datetime
import logging


class Config:
    base_dir = Path("/raid_elmo/home/lr/moriy")
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

    result = model.transcribe(str(VIDEO_PATH), language=decode_lang, verbose=True)
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