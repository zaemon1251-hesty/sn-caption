import whisper
from pathlib import Path
import json
from tqdm import tqdm
import os
from datetime import datetime
from loguru import logger

try:
    from sn_script.config import Config, half_number
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config, half_number


def run_transcribe(model, video_dir_path: Path):
    VIDEO_DIR = video_dir_path
    VIDEO_PATH = VIDEO_DIR / f"{half_number}_224p.mkv"
    TRANSCRIBE_TEXT_PATH = VIDEO_DIR / f"{half_number}_224p.txt"
    TRANSCRIBE_JSON_PATH = VIDEO_DIR / f"{half_number}_224p.json"

    assert model.is_multilingual

    result = model.transcribe(str(VIDEO_PATH), verbose=True)
    with open(TRANSCRIBE_TEXT_PATH, "w") as f:
        f.writelines(result["text"])
    with open(TRANSCRIBE_JSON_PATH, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    model = whisper.load_model("large")

    for target in tqdm(Config.targets):
        target_dir_path = Config.base_dir / target

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
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.add(
        "logs/llm_anotator_{time}.log".format(time=time_str),
    )
    main()
