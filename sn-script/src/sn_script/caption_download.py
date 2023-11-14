from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from SoccerNet.utils import getListGames
from pathlib import Path
import os

PASSWORD = os.environ.get("SOCCERNET_PASSWORD")
LOCAL_DIRECTORY = os.environ.get(
    "SOCCERNET_LOCAL_DIRECTORY"
)  # LOCAL_DIRECTORY = "/path/to/SoccerNet"
TARGET_VIDEO_FILES = os.environ.get("SOCCERNET_TARGET_VIDEO_FILES")


def main():
    mySNdl = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    mySNdl.password = PASSWORD
    game_list = getListGames("all")
    target_games = []
    with open(TARGET_VIDEO_FILES, "r") as f:
        for line in f:
            target_games.append(line.strip().rstrip("/"))
    target_ids = [i for i, game in enumerate(game_list) if game in target_games]

    for target_id in target_ids:
        mySNdl.downloadGameIndex(
            target_id,
            files=["1_720p.mkv", "2_720p.mkv", "Labels-caption.json", "Labels.json"],
            verbose=1,
        )


if __name__ == "__main__":
    main()
