from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from SoccerNet.utils import getListGames
from pathlib import Path
import os

try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config

os.environ.setdefault("SOCCERNET_PASSWORD", "s0cc3rn3t")
os.environ.setdefault("SOCCERNET_LOCAL_DIRECTORY", str(Config.base_dir.absolute()))

PASSWORD = os.environ.get("SOCCERNET_PASSWORD")
LOCAL_DIRECTORY = os.environ.get(
    "SOCCERNET_LOCAL_DIRECTORY"
)  # LOCAL_DIRECTORY = "/path/to/SoccerNet"


def main():
    mySNdl = SNdl(LocalDirectory=LOCAL_DIRECTORY)
    mySNdl.password = PASSWORD
    game_list = getListGames("all")
    target_games = []
    for target in Config.targets:
        target_games.append(target.strip().rstrip("/"))
    target_ids = [i for i, game in enumerate(game_list) if game in target_games]

    print(f"Target ids are {target_ids}.")
    for target_id in target_ids:
        mySNdl.LocalDirectory = LOCAL_DIRECTORY
        print(f"Downloading {game_list[target_id]}...")
        print(game_list[target_id] in getListGames("all"))
        mySNdl.downloadGameIndex(
            target_id,
            files=[
                "1_720p.mkv",
                "2_720p.mkv",
                "Labels-caption.json",
                "Labels.json",
                "Labels-v2.json",
            ],
            verbose=1,
        )


if __name__ == "__main__":
    print(LOCAL_DIRECTORY, "is used.")
    print(f"Target games are {Config.targets}.")
    main()

    print("Done!")
