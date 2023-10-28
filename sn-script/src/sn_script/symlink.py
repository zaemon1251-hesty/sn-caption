from pathlib import Path
import os

src_dir = Path("/Users/heste/Downloads/SoccerNet/")
dst_dir = Path("/Users/heste/workspace/sn-caption") / "data"

first_game_suffix = "1_720p.mkv"
second_game_suffix = "2_720p.mkv"

target_games = []
with open(Path(__file__).parent.parent / "data" / "video_files.txt", "r") as f:
    for line in f:
        target_games.append(line.strip().rstrip("/"))

for target_game in target_games:
    src_video_file_path1 = src_dir / target_game / first_game_suffix
    src_video_file_path2 = src_dir / target_game / second_game_suffix

    dst_game_name = target_game.split("/")[-1]
    dst_video_path_1 = dst_dir / dst_game_name / first_game_suffix
    dst_video_path_2 = dst_dir / dst_game_name / second_game_suffix

    # create symlink
    try:
        os.symlink(src_video_file_path1, dst_video_path_1)
    except FileExistsError:
        pass
    try:
        os.symlink(src_video_file_path2, dst_video_path_2, target_is_directory=True)
    except FileExistsError:
        pass
