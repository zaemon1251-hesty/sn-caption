from __future__ import annotations
import csv
import json
from pathlib import Path
from .config import Config


# JSONデータをPythonの辞書として読み込む
def write_csv(data: dict | list, output_csv_path: str | Path):
    """CSVファイルに変換"""

    # JSONデータをPythonの辞書として読み込んだ場合、segmentsの中身だけを抽出する
    if isinstance(data, dict):
        data = data["segments"]

    if not isinstance(data, list):
        raise ValueError("data must be list or dict, but got {}".format(type(data)))

    with open(output_csv_path, "w", newline="", encoding="utf_8_sig") as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダを書き込む
        writer.writerow(["id", "start", "end", "text", "大分類", "小分類", "備考"])
        # 各segmentから必要なデータを抽出してCSVに書き込む
        for segment in data:
            writer.writerow(
                [
                    segment["id"],
                    seconds_to_gametime(segment["start"]),
                    seconds_to_gametime(segment["end"]),
                    segment["text"],
                    "",
                    "",
                    "",
                ]
            )
    print(f"CSVファイルが生成されました。:{output_csv_path}")


def seconds_to_gametime(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02}:{int(s):02}"


def main():
    half_number = 1
    for target in Config.targets:
        target: str = target.rstrip("/").split("/")[-1]
        json_path = Config.base_dir / target / f"{half_number}_224p.json"
        csv_path = Config.base_dir / target / f"{half_number}_224p.csv"
        with open(json_path, "r") as f:
            json_data = json.load(f)
        write_csv(json_data, csv_path)


if __name__ == "__main__":
    main()
