import json
from pathlib import Path
import os

import requests as req
from requests_oauthlib import OAuth1
from xml.etree.ElementTree import *


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
    minnano_honyaku: dict = {
        "url": "https://mt-auto-minhon-mlt.ucri.jgn-x.jp/api/mt/generalNT_en_ja/",
        "user_name": "zaemon1251",
        "API_key": "2da3b429e729b05b3521875e89cc8a15062b1252b",
        "API_secret": "8b5c1ba43f243a0577f1ce5c8302198b"
    }


def translate(text):
    url = Config.minnano_honyaku["url"]
    key = Config.minnano_honyaku["API_key"]
    name = Config.minnano_honyaku["user_name"]
    secret = Config.minnano_honyaku["API_secret"]

    consumer = OAuth1(key , secret)

    params = {
        'key': key,
        'name': name,
        'type': 'json',
        'text': text,
    }    # その他のパラメータについては、各APIのリクエストパラメータに従って設定してください。

    try:
        res = req.post(url , data=params , auth=consumer)
        res.encoding = 'utf-8'
        response = json.loads(res.content.decode('utf-8'))
        print(response['resultset'])

        message = response['resultset']['result']['text']
        return message

    except Exception as e:
        print('=== Error ===')
        print('type:' + str(type(e)))
        print('args:' + str(e.args))
        print('e:' + str(e))



def convert_to_vtt(json_data):
    # WebVTT header
    vtt_content = 'WEBVTT\n\n'

    for segment in json_data["segments"]:
        start_time = seconds_to_vtt_time(segment["start"])
        end_time = seconds_to_vtt_time(segment["end"])

        vtt_content += f"{start_time} --> {end_time}\n"

        segment["text"] = segment["text"].strip()

        segment["text"] = translate(segment["text"])

        vtt_content += segment["text"] + "\n"
        vtt_content += "major_class: \n"
        vtt_content += "minor_class: \n"
        vtt_content += "\n\n"

    return vtt_content


def seconds_to_vtt_time(seconds):
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    ms = (s % 1) * 1000
    return f"{int(h):02}:{int(m):02}:{int(s):02}.{int(ms):03}"


if __name__ == "__main__":
    half_number = 1

    print(translate("Hello, world!"))

    # for target in Config.targets:
    #     target: str = target.rstrip("/").split("/")[-1]
    #     json_path = Config.base_dir / target / f"{half_number}_224p.json"
    #     vtt_path = Config.base_dir / target / f"{half_number}_224p.vtt"

    #     with open(json_path, "r") as f:
    #         json_data = json.load(f)

    #     vtt_content = convert_to_vtt(json_data)

    #     with open(vtt_path, "w") as f:
    #         f.write(vtt_content)

