import json
from pathlib import Path
import os

import requests as req
from requests_oauthlib import OAuth1
from xml.etree.ElementTree import *


try:
    from sn_script.config import Config
except ModuleNotFoundError:
    import sys

    sys.path.append(".")
    from src.sn_script.config import Config


def translate(text):
    url = Config.minnano_honyaku["url"]
    key = Config.minnano_honyaku["API_key"]
    name = Config.minnano_honyaku["user_name"]
    secret = Config.minnano_honyaku["API_secret"]

    consumer = OAuth1(key, secret)

    params = {
        "key": key,
        "name": name,
        "type": "json",
        "text": text,
    }  # その他のパラメータについては、各APIのリクエストパラメータに従って設定してください。

    try:
        res = req.post(url, data=params, auth=consumer)
        res.encoding = "utf-8"
        response = json.loads(res.content.decode("utf-8"))
        print(response["resultset"])

        message = response["resultset"]["result"]["text"]
        return message

    except Exception as e:
        print("=== Error ===")
        print("type:" + str(type(e)))
        print("args:" + str(e.args))
        print("e:" + str(e))


def convert_to_vtt(json_data):
    # WebVTT header
    vtt_content = "WEBVTT\n\n"

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
    m, s = divmod(seconds, 60)
    return f"{int(m):02}:{int(s):02}"


if __name__ == "__main__":
    half_number = 1

    print(translate("Hello, world!"))
