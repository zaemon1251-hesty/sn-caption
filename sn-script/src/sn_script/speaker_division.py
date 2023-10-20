from pyannote.audio import Pipeline
from pyannote.core import Segment
import pandas as pd
from sn_script.config import Config

# パイプラインの初期化
pipeline = Pipeline.from_pretrained("pyannote/segmentation")

# 音声ファイルへのパス
AUDIO_FILE = "your_audio.mp4"
SPEAKER_FILE = "your_speaker.csv"

# パイプラインを使用して音声ファイルの自動話者分類を実行
diarization = pipeline(AUDIO_FILE)

# 結果を格納するためのデータフレームを作成
data = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    data.append({
        'start': segment.start,
        'end': segment.end,
        'speaker': speaker
    })

# データフレームにデータをセット
df = pd.DataFrame(data)

# 結果をCSVファイルに保存
df.to_csv(SPEAKER_FILE, index=False)
