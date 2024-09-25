from __future__ import annotations
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import logging
import json

from collections import Counter
from torchtext.vocab import vocab

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import getMetaDataTask
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


def collate_fn_padd(batch):
    """
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    captions = [t[-1] for t in batch]
    idx = [t[-3:-1] for t in batch]
    # padd
    tokens = [
        ([SOS_TOKEN] + t[-4] + [EOS_TOKEN]) if t[-4] else [PAD_TOKEN, PAD_TOKEN]
        for t in batch
    ]
    tokens = [torch.Tensor(t).long() for t in tokens]
    # get sequence lengths
    lengths = torch.tensor([len(t) for t in tokens])
    tokens = pad_sequence(tokens, batch_first=True)
    # compute mask
    mask = tokens != PAD_TOKEN
    return (
        default_collate([t[:-4] for t in batch]) + [tokens],
        lengths,
        mask,
        captions,
        idx,
    )


def collate_fn_padd_spotting(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    # clipデータをバッチごとにパディング
    # ラベルのpaddingは0(発話なし)で行う
    # フィーチャーとラベルをそれぞれまとめる
    feats, labels = zip(*batch)

    # バッチごとにフィーチャーとラベルをパディング
    feats_padded = pad_sequence(feats, batch_first=True, padding_value=PAD_TOKEN)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return feats_padded, labels_padded


def feats2clip(feats, stride, clip_length, padding="replicate_last", off=0):
    if padding == "zeropad":
        pad = feats.shape[0] - int(feats.shape[0] / stride) * stride
        m = torch.nn.ZeroPad2d((0, 0, clip_length - pad, 0))
        feats = m(feats)

    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length - off):
        idxs.append(idx + i)
    idxs = torch.stack(idxs, dim=1)

    if padding == "replicate_last":
        idxs = idxs.clamp(0, feats.shape[0] - 1)
    return feats[idxs, ...]


class SoccerNetClips(Dataset):
    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split=["train"],
        version=2,
        framerate=2,
        window_size=15,
    ):
        self.path = path
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version
        self.framerate = framerate
        self.listGames = getListGames(split, task="caption")

        labels, num_classes, dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", version
        )
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

        self.game_feats = []
        self.game_labels = []

        logging.info(
            f"SoccerNetClips Dataset of ({split}) loading features and labels into system memory"
        )
        for game in tqdm(self.listGames):
            feat_half1, label_half1 = self.load_features_and_labels(game, half=1)
            feat_half2, label_half2 = self.load_features_and_labels(game, half=2)

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)

    def __len__(self):
        # len(self.game_feats) = num_games * 2(first half and second half) * sum(num_frames//self.window_size for each num_frames in game)
        return len(self.game_feats)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index (unit: spot index index = 0, 1, 2, ..., len(self.game_feats))

        Returns:
            Tensor (num_frame_list[=num_frames//self.window_size], self.window_size_frame, feature dim): features
            Tensor (self.num_classes + 1, ): labels
        """
        return self.game_feats[index, :, :], self.game_labels[index, :]

    def load_features_and_labels(self, game, half):
        feature_path = os.path.join(self.path, game, f"{half}_{self.features}")
        features = np.load(feature_path)
        features = features.reshape(-1, features.shape[-1])

        features = feats2clip(
            torch.from_numpy(features),
            stride=self.window_size_frame,
            off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        label_path = os.path.join(self.path, game, self.labels)

        assert os.path.exists(label_path), f"{label_path} does not exist."

        # class 数 + 1 (no utterance)
        labels = np.zeros((features.shape[0], self.num_classes + 1))

        # No utterance labelの value は 1 で初期化
        labels[:, 0] = 1

        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]

            for annotation in annotations:
                time = annotation["gameTime"]
                event = annotation["label"]

                if event not in self.dict_event:
                    continue

                label_half = int(time[0])

                minutes, seconds = time.split(" ")[-1].split(":")
                minutes, seconds = int(minutes), int(seconds)
                frame = self.framerate * (seconds + 60 * minutes)

                # フレームの束のインデックス
                frame_list_idx = frame // self.window_size_frame

                # if label outside temporal of view
                if label_half != half or frame_list_idx >= labels.shape[0]:
                    continue

                # frameがfeaturesの長さを超える場合は最後のフレームにする
                frame_list_idx = min(frame_list_idx, features.shape[0] - 1)

                label = self.dict_event[event]

                value = 1

                # no utterance labelの value を 0 にする
                labels[frame_list_idx][0] = 0

                # フレームの束に対してlabelを代入
                labels[frame_list_idx][label + 1] = value

        return features, labels


class SoccerNetClipsTesting(Dataset):
    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split=["test"],
        version=2,
        framerate=2,
        window_size=15,
    ):
        self.path = path
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.version = version
        self.split = split
        self.listGames = getListGames(split, task="caption")

        labels, num_classes, dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", version
        )
        self.labels = labels
        self.num_classes = num_classes
        self.dict_event = dict_event

    def __len__(self):
        return len(self.listGames)

    def __getitem__(self, index):
        game = self.listGames[index]
        feat_half1, label_half1 = self.load_features_and_labels_test(game, half=1)
        feat_half2, label_half2 = self.load_features_and_labels_test(game, half=2)

        return game, feat_half1, feat_half2, label_half1, label_half2

    def load_features_and_labels_test(self, game, half):
        feature_path = os.path.join(self.path, game, f"{half}_{self.features}")
        features = np.load(feature_path)
        features = features.reshape(-1, features.shape[-1])

        features = feats2clip(
            torch.from_numpy(features),
            stride=1,
            off=int(self.window_size_frame / 2),
            clip_length=self.window_size_frame,
        )

        label_path = os.path.join(self.path, game, self.labels)

        assert os.path.exists(label_path), f"{label_path} does not exist."

        labels = np.zeros((features.shape[0], self.num_classes))

        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
            for annotation in annotations:
                time = annotation["gameTime"]
                event = annotation["label"]

                half_annotation = int(time[0])
                if (
                    event not in self.dict_event or half_annotation != half
                ):  # TODO 意味がわからない
                    continue

                minutes, seconds = map(int, time.split(" ")[-1].split(":"))
                frame = self.framerate * (seconds + 60 * minutes)
                frame = min(frame, features.shape[0] - 1)
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                # valueを代入する場所、training時は label+1 だが、test時はlabelにする？
                # test時は全フレームに対して予測するので、frame_list_idxは使わない
                labels[frame][label] = value

        return features, labels


class SoccerNetCaptions(Dataset):
    """
    This class is used to download and pre-compute clips and captions from the SoccerNet dataset for captining training phase.
    """

    def __init__(
        self,
        path,
        features="ResNET_TF2_PCA512.npy",
        split=["train"],
        version=2,
        framerate=2,
        window_size=15,
    ):
        self.path = path

        if not isinstance(split, list):
            split = [s for s in split]

        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version
        self.labels, self.num_classes, self.dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", version
        )

        self.listGames = self.listGames
        self.data = list()
        self.game_feats = list()

        for game_id, game in enumerate(tqdm(self.listGames)):
            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            for caption_id, annotation in enumerate(labels["annotations"]):
                time = annotation["gameTime"]
                event = annotation["label"]
                half = int(time[0])
                if event not in self.dict_event or half > 2:
                    continue

                minutes, seconds = time.split(" ")[-1].split(":")
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (seconds + 60 * minutes)

                self.data.append(
                    (
                        (game_id, half - 1, frame),
                        (caption_id, annotation["anonymized"]),
                    )
                )

        # launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        # launch a TextProcessor that will tokenize a caption
        self.text_processor = SoccerNetTextProcessor(self.getCorpus(split=split))
        self.vocab_size = len(self.text_processor.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            caption_tokens (np.array): tokens of captions.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
            caption (List[strings]): list of original captions.
        """
        clip_id, (caption_id, caption) = self.data[idx]
        game_id = clip_id[0]
        game = self.listGames[game_id]

        l_pad = self.window_size_frame // 2 + self.window_size_frame % 2
        r_pad = self.window_size_frame // 2
        feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        feat_half1 = np.pad(
            feat_half1.reshape(-1, feat_half1.shape[-1]),
            ((l_pad, r_pad), (0, 0)),
            "edge",
        )
        feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
        feat_half2 = np.pad(
            feat_half2.reshape(-1, feat_half2.shape[-1]),
            ((l_pad, r_pad), (0, 0)),
            "edge",
        )

        vfeats = self.video_processor(clip_id, {game_id: (feat_half1, feat_half2)})
        caption_tokens = self.text_processor(caption)

        return vfeats, caption_tokens, game_id, caption_id, caption

    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [
            annotation["anonymized"]
            for game in getListGames(split, task="caption")
            for annotation in json.load(
                open(os.path.join(self.path, game, self.labels))
            )["annotations"]
        ]
        return corpus

    def detokenize(self, tokens, remove_EOS=True):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return (
            string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}")
            if remove_EOS
            else string
        )


class SoccerNetVideoProcessor(object):
    """video_fn is a tuple of (video_id, half, frame)."""

    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, video_fn, feats):
        video_id, half, frame = video_fn
        video_feature = feats[video_id][half]
        # make sure that the clip lenght is right
        start = min(frame, video_feature.shape[0] - self.clip_length)
        video_feature = video_feature[start : start + self.clip_length]

        return video_feature


class SoccerNetTextProcessor(object):
    """
    A generic Text processor
    tokenize a string of text on-the-fly.
    """

    def __init__(self, corpus, min_freq=5):
        import spacy

        spacy_token = spacy.load("en_core_web_sm").tokenizer
        # Add special case rule
        spacy_token.add_special_case("[PLAYER]", [{"ORTH": "[PLAYER]"}])
        spacy_token.add_special_case("[COACH]", [{"ORTH": "[COACH]"}])
        spacy_token.add_special_case("[TEAM]", [{"ORTH": "[TEAM]"}])
        spacy_token.add_special_case("([TEAM])", [{"ORTH": "([TEAM])"}])
        spacy_token.add_special_case("[REFEREE]", [{"ORTH": "[REFEREE]"}])
        self.tokenizer = lambda s: [c.text for c in spacy_token(s)]
        self.min_freq = min_freq
        self.build_vocab(corpus)

    def build_vocab(self, corpus):
        counter = Counter([token for c in corpus for token in self.tokenizer(c)])
        voc = vocab(
            counter,
            min_freq=self.min_freq,
            specials=["[PAD]", "[SOS]", "[EOS]", "[UNK]", "[MASK]", "[CLS]"],
        )
        voc.set_default_index(voc["[UNK]"])
        self.vocab = voc

    def __call__(self, text):
        return self.vocab(self.tokenizer(text))

    def detokenize(self, tokens):
        return " ".join(self.vocab.lookup_tokens(tokens))


class PredictionCaptions(Dataset):
    def __init__(
        self,
        SoccerNetPath,
        PredictionPath,
        features="ResNET_TF2_PCA512.npy",
        split=["train"],
        version=2,
        framerate=2,
        window_size=15,
    ):
        self.path = SoccerNetPath
        self.PredictionPath = PredictionPath
        self.listGames = getListGames(split, task="caption")
        self.features = features
        self.window_size_frame = window_size * framerate
        self.version = version
        self.labels, _, self.dict_event, _ = getMetaDataTask(
            "caption", "SoccerNet", version
        )
        self.split = split
        self.listGames = self.listGames

        self.data = list()
        self.game_feats = list()

        for game_id, game in enumerate(tqdm(self.listGames)):
            # Load labels
            preds = json.load(
                open(os.path.join(self.PredictionPath, game, "results_spotting.json"))
            )

            for caption_id, annotation in enumerate(preds["predictions"]):

                if annotation["label"] not in self.dict_event:
                    continue

                time = annotation["gameTime"]
                half = int(time[0])
                if half > 2:
                    continue

                minutes, seconds = time.split(" ")[-1].split(":")
                minutes, seconds = int(minutes), int(seconds)
                frame = framerate * (int(seconds) + 60 * int(minutes))

                self.data.append(((game_id, half - 1, frame), caption_id))

        # launch a VideoProcessor that will create a clip around a caption
        self.video_processor = SoccerNetVideoProcessor(self.window_size_frame)
        # launch a TextProcessor that will tokenize a caption
        self.text_processor = None
        self.vocab_size = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            vfeats (np.array): clip of features.
            clip_id (np.array): clip id.
            caption_id (np.array): caption id.
        """
        # clip_id, (caption_id, _) = self.data[idx]
        clip_id, caption_id = self.data[idx]
        game_id = clip_id[0]
        game = self.listGames[game_id]

        l_pad = self.window_size_frame // 2 + self.window_size_frame % 2
        r_pad = self.window_size_frame // 2
        feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        feat_half1 = np.pad(
            feat_half1.reshape(-1, feat_half1.shape[-1]),
            ((l_pad, r_pad), (0, 0)),
            "edge",
        )
        feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
        feat_half2 = np.pad(
            feat_half2.reshape(-1, feat_half2.shape[-1]),
            ((l_pad, r_pad), (0, 0)),
            "edge",
        )

        vfeats = self.video_processor(clip_id, {game_id: (feat_half1, feat_half2)})
        return vfeats, game_id, caption_id

    def detokenize(self, tokens, remove_EOS=True):
        """
        Args:
            tokens (List[int]): tokens of caption
        Returns:
            caption (string): string obtained after replacing each token by its corresponding word
        """
        string = self.text_processor.detokenize(tokens)
        return (
            string.rstrip(f" {self.text_processor.vocab.lookup_token(EOS_TOKEN)}")
            if remove_EOS
            else string
        )

    def getCorpus(self, split=["train"]):
        """
        Args:
            split (string): split of dataset
        Returns:
            corpus (List[string]): vocabulary build from split.
        """
        corpus = [
            annotation["anonymized"]
            for game in getListGames(split, task="caption")
            for annotation in json.load(
                open(os.path.join(self.path, game, self.labels))
            )["annotations"]
        ]
        return corpus


class CommentaryClips(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting training phase.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split="train",
        framerate=1,
        window_size=15,
    ):
        assert isinstance(split, str), "split should be a string"
        assert split in ["train", "valid"], "split should be either 'train' or 'valid'"

        self.framerate = framerate
        self.path = path
        self.features = features
        self.window_size_frame = window_size * framerate

        label_template = "commentary_dataset/gpt-3.5-turbo-1106_500game_{half_nunber}_llm_annotation_{split}.csv"

        self.label_df_half1 = pd.read_csv(
            os.path.join(self.path, label_template.format(half_nunber=1, split=split))
        )
        self.label_df_half2 = pd.read_csv(
            os.path.join(self.path, label_template.format(half_nunber=2, split=split))
        )

        # TODO データ準備の時点で対処すべき
        self.label_df_half1 = self.label_df_half1.dropna(subset=["target_label"])
        self.label_df_half2 = self.label_df_half2.dropna(subset=["target_label"])

        # TODO データ準備の時点で対処すべき
        games_half1 = self.label_df_half1["game"].unique().tolist()
        games_half2 = self.label_df_half2["game"].unique().tolist()
        list_games = list(set(games_half1) & set(games_half2))

        self.listGames = list_games

        self.num_classes = 2  # 映像の説明, 付加的情報

        logging.info("Pre-compute clips")

        self.game_feats = list()
        self.game_labels = list()

        for game in tqdm(self.listGames):
            # filter labels by game
            label_df_half1_by_game: pd.DataFrame = self.label_df_half1[
                self.label_df_half1["game"] == game
            ].sort_values("target_frameid")
            label_df_half2_by_game: pd.DataFrame = self.label_df_half2[
                self.label_df_half2["game"] == game
            ].sort_values("target_frameid")

            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(
                torch.from_numpy(feat_half1),
                stride=self.window_size_frame,
                clip_length=self.window_size_frame,
            )
            feat_half2 = feats2clip(
                torch.from_numpy(feat_half2),
                stride=self.window_size_frame,
                clip_length=self.window_size_frame,
            )

            # Load labels
            # self.num_classes + 1 = クラス数 + 1 (Back Ground -> BG)
            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes + 1))
            label_half1[:, 0] = 1  # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes + 1))
            label_half2[:, 0] = 1  # those are BG classes

            label_half1 = self.preprocess_label_array(
                label_half1, label_df_half1_by_game
            )
            label_half2 = self.preprocess_label_array(
                label_half2, label_df_half2_by_game
            )

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)

    def preprocess_label_array(self, label_array: np.ndarray, label_df: pd.DataFrame):
        # frame_list_idx = target_frameid // self.window_size_frame でクリップのインデックスを取得
        # target_label = 1,2
        # label_half_x[frame_list_idx][target_label] = 1 でラベルを付与 (BGを0にしておく)
        label_length = label_array.shape[0]

        for _, row in label_df.iterrows():
            target_frameid = row["target_frameid"]
            target_label = int(row["target_label"])

            minute, seconds = target_frameid // 60, target_frameid % 60

            frame = self.framerate * (seconds + 60 * minute)

            frame_list_idx = self.framerate * (frame // self.window_size_frame)

            if frame_list_idx >= label_length:
                continue

            # BGを0にしておく
            label_array[frame_list_idx][0] = 0
            # that's my class
            label_array[frame_list_idx][target_label] = 1

        return label_array

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            clip_feat (np.array): clip of features.
            clip_labels (np.array): clip of labels for the segmentation.
            clip_targets (np.array): clip of targets for the spotting.
        """
        return self.game_feats[index, :, :], self.game_labels[index, :]

    def __len__(self):
        return len(self.game_feats)


class CommentaryClipsTesting(Dataset):
    """
    This class is used to download and pre-compute clips from the SoccerNet dataset for spotting inference phase.
    """

    def __init__(
        self,
        path,
        features="ResNET_PCA512.npy",
        split="test",
        framerate=2,
        window_size=15,
    ):
        assert framerate == 1, "framerate should be 1 for commentary-spotting task"
        assert isinstance(split, str), "split should be a string"
        assert split in [
            "test",
        ], "split should be 'test'"

        # test時に参照されるから、便宜上用意する
        self.version = 2

        self.path = path
        self.features = features
        self.window_size_frame = window_size * framerate
        self.split = split

        label_template = "commentary_dataset/gpt-3.5-turbo-1106_500game_{half_nunber}_llm_annotation_{split}.csv"

        self.label_df_half1 = pd.read_csv(
            os.path.join(self.path, label_template.format(half_nunber=1, split=split))
        )
        self.label_df_half2 = pd.read_csv(
            os.path.join(self.path, label_template.format(half_nunber=2, split=split))
        )

        # TODO データ準備の時点で対処すべき
        self.label_df_half1 = self.label_df_half1.dropna(subset=["target_label"])
        self.label_df_half2 = self.label_df_half2.dropna(subset=["target_label"])

        # TODO データ準備の時点で対処すべき
        games_half1 = self.label_df_half1["game"].unique().tolist()
        games_half2 = self.label_df_half2["game"].unique().tolist()
        list_games = list(set(games_half1) & set(games_half2))

        self.listGames = list_games  # Config.targets

        self.num_classes = 2  # 映像の説明, 付加的情報

    def __getitem__(self, index):
        """
        Args:
            index (int): Index (unit: one game)
        Returns:
            game (str): game name.
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        game = self.listGames[index]

        # filter labels by game
        label_df_half1_by_game: pd.DataFrame = self.label_df_half1[
            self.label_df_half1["game"] == game
        ].sort_values("target_frameid")
        label_df_half2_by_game: pd.DataFrame = self.label_df_half2[
            self.label_df_half2["game"] == game
        ].sort_values("target_frameid")

        # Load features
        feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))

        # Load labels
        # test時はBGを考慮しない
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        label_half1 = self.preprocess_label_array_test(
            label_half1, label_df_half1_by_game
        )
        label_half2 = self.preprocess_label_array_test(
            label_half2, label_df_half2_by_game
        )

        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=1,
            clip_length=self.window_size_frame,
        )
        feat_half2 = feats2clip(
            torch.from_numpy(feat_half2),
            stride=1,
            clip_length=self.window_size_frame,
        )

        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def preprocess_label_array_test(
        self, label_array: np.ndarray, label_df: pd.DataFrame
    ):
        # target_label = 1,2 -> 0,1 に変換 (mAPを算出する時、BGを考慮しないため)
        # label_half_x[frame_list_idx][target_label-1] = 1 でラベルを付与
        label_length = label_array.shape[0]

        for _, row in label_df.iterrows():
            target_frameid = row["target_frameid"]
            target_label = int(row["target_label"])

            if target_frameid >= label_length:
                continue

            # BGを0にしておく
            label_array[target_frameid][0] = 0
            # that's my class
            label_array[target_frameid][target_label - 1] = 1

        return label_array

    def __len__(self):
        return len(self.listGames)


class CommentaryClipsForDiffEstimation(Dataset):
    """
    直前の発話終了(開始)時間を考慮したデータセット
    """

    def __init__(
        self,
        path,
        split="train",
    ):
        assert isinstance(split, str), "split should be a string"
        assert split in [
            "train",
            "valid",
            "test",
        ], "split should be either 'train' or 'valid'"

        self.path = path

        label_template = "commentary_dataset/gpt-3.5-turbo-1106_500game_{half_nunber}_llm_annotation_{split}.csv"

        self.label_df_half1 = pd.read_csv(
            os.path.join(self.path, label_template.format(half_nunber=1, split=split))
        )
        self.label_df_half2 = pd.read_csv(
            os.path.join(self.path, label_template.format(half_nunber=2, split=split))
        )

        # TODO データ準備の時点で対処すべき
        self.label_df_half1 = self.label_df_half1.dropna(subset=["target_label"])
        self.label_df_half2 = self.label_df_half2.dropna(subset=["target_label"])

        # TODO データ準備の時点で対処すべき
        games_half1 = self.label_df_half1["game"].unique().tolist()
        games_half2 = self.label_df_half2["game"].unique().tolist()
        list_games = list(set(games_half1) & set(games_half2))

        self.listGames = list_games

        self.num_classes = 2  # 映像の説明, 付加的情報

        logging.info("Pre-compute clips")

        self.data = []

        for game in tqdm(self.listGames):
            # filter labels by game
            label_df_half1_by_game: pd.DataFrame = self.label_df_half1[
                self.label_df_half1["game"] == game
            ].sort_values("target_frameid")
            label_df_half2_by_game: pd.DataFrame = self.label_df_half2[
                self.label_df_half2["game"] == game
            ].sort_values("target_frameid")

            # 直前の発話開始フレームを self.prev_dataに入れて
            # 現在の発話開始フレーム, class を self.current_dataに入れる
            for i, row in enumerate(label_df_half1_by_game.itertuples()):
                # 最初の行は無視
                if i == 0:
                    continue
                previous_frameid = label_df_half1_by_game.iloc[i - 1]["target_frameid"]
                target_frameid = row.target_frameid
                target_label = row.target_label

                self.data.append((game, previous_frameid, target_frameid, target_label))

            for i, row in enumerate(label_df_half2_by_game.itertuples()):
                # 最初の行は無視
                if i == 0:
                    continue
                previous_frameid = label_df_half2_by_game.iloc[i - 1]["target_frameid"]
                target_frameid = row.target_frameid
                target_label = row.target_label

                self.data.append((game, previous_frameid, target_frameid, target_label))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            end_time[index-1] (int): 直前の発話終了時間
            start_time[index] (int): 現在の発話開始時間
        """
        # self.data[index] = [game_name, previous_frameid, target_frameid, target_label]
        # game_nameは使わない
        return self.data[index][1:]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    root = "/Users/heste/workspace/soccernet/sn-script"

    # dataset_Train = SoccerNetCaptions(
    #     path=root,
    #     features="baidu_soccer_embeddings.npy",
    #     split=["train"],
    #     version=2,
    #     framerate=2,
    #     window_size=15,
    # )
    # print(f"{dataset_Train.vocab_size=}")

    # dataset_Test = SoccerNetCaptions(
    #     path=root,
    #     features="baidu_soccer_embeddings.npy",
    #     split=["test"],
    #     version=2,
    #     framerate=2,
    #     window_size=15,
    # )
    # print(f"{dataset_Test.vocab_size=}")
    # test_loader = torch.utils.data.DataLoader(
    #     dataset_Test,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     collate_fn=collate_fn_padd,
    # )
    # batch = next(iter(test_loader))
    # (feats, caption), lengths, mask, caption_or, idx = batch
    # print(feats, caption)
    # print(test_loader.dataset.detokenize([55, 22, 33, 2]))
    # print(idx)
    dataset_Test = CommentaryClipsForDiffEstimation(
        path=root,
        split="test",
    )

    def predict_diff_and_label(previous_frameid):
        mean_silence_sec = 4.9
        fps = 1
        label_space = [1, 2]
        label_prob = [0.87, 0.13]

        next_frameid = previous_frameid + int(mean_silence_sec * fps)
        next_label = np.random.choice(label_space, p=label_prob)
        return (next_frameid, next_label)

    for i, (previous_frameid, target_frameid, target_label) in enumerate(dataset_Test):
        next_frameid, label = predict_diff_and_label(previous_frameid)
        print(
            f"previous_frameid: {previous_frameid}, target_frameid: {target_frameid}, target_label: {target_label}, next_frameid: {next_frameid}, label: {label}"
        )
        if i == 10:
            break
