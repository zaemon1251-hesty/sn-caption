import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from SoccerNet.Evaluation.utils import AverageMeter
import os
from dataset import (
    collate_fn_padd,
    SoccerNetClips,
    SoccerNetCaptions,
    SoccerNetClipsTesting,
    PredictionCaptions,
    SoccerNetVideoProcessor,
    SoccerNetTextProcessor
)


class SoccerNetClipsDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 features="ResNET_PCA512.npy",
                 split=["train"],
                 version=2,
                 framerate=2,
                 window_size=15,
                 batch_size=32,
                 num_workers=4) -> None:
        super(SoccerNetClipsDataModule, self).__init__()
        self.path = path
        self.features = features
        self.split = split
        self.version = version
        self.framerate = framerate
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.soccernet_clips_train = SoccerNetClips(self.path, self.features, self.split, self.version, self.framerate, self.window_size)
        self.soccernet_clips_val = SoccerNetClips(self.path, self.features, self.split, self.version, self.framerate, self.window_size)
        self.soccernet_clips_test = SoccerNetClipsTesting(self.path, self.features, self.split, self.version, self.framerate, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.soccernet_clips_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn_padd)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.soccernet_clips_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_padd)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.soccernet_clips_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_padd)


class SoccerNetCaptionsDataModule(pl.LightningDataModule):
    def __init__(self,
                 path,
                 features="ResNET_PCA512.npy",
                 split=['train'],
                 version=2,
                 framerate=2,
                 window_size=15,
                 batch_size=15,
                 num_workers=4) -> None:
        super(SoccerNetCaptionsDataModule).__init__()
        self.path = path
        self.features = features
        self.split = split
        self.version = version
        self.framerate = framerate
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.soccernet_captions_train = SoccerNetCaptions(self.path, self.features, self.split, self.version, self.framerate, self.window_size)
        self.soccernet_captions_val = SoccerNetCaptions(self.path, self.features, self.split, self.version, self.framerate, self.window_size)
        self.soccernet_captions_test = SoccerNetCaptions(self.path, self.features, self.split, self.version, self.framerate, self.window_size)

    def prepare_data(self) -> None:
        """tokenize the captions and save the vocabulary in just one process.

        Returns:
            None
        """
        self.text_processor = SoccerNetTextProcessor(self.path, self.split, self.version)
        return super().prepare_data()

    def train_dataloader(self):
        return DataLoader(self.soccernet_captions_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn_padd)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.soccernet_captions_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_padd)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.soccernet_captions_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn_padd)

