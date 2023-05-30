from multiprocessing.sharedctypes import Value
from pathlib import Path
import os
import copy
import torch
from typing import Optional

import numpy as np
import tqdm
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split


class FiveKDataset(Dataset):
    def __init__(self, dnr_C_dir, mode="train", transform=None):
        """MIT FiveK Dataset
        dnr_C_dir: path to directory "C" as downloaded from https://www.dropbox.com/sh/web5of2dswd55b3/AABs5xY3V1CXEzfGWzBw9OUQa?dl=0&preview=C.zip
        mode: 'train' or 'test' (made using random250.txt split)
        limit_to: limit the number of images in the set
        transform: 'resize' or None, resize to (256, 256)
        """
        assert mode in [
            "train",
            "test",
        ], f"mode should be 'train' or 'test' but is {mode}"
        self.data_Path = Path(dnr_C_dir)
        self.mode = mode
        self.img_names = sorted(os.listdir(self.data_Path / mode / "raw"))
        self.transform = transform
        if self.transform and mode == "test":
            print(
                "Be careful, you are using a transform over the test dataset"
            )
        self.batch_compatible = self.transform == "resize"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        raw_img_path = self.data_Path / self.mode / "raw" / self.img_names[idx]
        target_img_path = (
            self.data_Path / self.mode / "target" / self.img_names[idx]
        )
        raw, target = Image.open(raw_img_path), Image.open(target_img_path)
        if (
            self.transform == "resize"
        ):  # use NEAREST because of speed, use 256 because it captures both colors and semantics
            raw, target = raw.resize(
                (256, 256), resample=Image.Resampling.NEAREST
            ), target.resize((256, 256), resample=Image.Resampling.NEAREST)
        return self.minmaxnorm(torch.from_numpy(np.array(raw))).permute(
            2, 0, 1
        ), self.minmaxnorm(torch.from_numpy(np.array(target)).permute(2, 0, 1))

    def minmaxnorm(self, x):
        """Normalize between min and max to get x in [0, 1]"""
        return (x - x.min()) / (x.max() - x.min())


class FiveKDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule holding dataloaders for each mode.
    Same parameters than in dataset.
    Dataloading with multiprocessing if `dl_fast==True`.
    If not resizing, batch size should be 1 (otherwise we break default_collate)
    Test set is not resized.
    Train data are the first 4500 images in the "train" folder.
    Val data are the last 250 images in the "train" folder (not resized).
    """

    def __init__(
        self,
        dnr_C_dir: str = "/home/franchesoni/data/mit5k/dnr/C",
        batch_size: int = 32,
        transform=None,
        dl_fast=False,
    ):
        super().__init__()
        self.data_dir = dnr_C_dir
        self.batch_size = batch_size
        self.transform = transform
        assert transform == "resize" or batch_size == 1
        self.train_dl_kwargs = (
            dict(
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=16,
                persistent_workers=True,
            )
            if dl_fast
            else {"shuffle": True}
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = FiveKDataset(
            self.data_dir, "train", self.transform
        )
        self.train_dataset.img_names = self.train_dataset.img_names[:4500]
        self.val_dataset = FiveKDataset(self.data_dir, "train", None)
        self.val_dataset.img_names = self.val_dataset.img_names[4500:]
        self.test_dataset = FiveKDataset(self.data_dir, "test", None)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            **self.train_dl_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)


def find_smallest_sides():
    data_dir = "/home/franchesoni/data/mit5k/dnr/C"
    batch_size = 10
    dataset = FiveKDataset(data_dir, mode="train", limit_to=None)
    assert dataset.batch_compatible or batch_size == 1
    dataloader = DataLoader(dataset, batch_size=batch_size)
    hmin, wmin = 1e6, 1e6
    for raw, target in tqdm.tqdm(dataloader):
        assert raw.size() == target.size()
        h, w = raw.size()[1:3]
        if h < hmin:
            hmin = h
        if w < wmin:
            wmin = w
    print(f"hmin = {hmin}, wmin = {wmin}")


if __name__ == "__main__":
    find_smallest_sides()
