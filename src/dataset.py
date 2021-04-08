import os
import pandas as pd
import numpy as np

import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
from config import Config

import pytorch_lightning as pl
import tokenizer
import logging

logging.basicConfig(level=logging.INFO)


class ImageDataset(Dataset):
    def __init__(self, df, tokenizer, task='train'):
        super(ImageDataset, self).__init__()
        self.df = df
        self.task = task
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.iloc[index].image
        img_path = os.path.join(Config.data_dir, 'images', img_name)
        img = np.array(Image.open(fp=img_path)).transpose(2, 0, 1)

        sample = {"img": th.from_numpy(img / 255.0).float()}

        if self.task == "train":
            label = self.df.iloc[index].image.split(".")[0]
            label = self.tokenizer.encode(text=label)['input_ids']

            sample.update({"label": th.tensor(label, dtype=th.long)})

        return sample


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer,
                 df: pd.DataFrame,
                 val_size=Config.validation_pct,
                 test_bs=Config.test_bs,
                 train_bs=Config.train_bs):
        super(DataModule, self).__init__()
        self.df = df
        self.val_size = val_size
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.tokenizer = tokenizer

    def setup(self):
        self.val_df = self.df.reset_index(drop=True).sample(frac=self.val_size)
        self.train_df = self.df.drop(self.val_df.index)

        self.train_ds = ImageDataset(df=self.train_df,
                                     task='train',
                                     tokenizer=self.tokenizer)

        self.val_ds = ImageDataset(df=self.val_df,
                                   task='train',
                                   tokenizer=self.tokenizer)
        logging.info(
            msg=
            f"Training on {len(self.train_df)} and validating on {len(self.val_df)}"
        )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          batch_size=self.train_bs,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=Config.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          batch_size=self.test_bs,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=Config.num_workers)


if __name__ == "__main__":

    df = pd.read_csv(os.path.join(Config.data_dir, 'Train.csv'))
    tokenizer = tokenizer.Tokenizer()
    dm = DataModule(df=df, tokenizer=tokenizer)
    dm.setup()

    for data in dm.val_dataloader():
        images = data['img']
        labels = data['label']
        print(images.shape)
        print(labels.shape)
