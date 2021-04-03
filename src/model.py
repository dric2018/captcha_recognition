import os
import sys
import pandas as pd

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import timm
from torchvision import models

from config import Config
from dataset import DataModule

import logging

logging.basicConfig(level=logging.INFO)


class Model(pl.LightningModule):
    def __init__(self,
                 num_layers=Config.num_decoder_layers,
                 dropout=Config.dropout_rate,
                 bidirectional=True,
                 hidden_size=Config.decoder_hidden_size,
                 model_name=Config.base_model):
        super(Model, self).__init__()

        logging.info(msg=f'Using {Config.base_model} as features extractor')
        self.encoder = timm.create_model(model_name=model_name,
                                         pretrained=True,
                                         num_classes=0,
                                         global_pool='')

        input_size = [
            m for m in self.encoder.modules() if isinstance(m, nn.Conv2d)
        ][-1].out_channels

        self.decoder = nn.GRU(input_size=input_size * 3,
                              hidden_size=hidden_size,
                              bidirectional=bidirectional,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=36 + 1)

    def configure_optimizers(self):
        pass

    def forward(self, inputs, targets=None):
        print(f'[Model info] inputs {inputs.size()}')
        features = self.encoder(inputs)
        print(f'[Model info] features {features.size()}')
        n, c, h, w = features.size()
        # reshape features for recurrence
        features = features.view(n, c * h, w).permute(0, 2, 1)
        print(f'[Model info] reshaped features {features.size()}')

        hidden_states, c = self.decoder(features)
        print(f'[Model info] hidden_states {hidden_states.size()}')
        logits = self.fc(hidden_states)
        print(f'[Model info] logits {logits.size()}')
        lengths = th.zeros((n, )).fill_(w)
        print(f'[Model info] lengths {lengths.size()}')

        return logits, lengths, None

    def training_step(self, batch_idx):
        pass

    def validation_step(self, batch_idx):
        pass

    def decode_prediction(self):
        pass


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Config.data_dir, 'Train.csv'))
    dm = DataModule(df=df)
    dm.setup()
    model = Model().cuda()
    for data in dm.val_dataloader():
        images = data['img']
        labels = data['label']
        logits, lens, loss = model(inputs=images.cuda(), targets=None)
        break