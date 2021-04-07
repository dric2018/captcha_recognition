import os
import sys
import pandas as pd

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import timm
from torchvision import models, transforms

from ctcdecode import CTCBeamDecoder

from config import Config
from dataset import DataModule
import utils
import tokenizer
import logging

logging.basicConfig(level=logging.INFO)


class Model(pl.LightningModule):
    def __init__(self,
                 num_layers=Config.num_decoder_layers,
                 dropout=Config.dropout_rate,
                 bidirectional=True,
                 hidden_size=Config.decoder_hidden_size,
                 transform=None,
                 pretrained=False,
                 model_name=Config.base_model):
        super(Model, self).__init__()

        self.transform = transform

        if pretrained:
            logging.info(
                msg=f'Using {Config.base_model} as features extractor')

            self.encoder = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                num_classes=Config.decoder_input_size,
            )
            # input_size = [
            #     m for m in self.encoder.modules() if isinstance(m, nn.Conv2d)
            # ][-1].out_channels
        else:
            logging.info(
                msg=f'Using the defined ConvNet as features extractor')

            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.ELU(),
                nn.Linear(in_features=69, out_features=32),
            )

        self.decoder = nn.GRU(input_size=Config.decoder_input_size * 21,
                              hidden_size=hidden_size,
                              bidirectional=bidirectional,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_size * 2,
                            out_features=len(Config.labels))

        self.predictions_decoder = CTCBeamDecoder(
            labels=Config.labels,
            model_path=None,
            alpha=0,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=100,
            num_processes=Config.num_workers,
            blank_id=0,
            log_probs_input=False)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad == True]

        optims = {
            "adam":
            th.optim.Adam(lr=Config.lr,
                          params=params,
                          eps=Config.eps,
                          weight_decay=Config.weight_decay),
            "adamw":
            th.optim.AdamW(lr=Config.lr,
                           params=params,
                           eps=Config.eps,
                           weight_decay=Config.weight_decay),
            "sgd":
            th.optim.SGD(lr=Config.lr,
                         params=params,
                         weight_decay=Config.weight_decay),
        }

        opt = optims[Config.optimizer]

        sc1 = th.optim.lr_scheduler.LambdaLR(optimizer=opt,
                                             lr_lambda=utils.ramp_scheduler,
                                             verbose=True)

        sc2 = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='max',
            factor=0.1,
            patience=Config.reducing_lr_patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=Config.cooldown,
            min_lr=0,
            eps=Config.eps,
            verbose=True,
        )

        if Config.reduce_lr_on_plateau:
            scheduler = sc2

            return {
                "optimizer": opt,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        else:
            scheduler = sc1

            return [opt], [scheduler]

    def forward(self, inputs, targets=None):
        if self.transform is not None:
            inputs = self.transform(inputs)
        print(f'[Model info] inputs {inputs.size()}')
        features = self.encoder(inputs)
        print(f'[Model info] features {features.size()}')
        n, c, h, w = features.size()
        # reshape features for recurrence
        features = features.view(n, c * h, w).permute(0, 2, 1)
        print(f'[Model info] reshaped features {features.size()}')

        hidden_states, c = self.decoder(features)
        print(f'[Model info] hidden_states {hidden_states.size()}')
        logits = self.fc(hidden_states).transpose(1, 0)
        print(f'[Model info] logits {logits.size()}')
        input_lengths = th.full(size=(logits.size(1), ),
                                fill_value=(logits.size(0)),
                                dtype=th.int32)

        print(f'[Model info] input lengths {input_lengths.size()}')
        print(f'[Model info] input lengths {input_lengths}')

        if targets is not None:
            logits = F.log_softmax(logits, dim=2)
            print('[Model info] targets ', targets.size())

            target_lengths = th.full(size=(targets.size(0), ),
                                     fill_value=targets.size(1),
                                     dtype=th.int32)
            print(f'[Model info] target lengths {target_lengths.size()}')
            print(f'[Model info] target lengths {target_lengths}')

            loss = self.get_loss(logits=logits,
                                 targets=targets,
                                 input_lengths=input_lengths,
                                 target_lengths=target_lengths)

            beam_results, _, _, out_lens = self.decode_predictions(
                logits=logits.transpose(0, 1), seq_lengths=target_lengths)
        else:
            loss = None

            beam_results, _, _, out_lens = self.decode_predictions(
                logits.transpose(0, 1), input_lengths)

        print("[INFO] beam_results ", beam_results.shape)

        return logits, beam_results, out_lens, loss

    def training_step(self, batch_idx):
        pass

    def validation_step(self, batch_idx):
        pass

    def decode_predictions(self, logits, seq_lengths):

        return self.predictions_decoder.decode(probs=logits,
                                               seq_lens=seq_lengths)

    def get_loss(self, logits, targets, input_lengths, target_lengths):

        return F.ctc_loss(logits, targets, input_lengths, target_lengths)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Config.data_dir, 'Train.csv'), nrows=100)
    tokenizer = tokenizer.Tokenizer()
    dm = DataModule(df=df, tokenizer=tokenizer)
    dm.setup()
    model = Model(pretrained=False).cuda()
    print(model)
    for data in dm.val_dataloader():
        images = data['img']
        labels = data['label']
        print("[INFO] inputs : ", images.shape)
        print("[INFO] labels : ", labels.shape)

        logits, beam_results, out_lens, loss = model(inputs=images.cuda(),
                                                     targets=labels.cuda())

        print('[INFO] Loss : ', loss)
        print('\t\t\t\t\t\t===== Decoding time =====\n')
        for batch_idx in range(beam_results.shape[0]):
            target_ids = labels[batch_idx]
            pred_ids = beam_results[batch_idx][0][:out_lens[batch_idx][0]]

            target_text = tokenizer.decode(ids=target_ids)
            pred_text = tokenizer.decode(ids=pred_ids)
            print('target ids: ', target_ids)
            print("Prediction ids: ", pred_ids)

            print('target text: ', target_text)
            print("Prediction text: ", pred_text)
            print()
        break