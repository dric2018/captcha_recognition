import os
import sys
import pandas as pd
import numpy as np

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
from tokenizer import Tokenizer
import logging

logging.basicConfig(level=logging.INFO)


class Model(pl.LightningModule):
    def __init__(self,
                 num_layers=Config.num_decoder_layers,
                 dropout=Config.dropout_rate,
                 bidirectional=True,
                 n_out_channels=256,
                 hidden_size=Config.decoder_hidden_size,
                 transform=None,
                 pretrained=False,
                 model_name=Config.base_model):
        super(Model, self).__init__()

        self.tokenizer = Tokenizer()
        self.transform = transform
        self.n_out_channels = n_out_channels
        self.best_loss = np.inf

        if pretrained:
            logging.info(
                msg=f'Using {Config.base_model} as features extractor')

            self.encoder = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                #num_classes=0,global_pool='',
                features_only=True,
            )

            self.decoder = nn.GRU(input_size=n_out_channels * 12,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  dropout=dropout)
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

    def select_features(self, features: list, n_out_channels: int = 128):
        features = [f for f in features if f.size(1) == n_out_channels]
        return features[-1]

    def forward(self, inputs, targets=None):
        if self.transform is not None:
            inputs = self.transform(inputs)
        # print(f'[Model info] inputs {inputs.size()}')
        features = self.encoder(inputs)
        features = self.select_features(features=features,
                                        n_out_channels=self.n_out_channels)
        # print(f'[Model info] features {features.size()}')

        n, c, h, w = features.size()
        # reshape features for recurrence
        features = features.view(n, c * h, w).permute(0, 2, 1)
        # print(f'[Model info] reshaped features {features.size()}')

        hidden_states, c = self.decoder(features)
        # print(f'[Model info] hidden_states {hidden_states.size()}')
        logits = self.fc(hidden_states).transpose(1, 0)
        # print(f'[Model info] logits {logits.size()}')
        input_lengths = th.full(size=(logits.size(1), ),
                                fill_value=(logits.size(0)),
                                dtype=th.int32)

        # print(f'[Model info] input lengths {input_lengths.size()}')
        # print(f'[Model info] input lengths {input_lengths}')

        if targets is not None:
            log_probs = F.log_softmax(logits, dim=2)
            # print('[Model info] targets ', targets.size())

            target_lengths = th.full(size=(targets.size(0), ),
                                     fill_value=targets.size(1),
                                     dtype=th.int32)
            # print(f'[Model info] target lengths {target_lengths.size()}')
            # print(f'[Model info] target lengths {target_lengths}')

            loss = self.get_loss(log_probs=log_probs,
                                 targets=targets,
                                 input_lengths=input_lengths,
                                 target_lengths=target_lengths)

            beam_results, _, _, out_lens = self.decode_predictions(
                logits=logits.transpose(0, 1), seq_lengths=target_lengths)
        else:
            loss = None

            beam_results, _, _, out_lens = self.decode_predictions(
                logits.transpose(0, 1), input_lengths)

        # print("[INFO] beam_results ", beam_results.shape)

        return log_probs, beam_results, out_lens, loss

    def training_step(self, batch_idx, batch):
        images = data['img']
        targets = data['label']
        # forward pass + compute metrics
        log_probs, beam_results, out_lens, loss = self(inputs=images,
                                                       targets=targets)

        pred_ids, pred_texts, target_texts = self.batch_decode(
            tokenizer=self.tokenizer,
            beam_results=beam_results,
            out_lens=out_lens,
            targets=targets,
            select_index=0)

        # compute accuracy
        train_acc = self.get_accuracy(pred_ids=pred_ids, targets=targets)

        # logging phase

        self.log("train_loss",
                 value=loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)

        self.log("train_acc",
                 value=train_acc,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=True)

        return {
            "loss": loss,
            "acc": train_acc,
        }

    def validation_step(self, batch_idx):
        images = data['img']
        targets = data['label']
        # forward pass + compute metrics
        log_probs, beam_results, out_lens, val_loss = self(inputs=images,
                                                           targets=targets)

        pred_ids, pred_texts, target_texts = self.batch_decode(
            tokenizer=self.tokenizer,
            beam_results=beam_results,
            out_lens=out_lens,
            targets=targets,
            select_index=0)

        # compute accuracy
        val_acc = self.get_accuracy(pred_ids=pred_ids, targets=targets)

        # logging phase
        self.log("val_loss",
                 value=val_loss,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        self.log("val_acc",
                 value=val_acc,
                 prog_bar=True,
                 logger=True,
                 on_step=False,
                 on_epoch=True)

        return {
            "loss": val_loss,
            "val_acc": val_acc,
            "images": images,
            "targets": targets,
            "predictions": predictions
        }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()
        # images
        images = th.stack([x['images'] for x in outputs])
        # targets
        targets = th.stack([x['targets'] for x in outputs])
        # predictions
        predictions = th.stack([x['predictions'] for x in outputs])

        # logging using tensorboard logger

        grid = utils.view_sample(images=images,
                                 labels=targets,
                                 predictions=predictions,
                                 return_image=True,
                                 show=False)

        self.logger.experiment.add_image(tag='predictions_grid',
                                         img_tensor=np.array(grid),
                                         dataformats='HWC')

        self.logger.experiment.add_scalar("Loss/Validation", avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation", avg_acc,
                                          self.current_epoch)
        # monitor acc improvements
        if avg_loss < self.best_loss:
            print("\n")
            print(
                f'[INFO] validation loss improved from {self.best_loss} to {avg_loss}'
            )
            self.best_loss = avg_loss
            print()
        else:
            print("\n")
            print('[INFO] validation loss did not improve')
            print()

    def decode_predictions(self, logits, seq_lengths):

        return self.predictions_decoder.decode(probs=logits,
                                               seq_lens=seq_lengths)

    def get_loss(self, log_probs, targets, input_lengths, target_lengths):

        return F.ctc_loss(blank=0,
                          reduction='mean',
                          log_probs=log_probs.cpu(),
                          targets=targets.cpu(),
                          input_lengths=input_lengths.cpu(),
                          target_lengths=target_lengths.cpu())

    def get_accuracy(self, pred_ids, targets):

        return (pred_ids.cpu() == targets.cpu()).float().mean()

    def batch_decode(self,
                     tokenizer,
                     beam_results,
                     out_lens,
                     targets,
                     select_index: int = 0):

        predictions_ids, predictions, target_texts = [], [], []

        # print('\t\t\t\t\t\t===== Decoding time =====\n')
        for batch_idx in range(targets.size(0)):
            target_ids = targets[batch_idx]
            pred_ids = beam_results[batch_idx][
                select_index][:out_lens[batch_idx][0]]

            target_text = tokenizer.decode(ids=target_ids)
            pred_text = tokenizer.decode(ids=pred_ids)
            # print('target ids: ', target_ids, target_ids.size())
            # print("Prediction ids: ", pred_ids, pred_ids.size())

            # print('target text: ', target_text)
            # print("Prediction text: ", pred_text)
            # print()

            if pred_ids.size(-1) < target_ids.size(-1):
                padding_len = targets.size(-1) - pred_ids.size(-1)
                padding = th.Tensor([0] * padding_len)

                pred_ids = th.cat(tensors=(pred_ids, padding), dim=0)

            predictions_ids.append(pred_ids)
            predictions.append(pred_text)
            target_texts.append(target_text)

        return th.stack(predictions_ids), predictions, target_texts


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(Config.data_dir, 'Train.csv'), nrows=1000)
    tokenizer = Tokenizer()
    dm = DataModule(df=df, tokenizer=tokenizer)
    dm.setup()
    tfms = transforms.Compose([
        transforms.Resize(size=(Config.img_H, Config.img_W)),
    ])
    model = Model(pretrained=True, transform=tfms).cuda()
    # print(model)

    for batch_idx, data in enumerate(dm.val_dataloader()):
        images = data['img']
        labels = data['label']
        print("[INFO] inputs : ", images.shape)
        print("[INFO] labels : ", labels.shape)

        log_probs, beam_results, out_lens, loss = model(inputs=images.cuda(),
                                                        targets=labels.cuda())

        pred_ids, pred_texts, target_texts = model.batch_decode(
            tokenizer=tokenizer,
            beam_results=beam_results,
            out_lens=out_lens,
            targets=labels,
            select_index=0)

        acc = model.get_accuracy(pred_ids=pred_ids, targets=labels)

        print(f'[INFO] Loss={loss} , acc={acc}')

        break