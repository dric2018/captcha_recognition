from config import Config
import os
import numpy as np
from typing import Union, List
import torch as th

import logging

logging.basicConfig(level=logging.INFO)


class Tokenizer:
    def __init__(self,
                 base=None,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 sos_token='[SOS]',
                 eos_token='[EOS]',
                 do_lower_case: bool = False,
                 labels: str = Config.labels):
        self.base = base
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.sos_token = sos_token
        self.do_lower_case = do_lower_case
        self.labels = labels
        self.vocab = {v: k for k, v in enumerate(self.labels)}
        self.vocab[pad_token] = len(self.vocab)
        self.vocab[unk_token] = len(self.vocab)
        self.vocab[sos_token] = len(self.vocab)
        self.vocab[eos_token] = len(self.vocab)

    def encode(self, text: str, padding: bool = False, max_length: int = None):
        if self.do_lower_case:
            text = text.lower()

        input_ids = [
            self.vocab[letter]
            if letter in self.labels else self.vocab[self.unk_token]
            for letter in text
        ]
        if padding:
            if max_length is not None:
                if max_length - len(text) > 0:
                    padding_len = max_length - len(text)
                    input_ids = input_ids + [self.vocab[self.pad_token]
                                             ] * padding_len
            else:
                logging.warning(
                    msg=
                    'Can only pad inputs if max_length argument is provided\n')

        code = {"input_ids": input_ids}

        return code

    def decode(self, ids: Union[th.Tensor, List, np.array]):
        tokens = [
            self.get_token_from_id(id_)
            if id_ != self.vocab[self.pad_token] else " " for id_ in ids
        ]
        text = "".join(tokens)

        return text

    def save(self, path):
        pass

    def get_token_from_id(self, id_: int):
        for k, v in zip(self.vocab.keys(), self.vocab.values()):
            if v == id_:
                token = k
                break
            else:
                token = self.unk_token

        return token

    def get_id_from_token(self, token: str):
        if token in self.labels:
            return self.vocab[token]
        else:
            logging.warning(msg='Unknown token found')
            return self.vocab[self.unk_token]
