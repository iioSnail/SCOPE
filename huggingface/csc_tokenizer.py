import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Union, Optional

import tokenizers
import torch
from torch import NoneType
from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import http_user_agent
from pypinyin import pinyin, Style
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy
from transformers.utils.generic import TensorType

try:
    from tokenizers import BertWordPieceTokenizer
except:
    from tokenizers.implementations import BertWordPieceTokenizer

from transformers import BertTokenizerFast, BatchEncoding

cache_path = Path(os.path.abspath(__file__)).parent


def download_file(filename: str, path: Path):
    if os.path.exists(cache_path / filename):
        return

    if os.path.exists(path / filename):
        shutil.copyfile(path / filename, cache_path / filename)
        return

    hf_hub_download(
        "iioSnail/ChineseBERT-for-csc",
        filename,
        local_dir=cache_path,
        user_agent=http_user_agent(),
    )
    time.sleep(0.2)


class ChineseBertTokenizer(BertTokenizerFast):

    def __init__(self, **kwargs):
        super(ChineseBertTokenizer, self).__init__(**kwargs)

        self.path = Path(kwargs['name_or_path'])
        vocab_file = cache_path / 'vocab.txt'
        config_path = cache_path / 'config'
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        self.max_length = 512

        download_file('vocab.txt', self.path)
        self.tokenizer = BertWordPieceTokenizer(str(vocab_file))

        # load pinyin map dict
        download_file('config/pinyin_map.json', self.path)
        with open(config_path / 'pinyin_map.json', encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)

        # load char id map tensor
        download_file('config/id2pinyin.json', self.path)
        with open(config_path / 'id2pinyin.json', encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)

        # load pinyin map tensor
        download_file('config/pinyin2tensor.json', self.path)
        with open(config_path / 'pinyin2tensor.json', encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def __call__(self,
                 text: Union[str, List[str], List[List[str]]] = None,
                 text_pair: Union[str, List[str], List[List[str]], NoneType] = None,
                 text_target: Union[str, List[str], List[List[str]]] = None,
                 text_pair_target: Union[str, List[str], List[List[str]], NoneType] = None,
                 add_special_tokens: bool = True,
                 padding: Union[bool, str, PaddingStrategy] = False,
                 truncation: Union[bool, str, TruncationStrategy] = None,
                 max_length: Optional[int] = None,
                 stride: int = 0,
                 is_split_into_words: bool = False,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors: Union[str, TensorType, NoneType] = None,
                 return_token_type_ids: Optional[bool] = None,
                 return_attention_mask: Optional[bool] = None,
                 return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False,
                 return_offsets_mapping: bool = False,
                 return_length: bool = False,
                 verbose: bool = True, **kwargs) -> BatchEncoding:
        encoding = super(ChineseBertTokenizer, self).__call__(
            text=text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
        )

        input_ids = encoding.input_ids

        pinyin_ids = None
        if type(text) == str:
            pinyin_ids = self.convert_ids_to_pinyin_ids(input_ids)

        if type(text) == list:
            pinyin_ids = []
            for ids in input_ids:
                pinyin_ids.append(self.convert_ids_to_pinyin_ids(ids))

        if torch.is_tensor(encoding.input_ids):
            pinyin_ids = torch.LongTensor(pinyin_ids)

        encoding['pinyin_ids'] = pinyin_ids

        return encoding

    def tokenize_sentence(self, sentence):
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert，token nums should be same as pinyin token nums
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        return input_ids, pinyin_ids

    def convert_ids_to_pinyin_ids(self, ids: List[int]):
        pinyin_ids = []
        tokens = self.convert_ids_to_tokens(ids)
        for token in tokens:
            if len(token) > 1:
                pinyin_ids.append([0] * 8)
                continue

            pinyin_string = pinyin(token, style=Style.TONE3, errors=lambda x: [['not chinese'] for _ in x])[0][0]

            if pinyin_string == "not chinese":
                pinyin_ids.append([0] * 8)
                continue

            if pinyin_string in self.pinyin2tensor:
                pinyin_ids.append(self.pinyin2tensor[pinyin_string])
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_ids.append(pinyin_ids)

        return pinyin_ids

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids
