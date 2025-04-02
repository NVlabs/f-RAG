# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from datamol-io/safe.
#
# Source:
# https://github.com/datamol-io/safe/blob/main/safe/trainer/data_utils.py
# https://github.com/datamol-io/safe/blob/main/safe/trainer/collator.py
#
# The license for this can be found in license_thirdparty/LICENSE_SAFE.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import torch
from functools import partial
from typing import Any, Callable, Dict, Optional
import datasets
import upath
import copy
from safe.tokenizer import SAFETokenizer
from safe.trainer.collator import SAFECollator


def tokenize_fn(
    row: Dict[str, Any],
    tokenizer: Callable,
    input_column: str = "input",
    retrieved_column: str = "retrieved",
    max_length: Optional[int] = None,
    padding: bool = False,
):
    fast_tokenizer = (
        tokenizer.get_pretrained() if isinstance(tokenizer, SAFETokenizer) else tokenizer
    )

    input_dict = fast_tokenizer(
        row[input_column],
        truncation=(max_length is not None),
        max_length=max_length,
        padding=padding,
        return_tensors=None,
    )

    retrieved_dict_list = [fast_tokenizer(
        retrieved,
        truncation=(max_length is not None),
        max_length=max_length,
        padding=padding,
        return_tensors=None,
    ) for retrieved in row[retrieved_column].split('.')]

    input_dict['retrieved_ids'] = [retrieved_dict['input_ids'] for retrieved_dict in retrieved_dict_list]
    input_dict['retrieved_token_type_ids'] = [retrieved_dict['token_type_ids'] for retrieved_dict in retrieved_dict_list]
    input_dict['retrieved_attention_mask'] = [retrieved_dict['attention_mask'] for retrieved_dict in retrieved_dict_list]
    return input_dict


def get_dataset(
    data_path,
    tokenizer,
    streaming: bool = True,
    input_column: Optional[str] = "input",
    retrieved_column: Optional[str] = "retrieved",
    max_length: Optional[int] = None,
    num_shards=1024,
):
    """Get the datasets from the config file"""
    raw_datasets = {}
    data_path = upath.UPath(str(data_path))

    # the we need to load from disk
    data_path = str(data_path)
    # for some reason, the datasets package is not able to load the dataset
    # because the split where not originally proposed
    raw_datasets = datasets.load_from_disk(data_path)

    if streaming:
        if isinstance(raw_datasets, datasets.DatasetDict):
            previous_num_examples = {k: len(dt) for k, dt in raw_datasets.items()}
            raw_datasets = datasets.IterableDatasetDict(
                {
                    k: dt.to_iterable_dataset(num_shards=num_shards)
                    for k, dt in raw_datasets.items()
                }
            )
            for k, dt in raw_datasets.items():
                if previous_num_examples[k] is not None:
                    setattr(dt, "num_examples", previous_num_examples[k])
        else:
            num_examples = len(raw_datasets)
            raw_datasets = raw_datasets.to_iterable_dataset(num_shards=num_shards)
            setattr(raw_datasets, "num_examples", num_examples)
    
    return raw_datasets.map(
        partial(
            tokenize_fn,
            tokenizer=tokenizer,
            input_column=input_column,
            retrieved_column=retrieved_column,
            max_length=max_length,
        )
    )
    

class SAFEFusionCollator(SAFECollator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = self.get_tokenizer()
        self.dot_token_id = self.tokenizer('.')['input_ids'][1]

    def __call__(self, samples):
        examples = copy.deepcopy(samples)

        input_examples = [{'input_ids': example['input_ids'],
                           'token_type_ids': example['token_type_ids'],
                           'attention_mask': example['attention_mask']}
                           for example in examples]
        
        batch = self.tokenizer.pad(
            input_examples,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )   # b x len
        
        fusion_mask = torch.full(batch['input_ids'].shape, True)
        loss_mask = torch.full(batch['input_ids'].shape, True)
        for i, b in enumerate(batch['input_ids']):
            # input is Frag2.Frag3.SimFrag1 or Frag2.SimFrag1
            idx = (b == self.dot_token_id).nonzero(as_tuple=True)[0][-1]
            fusion_mask[i, :idx] = False        # input[:idx] is Frag2.Frag3 or Frag2
            loss_mask[i, idx + 1:] = False      # input[idx + 1:] is SimFrag1
        batch['fusion_mask'] = fusion_mask
        
        labels = batch.get(self.label_key, batch["input_ids"].clone())
        labels[loss_mask] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        retrieved_examples = []
        for example in examples:
            for input_ids, token_type_ids, attention_mask in zip(example['retrieved_ids'],
                                                                 example['retrieved_token_type_ids'],
                                                                 example['retrieved_attention_mask']):
                retrieved_examples.append({'input_ids': input_ids,
                                           'token_type_ids': token_type_ids,
                                           'attention_mask': attention_mask})

        retrieved_batch = self.tokenizer.pad(
            retrieved_examples,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )   # (k * b) x ret_len
        batch_size = batch['input_ids'].shape[0]
        num_retrieved = retrieved_batch['input_ids'].shape[0] // batch_size
        for key in retrieved_batch:     # k x b x ret_len
            retrieved_batch[key] = retrieved_batch[key].reshape(batch_size, num_retrieved, -1).transpose(0, 1)
        
        return batch, retrieved_batch
