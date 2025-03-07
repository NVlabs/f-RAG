# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from datamol-io/safe.
#
# Source:
# https://github.com/datamol-io/safe/blob/main/safe/sample.py
#
# The license for this can be found in license/LICENSE_SAFE.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import random
import torch
import safetensors
from safe import SAFEDesign
from transformers import GenerationConfig
from safe.tokenizer import SAFETokenizer
from fusion.safegpt import SAFEFusionModel


class SAFEFusionDesign(SAFEDesign):
    @classmethod
    def load_default(cls):
        model = SAFEFusionModel.from_pretrained('datamol-io/safe-gpt')
        tokenizer = SAFETokenizer.from_pretrained('datamol-io/safe-gpt')
        gen_config = GenerationConfig.from_pretrained('datamol-io/safe-gpt')
        return cls(model=model, tokenizer=tokenizer, generation_config=gen_config, verbose=False)

    def load_fuser(self, fuser_path):
        fuser_parameters = {}
        with safetensors.safe_open(fuser_path, framework='pt') as f:
            for k in f.keys():
                fuser_parameters[k] = f.get_tensor(k)
                fuser_parameters[k].requires_grad = False
        self.model.load_state_dict(fuser_parameters, strict=False)
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        return self

    def _generate(
        self,
        n_samples = None,
        safe_prefix = None,
        n_retrieve = 10,
        **kwargs
    ):
        n_samples = 1   # only support single-sized batch

        tokenizer = self.tokenizer.get_pretrained()
        dot_token_id = tokenizer('.')['input_ids'][1]
        
        input_ids = tokenizer(safe_prefix, return_tensors="pt")
        retrieved_ids = [tokenizer(smiles, return_tensors=None)
                         for smiles in self._retrieve(n_retrieve)]
        retrieved_ids = tokenizer.pad(retrieved_ids, return_tensors='pt', padding=True)

        kwargs["do_sample"] = True
        kwargs["output_scores"] = True
        kwargs["return_dict_in_generate"] = True
        kwargs["num_return_sequences"] = n_samples
        kwargs["max_length"] = 100
        kwargs.setdefault("early_stopping", False)
        
        for k in input_ids:
            input_ids[k] = input_ids[k][:, :-1]
        for k, v in input_ids.items():
            if torch.is_tensor(v):
                input_ids[k] = v.to(self.model.device)  # 1, input_len

        for k, v in retrieved_ids.items():
            retrieved_ids[k] = v.unsqueeze(1).to(self.model.device) # num_ret, 1, ret_len
        
        outputs = self.model.generate(
            **input_ids,
            retrieved_ids=retrieved_ids,
            dot_token_id=dot_token_id,
            generation_config=self.generation_config,
            **kwargs
        )
        sequences = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return sequences

    def _retrieve(self, n_retrieve):
        return random.sample(self.frags, n_retrieve)
