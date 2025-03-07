# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import sys
import time

from loguru import logger
import transformers
from transformers import TrainingArguments, set_seed, Trainer
from safetensors.torch import save_file

from safe.tokenizer import SAFETokenizer
from safe.trainer.cli import ModelArguments, DataArguments

sys.path.insert(0, os.getcwd())

from fusion.safegpt import SAFEFusionModel
from fusion.trainer.utils import get_dataset, SAFEFusionCollator


class FusionTrainer(Trainer):
    def __init__(self, *args, dispatch_batches=False, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.accelerator.dispatch_batches = dispatch_batches
        except:
            self.accelerator._dispatch_batches = dispatch_batches

    def compute_loss(self, model, inputs_tuple, return_outputs=False):
        inputs, retrieved_inputs = inputs_tuple
        outputs = model(inputs, retrieved_inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        # We don't use .loss here since the model may return tuples instead of ModelOutput
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        fuser_parameters = {}
        for name, parameter in self.model.named_parameters():
            if name.startswith('fuser'):
                fuser_parameters[name] = parameter

        # overwrite the model checkpoint; save the fuser only
        save_file(fuser_parameters,
                  os.path.join(output_dir, 'model.safetensors'),
                  metadata={'format': 'pt'})


def train(model_args, data_args, training_args):
    # handling arguments
    training_args.do_train = True
    training_args.disable_tqdm = False
    training_args.remove_unused_columns = False
    training_args.save_only_model = True
    run_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    training_args.run_name = run_name
    logger.info(training_args)
    set_seed(training_args.seed)
    
    # load model and tokenizer
    model = SAFEFusionModel.from_pretrained('datamol-io/safe-gpt')
    tokenizer = SAFETokenizer.from_pretrained('datamol-io/safe-gpt')

    # load dataset
    with training_args.main_process_first():
        # if the dataset is streaming we tokenize on the fly as it would be faster
        dataset = get_dataset(
            data_args.dataset,
            tokenizer=tokenizer,
            streaming=data_args.streaming,
            max_length=model_args.model_max_length
        )
    
    data_collator = SAFEFusionCollator(
        tokenizer=tokenizer,
        max_length=model_args.model_max_length
    )
    
    n_params = sum([p.numel() for p in model.parameters()])
    n_train_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'{n_params} total parameters | {n_train_params} trainable parameters')

    trainer = FusionTrainer(
        model=model,
        tokenizer=None,
        dispatch_batches=(data_args.streaming is not True),
        train_dataset=dataset['train'].shuffle(seed=(training_args.seed or 42)),
        # eval_dataset=dataset.get('test', None),
        args=training_args,
        data_collator=data_collator
    )

    train_result = trainer.train()
    trainer.save_model()
    
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()


if __name__ == '__main__':
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)
