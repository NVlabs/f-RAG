# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from datamol-io/safe.
#
# Source:
# https://github.com/datamol-io/safe/blob/main/safe/trainer/model.py
#
# The license for this can be found in license_thirdparty/LICENSE_SAFE.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import torch
from transformers import GPT2DoubleHeadsModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
from transformers.generation.utils import GenerateDecoderOnlyOutput
from fusion.fuser import Fuser

from transformers import logging
logging.set_verbosity_error()


class GPT2ModelForFusion(GPT2Model):
    def before_fusion(
        self,
        input_ids,
        past_key_values = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        use_cache = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.reshape(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.reshape(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(None, self.config.n_layer)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None
        block, layer_past = self.h[0], past_key_values[0]

        outputs = block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask[0],
            use_cache=use_cache,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states.reshape(output_shape)

        if use_cache is True:
            presents = presents + (outputs[1],)

        return hidden_states, {'past_key_values': past_key_values,
                                'attention_mask': attention_mask,
                                'head_mask': head_mask,
                                'use_cache': use_cache,
                                'presents': presents}

    def after_fusion(self, hidden_states, past_key_values, attention_mask,
                     head_mask, use_cache, presents):
        output_shape = hidden_states.shape
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if i == 0:  # skip the first layer
                continue

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape(output_shape)
        
        return hidden_states, presents


class SAFEFusionModel(GPT2DoubleHeadsModel):
    def __init__(self, config):
        super().__init__(config)
        del self.multiple_choice_head
        self.transformer = GPT2ModelForFusion(config)
        self.fuser = Fuser(embed_dim=config.n_embd)
        
        # freeze the model except for the fuser
        for name, parameter in self.named_parameters():
            if name.startswith('fuser'):
                continue
            else:
                parameter.requires_grad = False
        
    def forward(self, inputs, retrieved_inputs=None):
        labels = None
        if 'labels' in inputs:      # train
            labels = inputs.pop('labels')
        if 'fusion_mask' in inputs:
            fusion_mask = inputs.pop('fusion_mask')
        
        hidden_states, kwargs = self.transformer.before_fusion(**inputs)
        if retrieved_inputs is not None:
            retrieved_hidden_states, _ = self.transformer.before_fusion(**retrieved_inputs)
            fused_hidden_states = self.fuser(hidden_states,
                                             retrieved_hidden_states,
                                             ~retrieved_inputs['attention_mask'].to(torch.bool))
            fusion_mask = fusion_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
            # fuse the fusion part only
            hidden_states = torch.where(fusion_mask, hidden_states, fused_hidden_states)
        hidden_states, past_key_values = self.transformer.after_fusion(hidden_states, **kwargs)
        lm_logits = self.lm_head(hidden_states)
        
        lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_func = torch.nn.CrossEntropyLoss()
            lm_loss = loss_func(shift_logits.reshape(-1, shift_logits.size(-1)),
                                shift_labels.reshape(-1))
        
        return GPT2DoubleHeadsModelOutput(loss=lm_loss,
                                          logits=lm_logits,
                                          past_key_values=past_key_values)

    def sample(
        self,
        input_ids,
        retrieved_ids = None,
        logits_processor = None,
        stopping_criteria = None,
        logits_warper = None,
        max_length = None,
        pad_token_id = None,
        eos_token_id = None,
        dot_token_id = None,
        output_attentions = None,
        output_hidden_states = None,
        output_scores = None,
        output_logits = None,
        return_dict_in_generate = None,
        synced_gpus = False,
        streamer = None,
        **model_kwargs,
    ):
        # init values
        eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)
        scores = ()

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False

        # auto-regressive generation
        first = True
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            if first:
                # prepare fusion_mask
                fusion_mask = torch.full(model_inputs['input_ids'][0].shape, True)
                fusion_mask[:-1] = False    # input[:-1] is the fusion part
                model_inputs['fusion_mask'] = fusion_mask.unsqueeze(0).to(model_inputs['input_ids'].device)
                
                outputs = self(model_inputs, retrieved_ids)
            else:
                outputs = self(model_inputs)
            first = False

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            scores += (next_token_scores,)
            
            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished:
                break

        return GenerateDecoderOnlyOutput(sequences=input_ids,
                                         past_key_values=model_kwargs.get("past_key_values"))

    def _validate_model_kwargs(self, model_kwargs):
        pass
