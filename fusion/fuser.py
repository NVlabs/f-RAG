# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from MolecularAI/MolBART.
#
# Source:
# https://github.com/MolecularAI/MolBART/blob/master/megatron_molbart/megatron_bart.py
#
# The license for this can be found in license_thirdparty/LICENSE_MOLBART.
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F


class Fuser(torch.nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.attn_dropout = torch.nn.Dropout(p=dropout)
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key_value = torch.nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, key_padding_mask=None):
        bsz, tgt_len, embed_dim = query.size()
        num_ret, bsz, ret_len, embed_dim = key.size()

        # Compute attention projections
        q = self.query(query)
        k, v = torch.split(self.key_value(key), embed_dim, dim=-1)
        
        # Scale query and reshape
        q = q.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim) * self.scaling    # b*n, t, d'
        k = k.contiguous().view(bsz * self.num_heads, -1, self.head_dim)  # b*n, k*r, d'
        v = v.contiguous().view(bsz * self.num_heads, -1, self.head_dim)  # b*n, k*r, d'

        # Compute attention scores
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # b*n, t, k*r
        
        # Apply padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            key_padding_mask = key_padding_mask.transpose(0, 1).contiguous().reshape(bsz, src_len)
            attn_weights = \
                attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Compute attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_dropout(attn_weights)

        # Compute context and output projection
        attn = torch.bmm(attn_probs, v)     # b*n, t, d'
        if attn.size(1) == 1:   # a single decoder step (sequence length == 1)
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn).transpose(0, 1)
        return attn
