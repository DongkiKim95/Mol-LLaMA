'''
Copyright (c) DP Technology.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
Adopted from 
https://github.com/deepmodeling/Uni-Mol/blob/main/unimol/unimol/models/unimol.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import get_activation_fn
from models.unimol.transformer_encoder_with_pair import TransformerEncoderWithPair

def init_bert_params(module):
    if not getattr(module, 'can_global_init', True):
        return
    def normal_(data):
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

class BaseUnicoreModel(nn.Module):
    """Base class for unicore models."""

    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_args = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. 
        """
        return super().load_state_dict(state_dict, strict)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)

class SimpleUniMolModel(BaseUnicoreModel):
    def __init__(
            self, 
            dictionary,
            unimol_encoder_layers = 15,
            unimol_encoder_embed_dim = 512,
            unimol_encoder_ffn_embed_dim = 2048,
            unimol_encoder_attention_heads = 64,
            unimol_emb_dropout = 0.1,
            unimol_dropout = 0.1,
            unimol_attention_dropout = 0.1,
            unimol_activation_dropout = 0.0,
            unimol_max_seq_len = 512,
            unimol_activation_fn = "gelu",
            unimol_delta_pair_repr_norm_loss = -1.0
        ):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), unimol_encoder_embed_dim, self.padding_idx
        )

        self.encoder = TransformerEncoderWithPair(
            encoder_layers=unimol_encoder_layers,
            embed_dim=unimol_encoder_embed_dim,
            ffn_embed_dim=unimol_encoder_ffn_embed_dim,
            attention_heads=unimol_encoder_attention_heads,
            emb_dropout=unimol_emb_dropout,
            dropout=unimol_dropout,
            attention_dropout=unimol_attention_dropout,
            activation_dropout=unimol_activation_dropout,
            max_seq_len=unimol_max_seq_len,
            activation_fn=unimol_activation_fn,
            no_final_head_layer_norm=unimol_delta_pair_repr_norm_loss < 0,
        )

        K = 128
        n_edge_type = 31 * 31
        self.gbf_proj = NonLinearHead(
            K, unimol_encoder_attention_heads, unimol_activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)
        self.num_features = unimol_encoder_embed_dim
        self.apply(init_bert_params)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_edge_type,
    ):
        padding_mask = src_tokens.eq(self.padding_idx).bool()
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
        return encoder_rep, ~padding_mask



class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)