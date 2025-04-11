import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import torch.distributed as dist
import math

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
        )

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)

        self.ln = nn.LayerNorm(d_model)
    def attention(self, q, k, v, key_padding_mask):
        return self.attn(q, k, v, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, q, k, v, key_padding_mask=None):
        x = q
        q, k, v = self.ln_q(q), self.ln_k(k), self.ln_v(v)
        x = x + self.attention(q, k, v, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.self_attn_block_2D = ResidualAttentionBlock(d_model, n_head)
        self.self_attn_block_3D = ResidualAttentionBlock(d_model, n_head)
        self.cross_attn_block_2Dto3D = ResidualAttentionBlock(d_model, n_head)
        self.cross_attn_block_3Dto2D = ResidualAttentionBlock(d_model, n_head)

    def forward(self, features_2d, features_3d, mask_2d, mask_3d):
        features_2d = self.self_attn_block_2D(features_2d, features_2d, features_2d, key_padding_mask=~mask_2d)
        features_2d = features_2d * mask_2d.unsqueeze(-1)

        features_3d = self.self_attn_block_3D(features_3d, features_3d, features_3d, key_padding_mask=~mask_3d)
        features_3d = features_3d * mask_3d.unsqueeze(-1)
        
        features_2d = self.cross_attn_block_3Dto2D(features_2d, features_3d, features_3d, key_padding_mask=~mask_3d)
        features_2d = features_2d * mask_2d.unsqueeze(-1)

        features_3d = self.cross_attn_block_2Dto3D(features_3d, features_2d, features_2d, key_padding_mask=~mask_2d)
        features_3d = features_3d * mask_3d.unsqueeze(-1)

        return features_2d, features_3d

class BlendingModule(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, dims):
        super(BlendingModule, self).__init__()

        self.cross_attn_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.cross_attn_blocks.append(CrossAttentionBlock(hidden_dim, num_heads))

        self.mlp = nn.ModuleDict()

        for key, dim in dims.items():
            self.mlp[key] = nn.Sequential(
                nn.Linear(dim, hidden_dim))


    def forward(self, graph_embeds, graph_masks):
        features_2d, features_3d = graph_embeds['moleculestm'], graph_embeds['unimol']
        mask_2d, mask_3d = graph_masks['moleculestm'], graph_masks['unimol']
        features_2d = self.mlp['moleculestm'](features_2d) * mask_2d.unsqueeze(-1)
        features_3d = self.mlp['unimol'](features_3d) * mask_3d.unsqueeze(-1)

        for cross_attn_block in self.cross_attn_blocks:
            features_2d, features_3d = cross_attn_block(features_2d, features_3d, mask_2d, mask_3d)
        
        features = torch.cat([features_2d, features_3d], dim=1)
        masks = torch.cat([mask_2d, mask_3d], dim=1)
        graph_rep_indices = [0, mask_2d.size(1)]
        return features, masks, graph_rep_indices
