"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

from unicore.data import Dictionary

from models.unimol.unimol import SimpleUniMolModel
from models.moleculestm.moleculestm import MoleculeSTM
from models.blending_module.blending_module import BlendingModule
from models.qformer.modeling_bert import BertConfig, BertLMHeadModel

from utils.dist_funs import pl_concat_all_gather

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MolLLaMAEncoder(nn.Module):
    def __init__(
        self,
        graph_encoder_config,
        blending_module_config,
        qformer_config,
        temperature=0.1,
        tune_gnn=False,
    ):
        super().__init__()
        self.num_query_tokens = qformer_config.num_query_tokens

        self.tune_gnn = tune_gnn
        self.encoder_types = graph_encoder_config.encoder_types

        # Initialize graph encoders
        self.graph_encoder, self.ln_graph = {}, {}
        for encoder_type in self.encoder_types:
            graph_encoder, ln_graph = \
                self.init_graph_encoder(encoder_type, graph_encoder_config, tune_gnn)

            self.graph_encoder[encoder_type] = graph_encoder
            self.ln_graph[encoder_type] = ln_graph
        
        self.graph_encoder = nn.ModuleDict(self.graph_encoder)
        self.ln_graph = nn.ModuleDict(self.ln_graph)

        # Initialize blending module
        dims = {encoder_type: graph_encoder.num_features \
                for encoder_type, graph_encoder in self.graph_encoder.items()}
        hidden_size = max([v for k, v in dims.items()])
        self.blending_module = BlendingModule(
            hidden_dim=hidden_size,
            num_heads=blending_module_config.num_heads,
            num_layers=blending_module_config.num_layers,
            dims=dims
        )

        # Initialize Qformer
        self.Qformer, self.query_tokens, self.scibert_tokenizer = \
                    self.init_Qformer(qformer_config, hidden_size)

        # Initialize Projectors, Not be used for stage2 training
        self.graph_proj, self.text_proj, self.gtm_head = \
                    self.init_projectors(self.Qformer.config.hidden_size, qformer_config.embed_dim)

        self.temperature = temperature

    def init_graph_encoder(self, encoder_type, graph_encoder_config, tune_gnn):
        if encoder_type == 'unimol':
            graph_encoder, ln_graph, unimol_dictionary = \
                    self.init_unimol_encoder(graph_encoder_config.unimol_config)
            self.unimol_dictionary = unimol_dictionary
        elif encoder_type == 'moleculestm':
            graph_encoder, ln_graph = \
                    self.init_moleculestm_encoder(graph_encoder_config.moleculestm_config)

        if not tune_gnn:
            for name, param in graph_encoder.named_parameters():
                param.requires_grad = False
            graph_encoder = graph_encoder.eval()
            graph_encoder.train = disabled_train
            logging.info(f"freeze {encoder_type} encoder")

        return graph_encoder, ln_graph
    
    def init_unimol_encoder(self, unimol_config):

        unimol_dictionary_path = hf_hub_download(
            repo_id=unimol_config.repo_id,
            filename=unimol_config.dictionary_filename,
        )
        unimol_dictionary = Dictionary.load(unimol_dictionary_path)
        unimol_dictionary.add_symbol("[MASK]", is_special=True)

        unimol_model = SimpleUniMolModel(dictionary=unimol_dictionary)

        unimol_ckpt_path = hf_hub_download(
            repo_id=unimol_config.repo_id,
            filename=unimol_config.weights_filename,
        )
        ckpt = torch.load(unimol_ckpt_path, map_location='cpu', weights_only=True)
        missing_keys, unexpected_keys = unimol_model.load_state_dict(ckpt['model'], strict=False)
        
        ln_graph = nn.LayerNorm(unimol_model.num_features)
        
        return unimol_model, ln_graph, unimol_dictionary
        
    def init_moleculestm_encoder(self, moleculestm_config):
        moleculestm_model = MoleculeSTM()
        moleculestm_ckpt_path = hf_hub_download(
            repo_id = moleculestm_config.repo_id,
            filename = moleculestm_config.filename,
        )
        moleculestm_ckpt = torch.load(moleculestm_ckpt_path, map_location='cpu', weights_only=True)
        moleculestm_model.load_state_dict(moleculestm_ckpt, strict=True)

        moleculestm_model.num_features = moleculestm_model.emb_dim
        ln_graph = nn.LayerNorm(moleculestm_model.num_features)

        return moleculestm_model, ln_graph

    def init_Qformer(self, qformer_config, hidden_size):
        bert_name = qformer_config.bert_name
        num_query_tokens = qformer_config.num_query_tokens
        cross_attention_freq = qformer_config.cross_attention_freq

        tokenizer = BertTokenizer.from_pretrained(bert_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        encoder_config = BertConfig.from_pretrained(bert_name)
        encoder_config.encoder_width = hidden_size
        
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_tokens

        Qformer = BertLMHeadModel.from_pretrained(
            bert_name, config=encoder_config
        )

        Qformer.resize_token_embeddings(len(tokenizer))
        state_dict = Qformer.state_dict()
        for name, param in Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens, tokenizer

    def init_projectors(self, qformer_hidden_size, embed_dim):
        graph_proj = nn.Linear(qformer_hidden_size, embed_dim)
        text_proj = nn.Linear(qformer_hidden_size, embed_dim)
        gtm_head = nn.Linear(qformer_hidden_size, 2)

        return graph_proj, text_proj, gtm_head

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def compute_loss(self, graph_batch, text_batch):
        batch_size = text_batch.input_ids.shape[0]

        batch_node, batch_mask, query_output = self.graph_forward(graph_batch)
        
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)

        text_feats = self.text_forward(text_batch.input_ids, text_batch.attention_mask)

        text_feat_all = pl_concat_all_gather(text_feats)
        graph_feat_all = pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats,
                                                        graph_feat_all, text_feat_all, 
                                                        return_sim=True)
        loss_gtm = self.molecule_text_matching(sim_t2g, sim_g2t, batch_node, batch_mask,
                                            text_batch, batch_size)
        loss_lm = self.molecule_captioning(text_batch, query_output, batch_size)

        return {
            "loss": loss_gtc + loss_gtm + loss_lm,
            "loss_gtc": loss_gtc,
            "loss_gtm": loss_gtm,
            "loss_lm": loss_lm,
        }

    def graph_forward(self, graph_batch):
        batch_nodes, batch_masks = {}, {}
        for encoder_type in self.encoder_types:
            batch_node, batch_mask = self.graph_encoder[encoder_type](**graph_batch[encoder_type])
            batch_node = self.ln_graph[encoder_type](batch_node)

            batch_nodes[encoder_type] = batch_node
            batch_masks[encoder_type] = batch_mask

        batch_node, batch_mask, _ = self.blending_module(batch_nodes, batch_masks)

        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            use_cache=True,
            return_dict=True,
        )

        return batch_node, batch_mask, query_output

    def text_forward(self, input_ids, mask):
        text_output = self.Qformer.bert(input_ids, attention_mask=mask, return_dict=True)
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats

    

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(dim=-1) # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(dim=-2) # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss

    def molecule_text_matching(self, sim_t2g, sim_g2t, batch_node, batch_mask,
                            text_batch, batch_size):
        ## not aggregate global tensor because of their different shapes
        g_emb_world = batch_node
        g_mask_world = batch_mask
        text_ids_world = text_batch.input_ids
        text_mask_world = text_batch.attention_mask
        with torch.no_grad():
            weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
            weights_t2g.fill_diagonal_(0)
            weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
            weights_g2t.fill_diagonal_(0)

        # select a negative graph for each text
        graph_embeds_neg = []
        graph_mask_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2g[b], 1).item()
            graph_embeds_neg.append(g_emb_world[neg_idx])
            graph_mask_neg.append(g_mask_world[neg_idx])
        
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
        graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_g2t[b], 1).item()
            text_ids_neg.append(text_ids_world[neg_idx])
            text_atts_neg.append(text_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_batch.input_ids, text_batch.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_batch.attention_mask, text_batch.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=self.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
        graph_atts_all = torch.cat([batch_mask, graph_mask_neg, batch_mask], dim=0)

        output_gtm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=graph_embeds_all,
            encoder_attention_mask=graph_atts_all,
            return_dict=True,
        )

        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
        gl_output = self.gtm_head(gl_embeddings)
        logits = gl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
            dim=0,
        ).to(self.device)
        loss_gtm = F.cross_entropy(logits, itm_labels)

        return loss_gtm


    def molecule_captioning(self, text_batch, query_output, batch_size):
        decoder_input_ids = text_batch.input_ids.clone()
        decoder_input_ids[:, 0] = self.scibert_tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.scibert_tokenizer.pad_token_id, -100
        )
        query_atts = torch.ones((batch_size, self.num_query_tokens), dtype=torch.long, device=self.device)
        
        attention_mask = torch.cat([query_atts, text_batch.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return loss_lm
