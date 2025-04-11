import torch
import contextlib
import pytorch_lightning as pl
from torch import optim
from trainer.optims import LinearWarmupCosineLRScheduler
from tqdm import tqdm
from typing import Any, Dict
import os

from models.mol_llama_encoder import MolLLaMAEncoder

def precision2dtype(precision):
    if precision == '16':
        return torch.float16
    elif precision == '32':
        return torch.float32
    elif precision.find('bf16') >= 0:
        return torch.bfloat16
    else:
        raise NotImplementedError()


class Stage1Trainer(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.train_config = train_config
        
        self.encoder = MolLLaMAEncoder(
            graph_encoder_config = model_config.graph_encoder_config,
            blending_module_config = model_config.blending_module_config,
            qformer_config = model_config.qformer_config,
            temperature = train_config.temperature,
            tune_gnn = train_config.tune_gnn,
        )
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, iupac_names = batch
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        batch_size = text_batch.input_ids.size(0)
        loss = self.encoder.compute_loss(graph_batch, text_batch)

        ###============== Overall Loss ===================###
        self.log("train_loss_gtc", float(loss['loss_gtc']), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_gtm", float(loss['loss_gtm']), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_lm", float(loss['loss_lm']), batch_size=batch_size, sync_dist=True)
        self.log("train_loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        graph_batch, text_batch, iupac_names = batch
        batch_size = text_batch.input_ids.size(0)
        loss = self.encoder.compute_loss(graph_batch, text_batch)
        ###============== Overall Loss ===================###
        self.log("val_loss_gtc", float(loss['loss_gtc']), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_gtm", float(loss['loss_gtm']), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_lm", float(loss['loss_lm']), batch_size=batch_size, sync_dist=True)
        self.log("val_loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)