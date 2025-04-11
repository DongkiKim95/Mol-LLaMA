import os
import torch
import warnings
import argparse
import yaml
from easydict import EasyDict as edict

import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger


from utils.configuration_mol_llama import MolLLaMAConfig
from utils.dist_funs import MyDeepSpeedStrategy
from trainer.stage1 import Stage1Trainer
from data_provider.stage1_dm import Stage1DM


## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A100 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)

def edict_to_dict(config):
    """
    Convert an EasyDict object to a regular dictionary.
    """
    if isinstance(config, edict):
        return {k: edict_to_dict(v) for k, v in config.items()}
    else:
        return config


def main(model_config, train_config, data_config):
    pl.seed_everything(0)

    model = Stage1Trainer(model_config, train_config)
    
    args = {'train': edict_to_dict(train_config), 
            'model': edict_to_dict(model_config), 
            'data': edict_to_dict(data_config)}
    model.save_hyperparameters(args)

    dm = Stage1DM(
        num_workers = data_config.num_workers, 
        batch_size = data_config.batch_size, 
        root = data_config.root,
        unimol_dictionary = model.encoder.unimol_dictionary, 
        scibert_tokenizer = model.encoder.scibert_tokenizer, 
        encoder_types = model_config.graph_encoder_config.encoder_types, 
        text_max_len = data_config.text_max_len,
    )
                    
    callbacks = [
        plc.ModelCheckpoint(dirpath="checkpoints/"+train_config.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=train_config.save_every_n_epochs, 
                                         save_top_k=-1,
                                         save_last=True,
                                         save_on_train_epoch_end=True)
    ]
    
    if len(train_config.devices) > 1:
        if train_config.strategy_name == 'deepspeed':
            strategy = MyDeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(start_method='spawn', find_unused_parameters=find_unused_parameters)
    else:
        strategy = 'auto'
        
    logger = CSVLogger(save_dir=f'./checkpoints/{train_config.filename}/')
    trainer = Trainer(
        accelerator=train_config.accelerator,
        devices=train_config.devices,
        precision=train_config.precision,
        max_epochs=train_config.max_epochs,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        check_val_every_n_epoch=train_config.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=strategy,
        logger=logger,
        limit_val_batches=10,
    )

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1 training')
    parser.add_argument('--train_config_path', type=str, default='configs/stage1/train_config.yaml')
    parser.add_argument('--data_config_path', type=str, default='configs/stage1/data_config.yaml')

    args = parser.parse_args()

    model_config = MolLLaMAConfig()
    data_config = edict(yaml.load(open(args.data_config_path), Loader=yaml.FullLoader))
    train_config = edict(yaml.load(open(args.train_config_path), Loader=yaml.FullLoader))

    print('-'*60)
    print(f'batch_size: {data_config.batch_size}\tnum_devices: {len(train_config.devices)}\taccumulate_grad_batches: {train_config.accumulate_grad_batches}')
    print(f'Total batch size: {data_config.batch_size * len(train_config.devices) * train_config.accumulate_grad_batches}')
    print('-'*60)
    
    main(model_config, train_config, data_config)