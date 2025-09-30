# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from data_provider.stage1_dataset import Stage1Dataset
from torch.utils.data import DataLoader
from data_provider.collaters import Mol3DCollater
from torch_geometric.data import Batch
import json

class Stage1Collater:
    def __init__(self, tokenizer, text_max_len, pad_idx, encoder_types):
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.encoder_types = encoder_types

    def __call__(self, batch):
        data_graphs, iupac_names = zip(*batch)

        graph_batch = {}
        if 'unimol' in self.encoder_types:
            data_unimol = []
            for data in data_graphs:
                data_unimol.extend(data['unimol'])
            graph_batch['unimol'] = self.d3_collater(data_unimol)
        if 'moleculestm' in self.encoder_types:
            data_moleculestm = []
            for data in data_graphs:
                data_moleculestm.extend(data['moleculestm'])
            graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)

        text_batch = self.tokenizer(iupac_names,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.text_max_len,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        return_attention_mask=True, 
                                        return_token_type_ids=False)
    
        return graph_batch, text_batch, iupac_names

class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        unimol_dictionary=None,
        scibert_tokenizer=None,
        encoder_types=None,
        text_max_len=512,
        unimol_max_atoms=512,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unimol_dictionary = unimol_dictionary
        self.scibert_tokenizer = scibert_tokenizer
        self.text_max_len = text_max_len
        self.unimol_max_atoms = unimol_max_atoms
        self.encoder_types = encoder_types

        print('Loading molecule data...')
        data_list = json.load(open(root + 'pubchem-molecules.json'))
        train_data_list = [data for data in data_list if data['split'] == 'pretrain']
        val_data_list = [data for data in data_list if data['split'] == 'valid']
        self.train_dataset = Stage1Dataset(train_data_list, unimol_dictionary, 
                                        encoder_types, unimol_max_atoms)
        self.val_dataset = Stage1Dataset(val_data_list, unimol_dictionary, 
                                            encoder_types, unimol_max_atoms)


    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=Stage1Collater(self.scibert_tokenizer,
                                 self.text_max_len, 
                                 self.unimol_dictionary.pad(),
                                 self.encoder_types)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=Stage1Collater(self.scibert_tokenizer,
                                 self.text_max_len, 
                                 self.unimol_dictionary.pad(),
                                 self.encoder_types)
        )

        return loader