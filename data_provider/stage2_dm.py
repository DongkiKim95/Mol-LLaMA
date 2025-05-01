import argparse
import os
import json
from collections import defaultdict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data_provider.collaters import Mol3DCollater
from data_provider.instruction_dataset import InstructionDataset
from data_provider.mol_dataset import MolDataset_cid

import torch
from torch_geometric.data import Batch
from transformers import BatchEncoding

from data_provider.tokenization_utils import batch_tokenize_messages_list



class Stage2Collater:
    def __init__(self, tokenizer, llama_version, pad_idx, encoder_types):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.encoder_types = encoder_types
        if 'unimol' in encoder_types:
            self.d3_collater = Mol3DCollater(pad_idx)

    def __call__(self, batch):
        data_graphs, messages_list, other_infos = zip(*batch)

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

        tokenized = batch_tokenize_messages_list(messages_list, self.tokenizer, 
                                                self.llama_version, padding_side='left')
        text_batch = tokenized

        other_infos_ = defaultdict(list)
        for key in other_infos[0].keys():
            for info in other_infos:
                other_infos_[key].append(info[key])

        return graph_batch, text_batch, other_infos_


class Stage2DM(LightningDataModule):
    def __init__(
            self,
            tokenizer,
            llama_version,
            num_workers,
            batch_size,
            root,
            unimol_dictionary,
            encoder_types,
            data_types,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types
        
        print('Loading molecule data...')
        data_list = json.load(open(root + 'pubchem-molecules.json'))
        mol_dataset = MolDataset_cid(data_list, unimol_dictionary, encoder_types)
        json_paths = [os.path.join(root, f'{data_type}.json') for data_type in data_types]

        self.train_dataset = InstructionDataset(
            json_paths=json_paths,
            mol_dataset = mol_dataset
        )
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            pin_memory=False,
                            drop_last=True,
                            persistent_workers=True,
                            collate_fn=Stage2Collater(self.tokenizer,
                                                    self.llama_version,
                                                    self.unimol_dictionary.pad(),
                                                    self.encoder_types)
                            )
        return loader


if __name__ == '__main__':
    from unicore.data import Dictionary
    from transformers import AutoTokenizer
    from data_provider.tokenizer import MolLLaMATokenizer

    dictionary = Dictionary.load('./all_checkpoints/Mol-LLaMA3.1/unimol/unimol_dict.txt')
    dictionary.add_symbol("[MASK]", is_special=True)

    # tokenizer = AutoTokenizer.from_pretrained('all_checkpoints/Llama-3.1-8B-Instruct')
    # llama_version = 'llama3'

    tokenizer = AutoTokenizer.from_pretrained('all_checkpoints/Llama-2-7b-chat-hf')
    llama_version = 'llama2'

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ["<mol>"]})
    tokenizer.mol_token_ids = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    tokenizer = MolLLaMATokenizer(tokenizer, llama_version=llama_version)

    dm = Stage2DM(
        num_workers=0,
        batch_size=4,
        root='data/Mol-LLaMA-Instruct/',
        unimol_dictionary=dictionary,
        encoder_types=['moleculestm', 'unimol'],
        data_types=['comprehensive_conversations'],
        # data_types = ['detailed_structural_descriptions'],
        # data_types = ['structure2chemical_features_relationships'],
        # data_types = ['structure2biological_features_relationships'],
        tokenizer=tokenizer,
    )

    train_loader = dm.train_dataloader()
    for batch in train_loader:
        import pdb; pdb.set_trace()
