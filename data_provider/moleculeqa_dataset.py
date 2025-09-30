import os
import json
from torch.utils.data import Dataset
from data_provider.mol_dataset import MolDataset_cid
from datasets import load_dataset

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data_provider.collaters import Mol3DCollater
from torch_geometric.data import Batch
from data_provider.tokenization_utils import batch_tokenize_messages_list

from collections import defaultdict


class MoleculeQADataset(Dataset):
    def __init__(self, json_path, mol_path, unimol_dict, encoder_type, 
                mol_type='mol', max_atoms=512, do_infer=False):
        super(MoleculeQADataset, self).__init__()

        data_list = json.load(open(mol_path, 'r'))
        self.mol_dataset = MolDataset_cid(data_list, unimol_dict, encoder_type, max_atoms)
        self.instruction_dataset = load_dataset("json", data_files=[json_path])['train'] 
        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
        self.graph_prompt = "<graph>" * 28
        self.mol_type = mol_type
        self.do_infer = do_infer
    

    def __len__(self):
        return len(self.mol_dataset)

    def __getitem__(self, index):
        text_data = self.instruction_dataset[index]

        cid = text_data['cid']
        data_graphs, data_others = self.mol_dataset[cid]
        num_mols = len(data_graphs[list(data_graphs.keys())[0]])

        messages = []
        messages.append({"role": "system", "content": text_data['system']})

        assert len(text_data['conversations']) == 1, "Only one conversation is expected in the dataset."

        for turn in text_data['conversations']:
            if self.mol_type == 'mol':
                user_prompt = turn['user'].replace('<mol>', self.mol_prompt)
            elif self.mol_type == 'SMILES':
                user_prompt = turn['user'].replace('<mol>', text_data['smiles'])
            elif self.mol_type == "SMILES,mol":
                user_prompt = turn['user'].replace('<mol>', text_data['smiles'][:96] + ' ' + self.mol_prompt)
            elif self.mol_type == 'SELFIES':
                user_prompt = turn['user'].replace('<mol>', text_data['selfies'])
            elif self.mol_type == "SMILES,<graph>":
                user_prompt = turn['user'].replace('<mol>', text_data['smiles'][:128] + ' ' + self.graph_prompt)
            # user_prompt += '\nPlease think step by step and give me the final answer.'
            messages.append({"role": "user", "content": user_prompt})
            if not self.do_infer:
                messages.append({"role": "assistant", "content": turn['assistant']})
            answer = turn['assistant']

        other_info = {
            "cid": cid,
            "task": text_data['category'],
            "answer": answer,
            "smiles": text_data['smiles'],
        }

        return data_graphs, messages, other_info

class MoleculeQACollater:
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


class MoleculeQADM(LightningDataModule):
    def __init__(
            self,
            tokenizer,
            llama_version,
            num_workers,
            batch_size,
            root,
            unimol_dictionary,
            encoder_types,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unimol_dictionary = unimol_dictionary
        self.encoder_types = encoder_types

        self.train_dataset = MoleculeQADataset(
            os.path.join(root, 'train.json'),
            os.path.join(root, 'train_mol.json'),
            unimol_dictionary,
            encoder_types,
            do_infer=False
        )

        self.test_dataset = MoleculeQADataset(
            os.path.join(root, 'test.json'),
            os.path.join(root, 'test_mol.json'),
            unimol_dictionary,
            encoder_types,
            do_infer=True,
        )

        self.val_dataset = MoleculeQADataset(
            os.path.join(root, 'test.json'),
            os.path.join(root, 'test_mol.json'),
            unimol_dictionary,
            encoder_types,
            do_infer=True,
        )
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=0, # self.num_workers,
                            pin_memory=False,
                            drop_last=True,
                            persistent_workers=False, # True,
                            collate_fn=MoleculeQACollater(self.tokenizer, self.llama_version, 
                                                          self.unimol_dictionary.pad(), self.encoder_types)
                            )
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset,
                            batch_size=self.batch_size*4,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=False,
                            drop_last=False,
                            persistent_workers=True,
                            collate_fn=MoleculeQACollater(self.tokenizer, self.llama_version, 
                                                          self.unimol_dictionary.pad(), self.encoder_types)
                            )
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size*4,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=False,
                            drop_last=False,
                            persistent_workers=True,
                            collate_fn=MoleculeQACollater(self.tokenizer, self.llama_version, 
                                                          self.unimol_dictionary.pad(), self.encoder_types)
                            )
        return loader

