import json
from torch.utils.data import Dataset
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from collections import defaultdict
from torch_geometric.data import Data, Batch
from data_provider.tokenization_utils import batch_tokenize_messages_list
from data_provider.collaters import Mol3DCollater
import numpy as np

class PAMPADataset(Dataset):
    def __init__(self, json_path, split, prompt_type, 
                unimol_dictionary):
        super().__init__()
        assert prompt_type in ['default', 'rationale', 'task_info'], "prompt_type must be one of ['default', 'rationale', 'task_info']"

        self.pampa_data = json.load(open(json_path, 'r'))
        self.prompts = self.pampa_data['prompts'][prompt_type]

        self.data_list = self.pampa_data['data_list']
        self.split = self.pampa_data['split']

        self.data_list = [self.data_list[i] for i in self.split[split]]

        self.mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"

        self.unimol_dictionary = unimol_dictionary

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        data_graphs = defaultdict(list)
        data_graphs['unimol'].append(
            get_unimol_data(data['atoms'], np.array(data['coordinates'][0]), self.unimol_dictionary))

        graphs = smiles2graph(data['smiles'])
        data_graphs['moleculestm'].append(
            Data(x=graphs['node_feat'], 
                edge_index=graphs['edge_index'], 
                edge_attr=graphs['edge_feat'])
        )

        # Prepare the prompt
        system_prompt = self.prompts['system']
        user_prompt = self.prompts['user'].replace('<mol>', self.mol_prompt)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return data_graphs, messages, data['answer'], data['smiles']

class PAMPACollater():
    def __init__(self, tokenizer, unimol_dictionary, llama_version):
        self.tokenizer = tokenizer
        self.llama_version = llama_version
        self.d3_collater = Mol3DCollater(unimol_dictionary.pad())
        
    def __call__(self, batch):
        data_graphs, messages_list, answers, smiles = zip(*batch)

        graph_batch = {}
        data_unimol = []
        for data in data_graphs:
            data_unimol.extend(data['unimol'])
        graph_batch['unimol'] = self.d3_collater(data_unimol)
        data_moleculestm = []
        for data in data_graphs:
            data_moleculestm.extend(data['moleculestm'])
        graph_batch['moleculestm'] = Batch.from_data_list(data_moleculestm)

        tokenized = batch_tokenize_messages_list(messages_list, self.tokenizer, 
                                                self.llama_version, padding_side='left')
        text_batch = tokenized

        return graph_batch, text_batch, answers, smiles

    