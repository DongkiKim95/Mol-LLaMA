from torch.utils.data import Dataset

from data_provider.mol_dataset import MolDataset_cid

class Stage1Dataset(Dataset):
    def __init__(self, data_list, unimol_dict, encoder_type, max_atoms):
        super(Stage1Dataset, self).__init__()

        self.mol_dataset = MolDataset_cid(data_list, unimol_dict, encoder_type, max_atoms)
        self.cid2idx = self.mol_dataset.cid2idx
        self.idx2cid = {v: k for k, v in self.cid2idx.items()}    

    def __len__(self):
        # return 256
        return len(self.mol_dataset)

    def __getitem__(self, index):
        cid = self.idx2cid[index]
        data_graph, data_others = self.mol_dataset[cid]

        return data_graph, data_others['iupac_name']