import torch
from collections import defaultdict

import numpy as np
from scipy.spatial import distance_matrix

from torch.utils.data import Dataset
from torch_geometric.data import Data

from data_provider.ogb_features import atom_to_feature_vector, bond_to_feature_vector

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object

    Adopted from OGB 
    """

    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = torch.from_numpy(edge_index)
    graph['edge_feat'] = torch.from_numpy(edge_attr)
    graph['node_feat'] = torch.from_numpy(x)
    graph['num_nodes'] = len(x)

    return graph 

def get_unimol_data(atoms, coordinates, dictionary, max_atoms=512, remove_Hs=True):
    atoms = np.array(atoms)
    assert len(atoms) == len(coordinates) and len(atoms) > 0
    assert coordinates.shape[1] == 3

    ## Remove Hydrogen atoms
    if remove_Hs:
        mask_hydrogen = atoms != "H"
        if sum(mask_hydrogen) > 0:
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

    ## Randomly sample atoms if the number of atoms is larger than max_atoms
    if len(atoms) > max_atoms:
        index = np.random.permutation(len(atoms))[:max_atoms]
        atoms = atoms[index]
        coordinates = coordinates[index]

    assert 0 < len(atoms) <= max_atoms

    atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()
    atom_vec = torch.cat([torch.LongTensor([dictionary.bos()]), 
                            atom_vec, 
                            torch.LongTensor([dictionary.eos()])])

    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

    ## obtain edge types; which is defined as the combination of two atom types
    edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
    dist = distance_matrix(coordinates, coordinates).astype(np.float32)
    coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

    return {'src_tokens': atom_vec, 'src_edge_type': edge_type, 'src_distance': dist}

class MolDataset_cid(Dataset):
    def __init__(self, data_list, dictionary, encoder_types, max_atoms=512):
        self.dictionary = dictionary
        
        self.data_list = data_list
        self.cid2idx = {data['cid']:idx for idx, data in enumerate(self.data_list)}
        self.encoder_types = encoder_types
        
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, cid):
        idx = self.cid2idx[cid]
        data = self.data_list[idx]
        data_atoms = [data['atoms']]
        data_coordinates = data['coordinates']
        smiles_list = [data['smiles']]
    
        data_graph = defaultdict(list)
        if 'unimol' in self.encoder_types:
            for atoms, coordinates in zip(data_atoms, data_coordinates):
                atoms = np.array(atoms)
                coordinates = np.array(coordinates)
                data_graph['unimol'].append(
                    get_unimol_data(atoms, coordinates, self.dictionary, 
                                        self.max_atoms, remove_Hs=True))
        if 'moleculestm' in self.encoder_types:
            for smiles in smiles_list:
                graph = smiles2graph(smiles)
                data_graph['moleculestm'].append(Data(x=graph['node_feat'], 
                                        edge_index=graph['edge_index'], 
                                        edge_attr=graph['edge_feat']))

        return data_graph, data