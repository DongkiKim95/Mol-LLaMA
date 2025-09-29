
from utils import data_utils

class Mol3DCollater:        
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, samples):
        atom_vec = [_['src_tokens'] for _ in samples]
        edge_type = [_['src_edge_type'] for _ in samples]
        dist = [_['src_distance'] for _ in samples]
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]


        return {'src_tokens': padded_atom_vec,
            'src_edge_type': padded_edge_type,
            'src_distance': padded_dist,
        }