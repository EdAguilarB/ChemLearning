import argparse
import os
import sys
import torch
import pandas as pd
from torch_geometric.data import Dataset
import streamlit as st



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class reaction_graph(Dataset):


    def __init__(self, opt: argparse.Namespace, filename: str, mol_cols: list, root: str, file) -> None:
        self.filename = filename
        self.mol_cols = mol_cols
        self._name = "BaseDataset"
        self._opt = opt
        self._root = root
        
        super().__init__(root = self._root)
        self.set = pd.read_csv(self.raw_paths[0])['fold']
        

    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        molecules = [f'reaction_{i}_{self._opt.split_type}_{self._opt.split_method[:2]}.pt' for i in list(self.data.index)]
        return molecules
    
    @property
    def _elem_list(self):
        elements = [
            'H', 
            'B', 
            'C', 
            'N', 
            'O', 
            'F', 
            'Na',
            'Mg',
            'Al',
            'Si', 
            'P',
            'S', 
            'Cl', 
            'K',
            'Ca',
            'Zn',
            'Br',
            'I']
        
        return elements
    
    def download(self):
        print(self.raw_paths)

    def process(self):
        raise NotImplementedError
    
    def _get_node_feats(self):
        raise NotImplementedError
    
    def _get_edge_features(self):
        raise NotImplementedError
    
    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):

        molecule = torch.load(os.path.join(self.processed_dir, 
                                f'reaction_{idx}_{self._opt.split_type}_{self._opt.split_method[:2]}.pt'), weights_only=False) 
        return molecule
    
    def _get_atom_chirality(self, CIP_dict, atom_idx):
        try:
            chirality = CIP_dict[atom_idx]
        except KeyError:
            chirality = 'No_Stereo_Center'

        return chirality
    
    def _one_h_e(self, x, allowable_set, ok_set=None, feature_name=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                st.warning(f'Value of {x} not found in the allowable set for {feature_name}. This could be because such value is no common.')
        return list(map(lambda s: x == s, allowable_set))
    
