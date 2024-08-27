import argparse
import torch
from torch_geometric.data import InMemoryDataset
import os
from icecream import ic
class reaction_graph(InMemoryDataset):

    def __init__(self, opt: argparse.Namespace, filename: str, mol_cols: list, root: str, file) -> None:
        self.filename = filename
        self.mol_cols = mol_cols
        self._name = "BaseDataset"
        self._opt = opt
        self._root = root
        
        super().__init__(root = self._root, transform=None, pre_transform=None)

        # Load the processed data if it exists
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = self.process()

    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        return [f'{self._name}.pt']
    
    @property
    def _elem_list(self):
        elements = [
            'H', 
            'B', 
            'C', 
            'N', 
            'O', 
            'F', 
            'Si', 
            'S', 
            'Cl', 
            'Br']
        
        return elements
    
    def download(self):
        raise NotImplementedError("The download method should be implemented in a subclass")

    def process(self):
        raise NotImplementedError("The process method should be implemented in a subclass")

    def _get_node_feats(self):
        raise NotImplementedError
    
    def _get_edge_features(self):
        raise NotImplementedError

    def create_data_object(self, row):
        # This method should create and return a Data object
        raise NotImplementedError("The create_data_object method should be implemented in a subclass")

    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def _get_atom_chirality(self, CIP_dict, atom_idx):
        try:
            chirality = CIP_dict[atom_idx]
        except KeyError:
            chirality = 'No_Stereo_Center'

        return chirality
    
    def _one_h_e(self, x, allowable_set, ok_set=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                print(x)
        return list(map(lambda s: x == s, allowable_set))