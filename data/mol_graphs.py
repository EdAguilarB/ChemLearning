import streamlit as st
import torch
from rdkit import Chem
import numpy as np

class molecular_graphs:
    def __init__(self, data, opt):
        self.data = data
        self._opt = opt

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
    
    def process(self):
        raise NotImplementedError
    
    def _get_node_feats(self, mol, graph_feat=None):

        all_node_feats = []
        CIPtuples = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=False))

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list, None, 'Atomic Symbol')
            # Feature 2: Atom degree
            node_feats += self._one_h_e(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6], None, 'Atom Degree')
            # Feature 3: Hybridization
            node_feats += self._one_h_e(atom.GetHybridization(), [0,1,2,3,4,5,6], None, 'Atom Hybridization')
            # Feature 4: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 5: In Ring
            node_feats += [atom.IsInRing()]
            # Feature 6: Chirality
            node_feats += self._one_h_e(self._get_atom_chirality(CIPtuples, atom.GetIdx()), ['R', 'S'], 'No_Stereo_Center', 'Atom Chirality')

            # Graph level features
            if graph_feat is not None:
                node_feats += graph_feat

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats, dtype=np.float32)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):

        all_edge_feats = []
        edge_indices = []

        for bond in mol.GetBonds():

            #list to save the edge features
            edge_feats = []

            # Feature 1: Bond type (as double)
            edge_feats += self._one_h_e(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])

            #feature 2: double bond stereochemistry
            edge_feats += self._one_h_e(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE], Chem.rdchem.BondStereo.STEREONONE)

            # Feature 3: Is in ring
            edge_feats.append(bond.IsInRing())

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            #create adjacency list
            edge_indices += [[i, j], [j, i]]

        all_edge_feats = np.asarray(all_edge_feats)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return torch.tensor(all_edge_feats, dtype=torch.float), edge_indices

    def _get_atom_chirality(self, CIP_dict, atom_idx):
        try:
            chirality = CIP_dict[atom_idx]
        except KeyError:
            chirality = 'No_Stereo_Center'

        return chirality
    

    def onek_encoding_unk(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
    
    def _one_h_e(self, x, allowable_set, ok_set=None, feature_name=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                st.warning(f'Value of {x} not found in the allowable set for {feature_name}. This could be because such value is no common.')
        return list(map(lambda s: x == s, allowable_set))