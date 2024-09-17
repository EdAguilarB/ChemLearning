import argparse
import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import  Data
import numpy as np 
from rdkit import Chem
import os
from tqdm import tqdm
from molvs import standardize_smiles
import sys
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


from icecream import ic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class predict_insilico():

    def __init__(self, data):
        self.data = data


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

    def process(self, opt):

        all_graphs = []

        st.text("Generating graph representation of the in silico library...")

        progress_bar = st.progress(0)

        for index, mols in self.data.iterrows():

            node_feats_mols = None
            all_smiles = []

            for molecule in opt.mol_cols:


                std_smiles = standardize_smiles(mols[molecule])

                all_smiles.append(std_smiles)

                mol = Chem.MolFromSmiles(std_smiles)

                mol = Chem.rdmolops.AddHs(mol)

                global_features = []

                for global_feat in opt.graph_features.keys():


                    if molecule in opt.graph_features[global_feat]:


                        if global_feat in opt.ohe_graph_feat:

                            uni_vals = self.data[global_feat].unique()

                            global_features += self._one_h_e(mols[global_feat], uni_vals)

                        else:
                            global_features += [mols[global_feat]]

                    else:


                        if global_feat in opt.ohe_graph_feat:
                            uni_vals = self.data[global_feat].unique()
                            global_features += self._one_h_e('qWeRtYuIoP', uni_vals)
                        else:
                            global_features += [0]

                
                node_feats = self._get_node_feats(mol, global_features)

                edge_attr, edge_index = self._get_edge_features(mol)

                if node_feats_mols is None:
                    node_feats_mols = node_feats
                    edge_index_mols = edge_index
                    edge_attr_mols = edge_attr

                else:
                    node_feats_mols = torch.cat([node_feats_mols, node_feats], axis=0)
                    edge_attr_mols = torch.cat([edge_attr_mols, edge_attr], axis=0)
                    edge_index += max(edge_index_mols[0]) + 1
                    edge_index_mols = torch.cat([edge_index_mols, edge_index], axis=1)

            
            if opt.target_variable in self.data.columns:
                y = torch.tensor(mols[opt.target_variable]).reshape(1)
            else:
                y = None

            if opt.mol_id_col_insilico is not None:
                idx = mols[opt.mol_id_col_insilico]

            data = Data(x=node_feats_mols,
                        edge_index=edge_index_mols,
                        edge_attr=edge_attr_mols,
                        y=y,
                        smiles = all_smiles,
                        idx = idx
                        )
            
            all_graphs.append(data)

            progress_bar.progress(index / self.data.shape[0])


        return all_graphs
                        

    def _get_node_feats(self, mol, graph_feat):

        all_node_feats = []
        CIPtuples = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=False))

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list)
            # Feature 2: Atom degree
            node_feats += self._one_h_e(atom.GetDegree(), [1, 2, 3, 4])
            # Feature 3: Hybridization
            node_feats += self._one_h_e(atom.GetHybridization(), [0,2,3,4])
            # Feature 4: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 5: In Ring
            node_feats += [atom.IsInRing()]
            # Feature 6: Chirality
            node_feats += self._one_h_e(self._get_atom_chirality(CIPtuples, atom.GetIdx()), ['R', 'S'], 'No_Stereo_Center')

            # Graph level features
            node_feats += graph_feat

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats, dtype=np.float32)
        return torch.tensor(all_node_feats, dtype=torch.float)
    

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)
    
    
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
    
    def _one_h_e(self, x, allowable_set, ok_set=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                pass
                #print(x)
        return list(map(lambda s: x == s, allowable_set))
    

    