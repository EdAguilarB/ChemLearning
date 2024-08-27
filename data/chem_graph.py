import argparse
import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
import os
from molvs import standardize_smiles
from data.graph_generator import reaction_graph
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from icecream import ic

class molecular_graph(reaction_graph):

    def __init__(self, opt: argparse.Namespace, filename: str, molcols: list, root: str = None, file=None, include_fold=True) -> None:
        self._include_fold = include_fold

        if self._include_fold:
            columns = file.columns
            if 'fold' not in columns:
                self.split_data(root, filename, file, opt)
            
            root = os.path.join(root, opt.experiment_name, 'data')

        super().__init__(opt=opt, filename=self.filename, mol_cols=molcols, root=root, file=file)

        self._name = "rhcaa_diene"

    
    def process(self):
        # Load your data here (e.g., from raw files) and process it into a list of Data objects
        data_list = []
        st.text('Generating Graph Representation of the Molecules...')
        progress_bar = st.progress(0)
        database = pd.read_csv(self.raw_paths[0])
        for i, row in database.iterrows():
            # Create a Data object for each row
            data = self.create_data_object(row)
            data_list.append(data)

            progress_bar.progress((i + 1) / database.shape[0])

        st.success('Graph Representation of the Molecules Generated')
        
        # Use InMemoryDataset's collate function to batch data
        data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_paths[0])
        return data, slices

    def create_data_object(self, row):
        # This method should create and return a Data object
        node_feats_mol = None
        all_smiles = []
        temp = row['temp'] / 100

        for reactant in self.mol_cols:  
            # Create a molecule object from the SMILES string
            std_smiles = standardize_smiles(row[reactant])
            all_smiles.append(std_smiles)
            mol = Chem.MolFromSmiles(std_smiles)
            mol = Chem.rdmolops.AddHs(mol)
            node_feats = self._get_node_feats(mol, row['Confg'], reactant, temp)
            edge_attr, edge_index = self._get_edge_features(mol)

            # Accumulate features
            if node_feats_mol is None:
                node_feats_mol = node_feats
                edge_idx_mol = edge_index
                edge_attr_mol = edge_attr

            else:
                node_feats_mol = torch.cat([node_feats_mol, node_feats], dim=0)
                edge_attr_mol = torch.cat([edge_attr_mol, edge_attr], axis=0)
                edge_index += max(edge_idx_mol[0]) + 1
                edge_idx_mol = torch.cat([edge_idx_mol, edge_index], axis=1)

        y = torch.tensor(row[self._opt.target_variable]).reshape(1)

        # Create the graph data object
        data = Data(x=node_feats_mol, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr,
                    y=y, 
                    smiles=all_smiles, 
                    fold = row['fold'])
        
        return data

    def _get_node_feats(self, mol, mol_confg, reactant, temperature):

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
            #feature 7: ligand configuration
            if reactant == 'Ligand':
                node_feats += self._one_h_e(mol_confg, [2, 1], 0)
            else:
                node_feats += [0,0]
            # feature 8: reaction temperature
            node_feats += [temperature]

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



    def split_data(self, root, filename, file, opt):


        if opt.problem_type == 'regression':  
            file['category'] = pd.cut(file[opt.target_variable], bins=5, labels=False)

        elif opt.problem_type == 'classification':
            file['category'] = file[opt.target_variable]


        if opt.split_type == 'tvt':

            val_size = len(file) * opt.val_size
            val_ratio = val_size / (len(file) * (1-opt.test_size))

            if opt.split_method == 'stratified':
                train_val_df, test_df = train_test_split(file, test_size=opt.test_size, stratify=file['category'], random_state=opt.global_seed)
                train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, stratify=train_val_df['category'], random_state=opt.global_seed)
            
            elif opt.split_method == 'random':
                train_val_df, test_df = train_test_split(file, test_size=opt.test_size, random_state=opt.global_seed)
                train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=opt.global_seed)


            file.loc[train_df.index, 'set'] = 'train'
            file.loc[val_df.index, 'set'] = 'val'
            file.loc[test_df.index, 'set'] = 'test'

            file.loc[train_df.index, 'fold'] = 2
            file.loc[val_df.index, 'fold'] = 1
            file.loc[test_df.index, 'fold'] = 0

            # Optional: Verify the distribution
            print(file['set'].value_counts())
            print(file.groupby('set')[opt.target_variable].describe())


        elif opt.split_type == 'cv' or opt.split_type == 'ncv': 

            if opt.split_method == 'stratified':
                folds = StratifiedKFold(n_splits = opt.folds, shuffle = True, random_state=opt.global_seed)
            
            elif opt.split_method == 'random':
                folds = KFold(n_splits = opt.folds, shuffle = True, random_state=opt.global_seed)

            test_idx = []

            for _, test in folds.split(np.zeros(len(file)), file['category']):
                test_idx.append(test)

            index_dict = {index: list_num for list_num, index_list in enumerate(test_idx) for index in index_list}

            file['fold'] = file.index.map(index_dict)

        
        filename = filename[:-4] + f'_{opt.split_type}_{opt.split_method[:2]}' + filename[-4:]

        self.filename = filename


        os.makedirs(os.path.join(root, opt.experiment_name,'data', 'raw'), exist_ok=True)

        file.to_csv(os.path.join(root, opt.experiment_name, 'data','raw', filename))

        print('{}.csv file was saved in {}'.format(filename, os.path.join(root, 'raw')))
