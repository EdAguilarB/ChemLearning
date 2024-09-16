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
from data.datasets import reaction_graph
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


from icecream import ic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class rhcaa_diene(reaction_graph):

    def __init__(self, opt:argparse.Namespace, filename: str, molcols: list, root: str = None, file=None, include_fold = True) -> None:

        self._include_fold = include_fold

        self.mol_identifier_col = opt.mol_id_col

        if self._include_fold:

            columns = file.columns
            if 'fold' not in columns:
                self.split_data(root, filename, file, opt)
            
            root = os.path.join(root, opt.experiment_name, f'data')

        super().__init__(opt = opt, filename = self.filename, mol_cols = molcols, root=root, file=file)

        self._name = "rhcaa_diene"
        
    def process(self):

        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        #self.data = self.data.reset_index()

        st.text('Generating Graph Representation of the Molecules...')

        progress_bar = st.progress(0)

        for index, reaction in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            node_feats_reaction = None
            all_smiles = []

            for reactant in self.mol_cols:  

                #create a molecule object from the smiles string
                std_smiles = standardize_smiles(reaction[reactant])

                all_smiles.append(std_smiles)

                mol = Chem.MolFromSmiles(std_smiles)

                mol = Chem.rdmolops.AddHs(mol)

                #initialize the graph level features
                global_features = []

                # checks the global features that the user wants to include
                for global_feat in self._opt.graph_features.keys():


                    # checks if the molecule has the global feature
                    if reactant in self._opt.graph_features[global_feat]:

                        # checks if the global feature has to be one hot encoded
                        if global_feat in self._opt.ohe_graph_feat:
                            #get the unique values of the global feature
                            uni_vals = self.data[global_feat].unique()
                            global_features += self._one_h_e(reaction[global_feat], uni_vals)
                
                        else:

                            global_features += [reaction[global_feat]]

                            #self.
                    
                    # condition where the molecule does not have the global feature but it is necessary to add the '0'
                    else:
                        if global_feat in self._opt.ohe_graph_feat:
                            uni_vals = self.data[global_feat].unique()
                            global_features += self._one_h_e('qWeRtYuIoP', uni_vals)
                        else:
                            global_features += [0]


                node_feats = self._get_node_feats(mol, global_features)

                edge_attr, edge_index = self._get_edge_features(mol)

                if node_feats_reaction is None:
                    node_feats_reaction = node_feats
                    edge_index_reaction = edge_index
                    edge_attr_reaction = edge_attr

                else:
                    node_feats_reaction = torch.cat([node_feats_reaction, node_feats], axis=0)
                    edge_attr_reaction = torch.cat([edge_attr_reaction, edge_attr], axis=0)
                    edge_index += max(edge_index_reaction[0]) + 1
                    edge_index_reaction = torch.cat([edge_index_reaction, edge_index], axis=1)

            y = torch.tensor(reaction[self._opt.target_variable]).reshape(1)

            if self.mol_identifier_col is not None:
                idx = reaction[self.mol_identifier_col]


            if self._include_fold:
                fold = reaction['fold']
            else:
                fold = None

            data = Data(x=node_feats_reaction, 
                        edge_index=edge_index_reaction, 
                        edge_attr=edge_attr_reaction, 
                        y=y,
                        smiles = all_smiles,
                        idx = idx,
                        fold = fold
                        ) 
            
            torch.save(data, 
                       os.path.join(self.processed_dir, 
                                    f'reaction_{index}_{self._opt.split_type}_{self._opt.split_method[:2]}.pt'))
            
            progress_bar.progress(index / self.data.shape[0])

        st.success("Graphs generated succesfully!")
    
    
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

