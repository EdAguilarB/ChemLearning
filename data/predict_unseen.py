import pandas as pd
import streamlit as st
import torch
from torch_geometric.data import  Data
import numpy as np 
from rdkit import Chem
import os
import sys
from data.mol_graphs import molecular_graphs




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class predict_insilico(molecular_graphs):
    def __init__(self, data, opt):
        super().__init__(data=data, opt=opt)


    def process(self,):

        opt = self._opt

        all_graphs = []

        st.text("Generating graph representation of the in silico library...")

        progress_bar = st.progress(0)

        for index, mols in self.data.iterrows():

            node_feats_mols = None
            all_smiles = []

            for molecule in opt.mol_cols:

                std_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mols[molecule]), canonical=True)

                all_smiles.append(std_smiles)

                mol = Chem.MolFromSmiles(std_smiles)

                mol = Chem.rdmolops.AddHs(mol)

                global_features = []

                for global_feat in opt.graph_features.keys():


                    if molecule in opt.graph_features[global_feat]:


                        if global_feat in opt.ohe_graph_feat:

                            uni_vals = opt.ohe_pos_vals[global_feat]

                            global_features += self._one_h_e(mols[global_feat], uni_vals)

                        else:
                            global_features += [mols[global_feat]]

                    else:
                        if global_feat in opt.ohe_graph_feat:
                            uni_vals = opt.ohe_pos_vals[global_feat]
                            global_features += self._one_h_e('qWeRtYuIoP', uni_vals, 'qWeRtYuIoP')
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

            progress_bar.progress(int(index+1) / self.data.shape[0])


        return all_graphs
                        

    
    


    


    

    