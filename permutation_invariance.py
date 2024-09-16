import torch
from torch_geometric.data import Data

import pandas as pd
import numpy as np 
from molvs import standardize_smiles
from rdkit import Chem

import random 

from icecream import ic

from options.base_options import BaseOptions

from torch_geometric.utils import to_networkx
import networkx as nx


opt = BaseOptions().parse()

data = pd.read_csv('data/datasets/rhcaa/raw/all_data.csv')


mol_cols = opt.mol_cols

simple_mols = pd.DataFrame({mol_cols[0]: 'CO', mol_cols[1]: 'CC', mol_cols[2]: 'C'}, index=[0])
simple_mols = simple_mols.iloc[0]  


#ic(simple_mols)


random.seed(42)

reaction = data.iloc[0]

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

def ohe(x, allowable_set):
    if x not in allowable_set:
        print(x)
    else:
        return list(map(lambda s: x == s, allowable_set))


def get_node_feats(mol):

    all_node_feats = []

    for atom in mol.GetAtoms():

        node_feats = []

        # Atom type
        #ic(atom.GetSymbol())
        node_feats += ohe(atom.GetSymbol(), elements)
        # Atom Degree
        #ic(atom.GetDegree())
        node_feats += ohe(atom.GetDegree(), [1, 2, 3, 4])
        # hybridization
        #ic(atom.GetHybridization())
        node_feats += ohe(atom.GetHybridization(), [0, 2, 3, 4])

        #node_feats += [atom.GetIsAromatic()]

        #node_feats += [atom.IsInRing()]

        

        all_node_feats.append(node_feats)

    all_node_feats = np.asarray(all_node_feats)
    #ic(all_node_feats.shape)
    return torch.tensor(all_node_feats, dtype=torch.float)

def get_edge_idx(mol):
    edge_idx = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        #ic(i, j)
        edge_idx += [[i, j], [j, i]]

    edge_idx = torch.tensor(edge_idx)
    edge_idx = edge_idx.t().to(torch.long).view(2, -1)
    #ic(edge_idx)
    #ic(edge_idx.shape)
    return edge_idx



def create_mol_graph(reaction, mol_cols = opt.mol_cols):
    node_feats_reaction = None 
    all_smiles = []
    total_nodes = 0  # Keep track of the number of nodes    


    for reactant in mol_cols:

        #ic(reactant)
        #ic(reaction[reactant])

        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(reaction[reactant]), canonical=True)
        all_smiles.append(smiles)
        #ic(smiles)

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.rdmolops.AddHs(mol)

        node_feats = get_node_feats(mol)

        edge_idx = get_edge_idx(mol)

        if node_feats_reaction is None:
            node_feats_reaction = node_feats
            edge_idx_reaction = edge_idx
        else:
            node_feats_reaction = torch.cat((node_feats_reaction, node_feats), axis = 0)
            edge_idx += total_nodes
            edge_idx_reaction = torch.cat((edge_idx_reaction, edge_idx), axis = 1)
        
        total_nodes += node_feats.shape[0]
        
    data = Data(x=node_feats_reaction, 
                    edge_index=edge_idx_reaction)
        
    return data




mol1 = create_mol_graph(reaction, mol_cols)
mols_list = random.sample(mol_cols, len(mol_cols))
mol2 = create_mol_graph(reaction, mol_cols = mols_list)


ic(mol1.x)
ic(mol1.edge_index)

ic(mol2.x)
ic(mol2.edge_index)

ic(mol_cols)
ic(mols_list)



from model.gcn import GCN



model = GCN(opt=opt, n_node_features=mol1.num_node_features)



ic(model.forward(mol1.x, mol1.edge_index))
ic(model.forward(mol2.x, mol2.edge_index))


# Original Graph
x_original = torch.tensor([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6],
                           [0.7, 0.8, 0.9],
                           [0.5, 0.3, 0.2]], dtype=torch.float)

edge_index_original = torch.tensor([[0, 1, 2],
                                    [1, 2, 0]], dtype=torch.long)

graph_original = Data(x=x_original, edge_index=edge_index_original)

# Permuted Graph
x_permuted = torch.tensor([[0.4, 0.5, 0.6],  # Node 1 -> Node 0
                           [0.7, 0.8, 0.9],  # Node 2 -> Node 1
                           [0.1, 0.2, 0.3],
                           [0.5, 0.3, 0.2]], dtype=torch.float)  # Node 0 -> Node 2

edge_index_permuted = torch.tensor([[2, 0, 1],
                                    [0, 1, 2]], dtype=torch.long)

graph_permuted = Data(x=x_permuted, edge_index=edge_index_permuted)


model2 = GCN(opt=opt, n_node_features=graph_original.num_node_features)

ic(model2.forward(graph_original.x, graph_original.edge_index))
ic(model2.forward(graph_permuted.x, graph_permuted.edge_index))


import networkx as nx

G1 = to_networkx(mol1, to_undirected=False, node_attrs=['x'])
G2 = to_networkx(mol2, to_undirected=False, node_attrs=['x'])
ic(nx.number_connected_components(G1.to_undirected()))
ic(nx.number_connected_components(G2.to_undirected()))
ic(nx.is_isomorphic(G1, G2, node_match=lambda n1, n2: n1['x'] == n2['x']))

G3 = to_networkx(graph_original, to_undirected=False, node_attrs=['x'])
G4 = to_networkx(graph_permuted, to_undirected=False, node_attrs=['x'])
ic(G3)
ic(nx.is_isomorphic(G3, G4, node_match=lambda n1, n2: n1['x'] == n2['x']))


