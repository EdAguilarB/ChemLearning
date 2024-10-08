import torch
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
import streamlit as st
from utils.other_utils import plot_molecule_importance
from utils.other_utils import plot_denoised_mols
from typing import Optional
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs
import numpy as np
import plotly.graph_objects as go


color_map = {"H": "lightgrey", "D": "#FFFFC0", "T": "#FFFFA0", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
             "B": "#FFB5B5", "C": "#909090", "C-13": "#505050", "C-14": "#404040", "N": "#3050F8", "N-15": "#105050",
             "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6",
             "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4",
             "Ca": "#3DFF00", "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
             "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F",
             "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1", "Rb": "#702EB0",
             "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E",
             "Ru": "#248F8F", "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
             "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F",
             "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7", "Pm": "#A3FFC7",
             "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C",
             "Er": "#00E675", "Tm": "#00D452", "Yb": "#00BF38", "Lu": "#00AB24", "Hf": "#4DC2FF", "Ta": "#4DA6FF",
             "W": "#2194D6", "Re": "#267DAB", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
             "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00", "At": "#754F45",
             "Rn": "#428296", "Fr": "#420066", "Ra": "#007D00", "Ac": "#70ABFA", "Th": "#00BAFF", "Pa": "#00A1FF",
             "U": "#008FFF", "Np": "#0080FF", "Pu": "#006BFF", "Am": "#545CF2", "Cm": "#785CE3", "Bk": "#8A4FE3",
             "Cf": "#A136D4", "Es": "#B31FD4", "Fm": "#B31FBA", "Md": "#B30DA6", "No": "#BD0D87", "Lr": "#C70066",
             "Rf": "#CC0059", "Db": "#D1004F", "Sg": "#D90045", "Bh": "#E00038", "Hs": "#E6002E", "Mt": "#EB0026",
             'other': '#f5c2cb', 'Bond': '#979797', 'Background': "#ffffff", 'All icon': "#000000", 'All atoms': "#000000"}

sizes = {
    'C': 69,    # Carbon
    'O': 66,    # Oxygen
    'N': 71,    # Nitrogen
    'H': 31,    # Hydrogen
    'B': 84,    # Boron
    'F': 64,    # Fluorine
    'Cl': 99,   # Chlorine
    'Br': 114,  # Bromine
    'I': 133,   # Iodine
    'S': 104,   # Sulfur
    'P': 110,   # Phosphorus
    'Si': 118,  # Silicon
    'Na': 186,  # Sodium
    'K': 231,   # Potassium
    'Mg': 160,  # Magnesium
    'Ca': 194,  # Calcium
    'Al': 143,  # Aluminum
    'Zn': 139   # Zinc
}


def explain_mols(opt, model, mol_graphs) -> None:

    loader = DataLoader(mol_graphs)


    if opt.algorithm == 'GNNExplainer':
        algorithm = GNNExplainer()
    else:
        algorithm = CaptumExplainer(opt.algorithm)

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    for mol in loader:
        explanation = explainer(x = mol.x, edge_index=mol.edge_index,  batch_index=mol.batch)
        plot_molecule_importance(opt, mol_graph=mol, explanation=explanation)


def plot_molecule_importance(opt, mol_graph, explanation):

    edge_idx = mol_graph.edge_index
    fa, la, coords, atom_symbol = mol_prep(opt=opt, mol_graph=mol_graph)
    edge_coords = dict(zip(range(fa, la), coords))

    edge_mask_dict, node_mask = get_masks(explanation=explanation, fa=fa, la=la, edge_idx=edge_idx, opt=opt)

    edge_mask_dict, node_mask = normalise_masks(edge_mask_dict=edge_mask_dict, node_mask=node_mask, opt=opt)
    

    coords_edges = [(np.concatenate([np.expand_dims(edge_coords[u], axis=1), np.expand_dims(edge_coords[v], axis =1)], 
                                axis = 1)) for u, v in edge_mask_dict.keys()]
    
    edge_weights = list(edge_mask_dict.values())
    opacity_edges = [(x + 1) / 2 for x in edge_weights]
    opacity_nodes = [(x + 1) / 2 for x in node_mask]

    neg_edges = [True if num < 0 else False for num in list(edge_mask_dict.values())]
    neg_nodes = [True if num < 0 else False for num in node_mask]

    scale_factor = opt.scale_factor

    sizes_plot = {key: value / scale_factor for key, value in sizes.items()}

    color_nodes_imp = ['red' if boolean else 'blue' for boolean in neg_nodes]
    color_edges_imp = ['red' if boolean else 'blue' for boolean in neg_edges]

    node_mask = np.array(node_mask)

    if opt.type_contrast == 'Continuous':
        node_mask = np.where(node_mask < opt.contrast_threshold, np.power(node_mask, 2), np.where(node_mask >0, np.sqrt(node_mask), -np.sqrt(-node_mask)))
    else:
        node_mask = np.where(node_mask < opt.contrast_threshold, 0, np.where(node_mask >0, 1, -1))

    if opt.algorithm == 'ShapleyValueSampling':
        atoms = trace_atoms(atom_symbol = atom_symbol, coords=coords,sizes=sizes_plot, 
                            colors=color_map)
        atoms_imp = trace_atom_imp(coords=coords, opacity=opacity_nodes, 
                               atom_symbol=atom_symbol, sizes=sizes_plot,color=color_nodes_imp)
        bond_imp = trace_bond_imp(coords_edges=coords_edges, edge_mask_dict=edge_mask_dict,
                              opacity=opacity_edges, color_edges=color_edges_imp)
        
    elif opt.algorithm == 'GNNExplainer':
        atoms = trace_atoms(atom_symbol=atom_symbol, coords=coords, sizes=sizes_plot, colors=color_map, transparencies=node_mask)
        atoms_imp = []
        bond_imp = []
    
    bonds = trace_bonds(coords_edges=coords_edges, edge_mask_dict=edge_mask_dict, width=opt.bond_width)
    
    
    all_traces(atoms=atoms, atoms_imp=atoms_imp, bonds=bonds, bonds_imp=bond_imp)


def denoise_graphs(opt, model, mol_graphs) -> None:

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    loader = DataLoader(mol_graphs, batch_size=1, shuffle=False)

    for mol in loader:
        masks  = explain_dataset(opt,
                                 mol_graph= mol,
                                    explainer = explainer,
                                    mol = opt.denoise_mol,)
            
        plot_denoised_mols(mask = masks,
                            graph = mol,
                            opt = opt)



def explain_dataset(opt,
                    mol_graph, 
                   explainer,
                   mol: Optional[str] = None,):
    
    mols = {}
    ia = 0
    
    for mol_name, smiles in zip(opt.mol_cols, mol_graph.smiles[0]):

        attributes = []

        mol = AddHs(Chem.MolFromSmiles(smiles))
        la = mol.GetNumAtoms()
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        coords = mol.GetConformer().GetPositions()
        atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]

        attributes.append(ia)
        attributes.append(ia+la)
        attributes.append(coords)
        attributes.append(atom_symbol)

        mols[mol_name] = attributes

        ia += la
    
    # Run the explanation function over the reaction graph
    explanation = explainer(x = mol_graph.x, 
                            edge_index=mol_graph.edge_index,  
                            batch_index=mol_graph.batch)
    
    
    # Get the masks for each node within the molecule
    masks = explanation.node_mask

    # Normalize the masks by the attribution of the most important node
    masks = masks/torch.max(masks.sum(dim=1))

    # Append the masks for each molecule to the list
    masks_mol = {}
    for mol in mols.keys():
        fa, la, _, _, = mols[mol]
        masks_mol[mol] = masks[fa:la]

    return masks_mol[opt.explain_mol]







def mol_prep(opt, mol_graph, ):

    mols = {}

    ia = 0

    for mol_name, smiles in zip(opt.mol_cols, mol_graph.smiles[0]):

        attributes = []

        mol = AddHs(Chem.MolFromSmiles(smiles))
        la = mol.GetNumAtoms()
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        coords = mol.GetConformer().GetPositions()
        atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]

        attributes.append(ia)
        attributes.append(ia+la)
        attributes.append(coords)
        attributes.append(atom_symbol)

        mols[mol_name] = attributes

        ia += la

    fa, la, coords, atom_symbol = mols[opt.explain_mol] 

    return fa, la, coords, atom_symbol

def get_masks(explanation, fa, la, edge_idx, opt):
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *edge_idx)):
        u, v = u.item(), v.item()
        if u in range(fa, la):
                if u > v:
                        u, v = v, u
                edge_mask_dict[(u, v)] += val.item()

    if opt.normalize == 'All':
        node_mask = node_mask/torch.max(node_mask.sum(dim=1))

    atom_identity = 18
    degree = 7
    hyb = 7
    aromatic = 1
    ring = 1
    chiral = 2

    importances = []
    importances.append(node_mask[:, 0:atom_identity])
    importances.append(node_mask[:, atom_identity:atom_identity+degree])
    importances.append(node_mask[:, atom_identity+degree:atom_identity+degree+hyb])
    importances.append(node_mask[:, atom_identity+degree+hyb:atom_identity+degree+hyb+aromatic])
    importances.append(node_mask[:, atom_identity+degree+hyb+aromatic:atom_identity+degree+hyb+aromatic+ring])
    importances.append(node_mask[:, atom_identity+degree+hyb+aromatic+ring:atom_identity+degree+hyb+aromatic+ring+chiral])

    if opt.analysis == 'Atom Identity':
        importance = importances[0]

    elif opt.analysis == 'Atom Degree':
        importance = importances[1]

    elif opt.analysis == 'Atom Hybridization':
        importance = importances[2]

    elif opt.analysis == 'Is Atom Aromatic?':
        importance = importances[3]

    elif opt.analysis == 'Is Atom in Ring?':
        importance = importances[4]

    elif opt.analysis == 'Atom Chirality':
        importance = importances[5]      

    else:
        importance = node_mask

    if opt.normalize=='Molecule':
        if torch.max(importance.sum(dim=1)) == 0:
            st.warning(f'All attributions are zero for feature {opt.analysis} in molecule {opt.explain_mol}. Please select another molecule or feature to explain or other normalization method.')
            st.stop()
        importance = importance/torch.max(torch.abs(importance).sum(dim=1))
    
    importance=importance[fa:la]

    return edge_mask_dict, importance



def normalise_masks(edge_mask_dict, node_mask, opt):

    # Find the negative edges 
    neg_edge = [True if num < 0 else False for num in list(edge_mask_dict.values())]

    # Identify the maximum and minimum values of the edge masks
    min_value_edge = abs(min(edge_mask_dict.values(), key=abs))
    max_value_edge = abs(max(edge_mask_dict.values(), key=abs))

    abs_dict = {key: abs(value) for key, value in edge_mask_dict.items()}
    abs_dict = {key: (value - min_value_edge) / (max_value_edge - min_value_edge) 
                for key, value in abs_dict.items()}
    
    edge_mask_dict_norm = {key: -value if convert else value for (key, value), convert 
                      in zip(abs_dict.items(), neg_edge)}
    
    node_mask = node_mask.sum(axis = 1)
    node_mask = [val.item() for val in node_mask]
    neg_nodes = [True if num < 0 else False for num in node_mask]
    max_node = abs(max(node_mask, key = abs))
    min_node = abs(min(node_mask, key = abs))
    abs_node = [abs(w) for w in node_mask]





    if opt.normalize == 'Features':
        if max_node == 0 and min_node == 0:
            st.warning(f'All attributions are zero for feature {opt.analysis} in molecule {opt.explain_mol}. Please select another molecule or feature to explain or other normalization method.')
            st.stop()
        else: 
            abs_node = [(w-min_node)/(max_node-min_node) for w in abs_node]
            node_mask_norm = [-w if neg_nodes else w for w, neg_nodes in zip(abs_node, neg_nodes)]
    else:
        node_mask_norm = node_mask
    
    return edge_mask_dict_norm, node_mask_norm

def trace_atoms(atom_symbol, coords, sizes, colors, transparencies=None):
    trace_atoms = [None] * len(atom_symbol)
    for i in range(len(atom_symbol)):
        marker_dict = {
            'symbol': 'circle',
            'size': sizes[atom_symbol[i]],
            'color': colors[atom_symbol[i]]
        }
        
        if transparencies is not None:
            marker_dict['opacity'] = transparencies[i]
        
        trace_atoms[i] = go.Scatter3d(
            x=[coords[i][0]],
            y=[coords[i][1]],
            z=[coords[i][2]],
            mode='markers',
            text=f'atom {atom_symbol[i]}',
            legendgroup='Atoms',
            showlegend=False,
            marker=marker_dict
        )
    return trace_atoms

def trace_atom_imp(coords, opacity, atom_symbol, sizes, color):
    trace_atoms_imp = [None] * len(atom_symbol)
    for i in range(len(atom_symbol)):

        trace_atoms_imp[i] = go.Scatter3d(x=[coords[i][0]],
                            y=[coords[i][1]],
                            z=[coords[i][2]],
                            mode='markers',
                            showlegend=False,
                            opacity=opacity[i],
                            text = f'atom {atom_symbol[i]}',
                            legendgroup='Atom importance',
                            marker=dict(symbol='circle',
                                                    size=sizes[atom_symbol[i]]*1.7,
                                                    color=color[i])
        )
    return trace_atoms_imp



def trace_bonds(coords_edges, edge_mask_dict, width=2):
    trace_edges = [None] * len(edge_mask_dict)
    
    for i in range(len(edge_mask_dict)):
        trace_edges[i]= go.Scatter3d(
            x=coords_edges[i][0],
            y=coords_edges[i][1],
            z=coords_edges[i][2],
            mode='lines',
            showlegend=False,
            legendgroup='Bonds',
            line=dict(color='black', width=width),
            hoverinfo='none')
        
    return trace_edges


def trace_bond_imp(coords_edges, edge_mask_dict, opacity, color_edges):
    trace_edge_imp = [None] * len(edge_mask_dict)
    for i in range(len(edge_mask_dict)):
        trace_edge_imp[i]= go.Scatter3d(
            x=coords_edges[i][0],
            y=coords_edges[i][1],
            z=coords_edges[i][2],
            mode='lines',
            showlegend=False,
            legendgroup='Bond importance',
            opacity=opacity[i],
            line=dict(color=color_edges[i], width=opacity[i]*15),
            hoverinfo='none')
        
    return trace_edge_imp




def all_traces(atoms, atoms_imp, bonds, bonds_imp):
    traces =   atoms + atoms_imp + bonds + bonds_imp
    fig = go.Figure(data=traces)
    fig.add_trace(go.Scatter3d(
    x=[None],
    y=[None],
    z=[None],
    mode='markers',
    legendgroup='Atoms',
    name='Atoms'
        ))

    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        legendgroup='Atom importance',
        name='Atom importance'
    ))

    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        legendgroup='Bonds',
        name='Bonds'
    ))

    fig.add_trace(go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode='markers',
        legendgroup='Bond importance',
        name='Bond importance',
        showlegend=True
    ))

    fig.update_layout(template =  'plotly_white')

    fig.update_layout(scene=dict(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                xaxis_title='', yaxis_title='', zaxis_title=''))


    st.plotly_chart(fig, use_container_width=True)




