import os
import sys
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
import pandas as pd

from utils.plot_utils import  plot_importances
from model.gcn import GCN
from utils.other_utils import explain_dataset, visualize_score_features, \
    plot_molecule_importance, get_graph_by_idx, plot_denoised_mols



sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def denoise_graphs(opt, exp_path:str) -> None:

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print('Analysing outer {}, inner {}'.format(outer, inner))

    model_path = os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set')

    train_loader = torch.load(os.path.join(model_path, 'train_loader.pth'))
    val_loader = torch.load(os.path.join(model_path, 'val_loader.pth'))
    test_loader = torch.load(os.path.join(model_path, 'test_loader.pth'))

    all_data = train_loader.dataset + val_loader.dataset + test_loader.dataset

    loader_all = DataLoader(all_data)

    model = GCN(opt, n_node_features=all_data[0].num_node_features)
    model_params = torch.load(os.path.join(model_path, 'model_params.pth'))
    model.load_state_dict(model_params)

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


    mol = get_graph_by_idx(loader_all, int(opt.denoise_reactions))

    masks  = explain_dataset(dataset = mol,
                                 explainer = explainer,
                                 mol = opt.denoise_mol,)
        
    plot_denoised_mols(mask = masks,
                           graph = mol,
                           mol = opt.denoise_mol,
                           analysis = opt.denoise_based_on,
                            norm = opt.norm_denoise,
                           )



def GNNExplainer_node_feats(opt, exp_path:str) -> None:


    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print('Analysing outer {}, inner {}'.format(outer, inner))

    model_path = os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set')

    train_loader = torch.load(os.path.join(model_path, 'train_loader.pth'))
    val_loader = torch.load(os.path.join(model_path, 'val_loader.pth'))
    test_loader = torch.load(os.path.join(model_path, 'test_loader.pth'))

    all_data = train_loader.dataset + val_loader.dataset + test_loader.dataset

    model = GCN(opt, n_node_features=all_data[0].num_node_features)
    model_params = torch.load(os.path.join(model_path, 'model_params.pth'))
    model.load_state_dict(model_params)

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

    ligand_masks, substrate_masks, boron_masks, _  = explain_dataset(test_loader.dataset, 
                                                                             explainer)

    ligands = visualize_score_features(score = ligand_masks)
    ligands = ligands.loc[ligands['score'] != 0]
    ligands['labels'] = ligands['labels'].apply(lambda m: 'L. '+m)
    print('Ligand node features score: \n', ligands)

    substrate = visualize_score_features(score = substrate_masks)
    substrate = substrate.loc[substrate['score'] != 0]
    substrate['labels'] = substrate['labels'].apply(lambda m: 'S. '+m)
    print('Substrate node features score: \n', substrate)

    boron = visualize_score_features(score = boron_masks)
    boron = boron.loc[boron['score'] != 0]
    boron['labels'] = boron['labels'].apply(lambda m: 'BR. '+m)
    print('Boron node features score: \n', boron)

    df = pd.concat([ligands, substrate, boron])
    df = df.sort_values('score', ascending=False)
    df['score'] = df['score'].astype(int)

    plot_importances(df = df, save_path=os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set'))



def shapley_analysis(opt, exp_path:str) -> None:

    outer, inner = opt.explain_model[0], opt.explain_model[1]

    print('Analysing outer {}, inner {}'.format(outer, inner))

    model_path = os.path.join(exp_path, f'Fold_{outer}_test_set', f'Fold_{inner}_val_set')

    train_loader = torch.load(os.path.join(model_path, 'train_loader.pth'))
    val_loader = torch.load(os.path.join(model_path, 'val_loader.pth'))
    test_loader = torch.load(os.path.join(model_path, 'test_loader.pth'))

    all_data = train_loader.dataset + val_loader.dataset + test_loader.dataset

    loader = DataLoader(all_data)

    model = GCN(opt, n_node_features=all_data[0].num_node_features)
    model_params = torch.load(os.path.join(model_path, 'model_params.pth'))
    model.load_state_dict(model_params)

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('ShapleyValueSampling'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )

    try:
        for molecule in tqdm(opt.shap_index):
            mol = get_graph_by_idx(loader, molecule)
            print('Analysing reaction {}'.format(molecule))
            print('Ligand id: {}'.format(mol.ligand_id[0]))
            print('Reaction ddG: {:.2f}'.format(mol.y.item()))
            print('Reaction predicted ddG: {:.2f}'.format(explainer.get_prediction(x = mol.x, edge_index=mol.edge_index, batch_index=mol.batch).item()))
            explanation = explainer(x = mol.x, edge_index=mol.edge_index,  batch_index=mol.batch)
            plot_molecule_importance(mol_graph=mol, mol='l', explanation=explanation, palette='normal')

    except:
        mol = get_graph_by_idx(loader, opt.shap_index)
        print('Analysing reaction {}'.format(opt.shap_index))
        print('Ligand id: {}'.format(mol.ligand_id[0]))
        print('Reaction ddG: {:.2f}'.format(mol.y.item()))
        print('Reaction predicted ddG: {:.2f}'.format(explainer.get_prediction(x = mol.x, edge_index=mol.edge_index, batch_index=mol.batch).item()))
        explanation = explainer(x = mol.x, edge_index=mol.edge_index,  batch_index=mol.batch)
        plot_molecule_importance(mol_graph=mol, mol='l', explanation=explanation, palette='normal')
