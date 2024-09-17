import os
import joblib
from options.base_options import BaseOptions
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from utils.utils_model import tml_report, network_outer_report, network_report
from data.rhcaa import rhcaa_diene


def predict_final_test() -> None:

    opt = BaseOptions().parse()

    # Get the current working directory
    current_dir = os.getcwd()
    
    # Load the final test set
    final_test =rhcaa_diene(opt, opt.filename_final_test, opt.mol_cols, opt.root_final_test, include_fold=False)
    test_loader = DataLoader(final_test, shuffle=False)

    descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', 'temp']

    # Load the data for tml
    test_set = pd.read_csv(f'{opt.root_final_test}/raw/{opt.filename_final_test}')

    experiments_gnn = os.path.join(current_dir, opt.log_dir_results, 'final_test', 'results_GNN')
    experiments_tml = os.path.join(current_dir, opt.log_dir_results, 'final_test', f'results_{opt.tml_algorithm}')

    for outer in range(1, opt.folds+1):
        print('Analysing models trained using as test set {}'.format(outer))
        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            print('Analysing models trained using as validation set {}'.format(real_inner))

            model_dir = os.path.join(current_dir, opt.log_dir_results, opt.filename[:-4], 'results_GNN', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = torch.load(model_dir+'/model.pth')
            model_params = torch.load(model_dir+'/model_params.pth')
            train_loader = torch.load(model_dir+'/train_loader.pth')
            val_loader = torch.load(model_dir+'/val_loader.pth')

            network_report(log_dir=experiments_gnn,
                           loaders=(train_loader, val_loader, test_loader),
                           outer=outer,
                           inner=real_inner,
                           loss_lists=[None, None, None],
                           model=model,
                           model_params=model_params,
                           best_epoch=None,
                           save_all=False)
            
            tml_dir = os.path.join(current_dir, opt.log_dir_results, opt.filename[:-4], f'results_{opt.tml_algorithm}', f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            model = joblib.load(tml_dir+'/model.sav')
            train_data = pd.read_csv(tml_dir+'/train.csv')
            val_data = pd.read_csv(tml_dir+'/val.csv')

            tml_report(log_dir=experiments_tml,
                       outer=outer,
                       inner=real_inner,
                       model=model,
                       data=(train_data,val_data,test_set),
                       save_all=False,
                       descriptors=descriptors)
            
                        
        network_outer_report(log_dir=f"{experiments_gnn}/Fold_{outer}_test_set/", 
                             outer=outer)
        
        network_outer_report(log_dir=f"{experiments_tml}/Fold_{outer}_test_set/", 
                             outer=outer)