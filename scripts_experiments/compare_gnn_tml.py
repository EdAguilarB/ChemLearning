import os
from options.base_options import BaseOptions
import pandas as pd
import numpy as np
from utils.utils_model import  extract_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, \
    accuracy_score, precision_score, recall_score
from utils.plot_utils import create_bar_plot, create_violin_plot, create_strip_plot, \
    create_parity_plot
from math import sqrt


def plot_results(exp_dir):

    opt = BaseOptions().parse()

    experiments_gnn = os.path.join(exp_dir, 'results_GNN')
    experiments_tml = os.path.join(exp_dir, f'results_{opt.tml_algorithm}')

    r2_gnn, mae_gnn, rmse_gnn = [], [], []
    accuracy_gnn, precision_gnn, recall_gnn = [], [], []
    r2_tml, mae_tml, rmse_tml = [], [], []
    accuracy_tml, precision_tml, recall_tml = [], [], []

    results_all = pd.DataFrame(columns = ['index', 'Test_Fold', 'Val_Fold', 'Method', 'real_ddG', 'predicted_ddG'])
    
    for outer in range(1, opt.folds+1):

        outer_gnn = os.path.join(experiments_gnn, f'Fold_{outer}_test_set')
        outer_tml = os.path.join(experiments_tml, f'Fold_{outer}_test_set')

        metrics_gnn = extract_metrics(file=outer_gnn+f'/performance_outer_test_fold{outer}.txt')
        metrics_tml = extract_metrics(file=outer_tml+f'/performance_outer_test_fold{outer}.txt')

        r2_gnn.append(metrics_gnn['R2'])
        mae_gnn.append(metrics_gnn['MAE'])
        rmse_gnn.append(metrics_gnn['RMSE'])
        accuracy_gnn.append(metrics_gnn['Accuracy'])
        precision_gnn.append(metrics_gnn['Precision'])
        recall_gnn.append(metrics_gnn['Recall'])

        r2_tml.append(metrics_tml['R2'])
        mae_tml.append(metrics_tml['MAE'])
        rmse_tml.append(metrics_tml['RMSE'])
        accuracy_tml.append(metrics_tml['Accuracy'])
        precision_tml.append(metrics_tml['Precision'])
        recall_tml.append(metrics_tml['Recall'])

        for inner in range(1, opt.folds):
    
            real_inner = inner +1 if outer <= inner else inner
            
            gnn_dir = os.path.join(experiments_gnn, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_gnn = pd.read_csv(gnn_dir+'/predictions_test_set.csv', index_col=0)

            df_gnn['Test_Fold'] = outer
            df_gnn['Val_Fold'] = real_inner
            df_gnn['Method'] = 'GNN'

            results_all = pd.concat([results_all, df_gnn], axis=0)

            tml_dir = os.path.join(experiments_tml, f'Fold_{outer}_test_set', f'Fold_{real_inner}_val_set')

            df_tml = pd.read_csv(tml_dir+'/predictions_test_set.csv', index_col=0)
            df_tml['Test_Fold'] = outer
            df_tml['Val_Fold'] = real_inner
            df_tml['Method'] = opt.tml_algorithm

            results_all = pd.concat([results_all, df_tml], axis=0)

    save_dir = f'{exp_dir}/GNN_vs_{opt.tml_algorithm}'
    os.makedirs(save_dir, exist_ok=True)
    
    mae_mean_gnn = np.array([entry['mean'] for entry in mae_gnn])
    mae_gnn_std = np.array([entry['std'] for entry in mae_gnn])

    mae_mean_tml = np.array([entry['mean'] for entry in mae_tml])
    mae_tml_std = np.array([entry['std'] for entry in mae_tml])

    rmse_mean_gnn = np.array([entry['mean'] for entry in rmse_gnn])
    rmse_gnn_std = np.array([entry['std'] for entry in rmse_gnn])

    rmse_mean_tml = np.array([entry['mean'] for entry in rmse_tml])
    rmse_tml_std = np.array([entry['std'] for entry in rmse_tml])

    minimun = np.min(np.array([
        (mae_mean_gnn - mae_gnn_std).min(), 
        (mae_mean_tml - mae_tml_std).min(),
        (rmse_mean_gnn - rmse_gnn_std).min(), 
        (rmse_mean_tml - rmse_tml_std).min()]))
    
    maximun = np.max(np.array([
        (mae_mean_gnn + mae_gnn_std).max(),
        (mae_mean_tml + mae_tml_std).max(),
        (rmse_mean_gnn + rmse_gnn_std).max(), 
        (rmse_mean_tml + rmse_tml_std).max()]))
    
    create_bar_plot(means=(mae_mean_gnn, mae_mean_tml), stds=(mae_gnn_std, mae_tml_std), min = minimun, max = maximun, metric = 'MAE', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(rmse_mean_gnn, rmse_mean_tml), stds=(rmse_gnn_std, rmse_tml_std), min = minimun, max = maximun, metric = 'RMSE', save_path= save_dir, tml_algorithm=opt.tml_algorithm)

    r2_mean_gnn = np.array([entry['mean'] for entry in r2_gnn])
    r2_gnn_std = np.array([entry['std'] for entry in r2_gnn])

    r2_mean_tml = np.array([entry['mean'] for entry in r2_tml])
    r2_tml_std = np.array([entry['std'] for entry in r2_tml])

    accuracy_mean_gnn = np.array([entry['mean'] for entry in accuracy_gnn])
    accuracy_gnn_std = np.array([entry['std'] for entry in accuracy_gnn])

    accuracy_mean_tml = np.array([entry['mean'] for entry in accuracy_tml])
    accuracy_tml_std = np.array([entry['std'] for entry in accuracy_tml])

    precision_mean_gnn = np.array([entry['mean'] for entry in precision_gnn])
    precision_gnn_std = np.array([entry['std'] for entry in precision_gnn])

    precision_mean_tml = np.array([entry['mean'] for entry in precision_tml])
    precision_tml_std = np.array([entry['std'] for entry in precision_tml])

    recall_mean_gnn = np.array([entry['mean'] for entry in recall_gnn])
    recall_gnn_std = np.array([entry['std'] for entry in recall_gnn])

    recall_mean_tml = np.array([entry['mean'] for entry in recall_tml])
    recall_tml_std = np.array([entry['std'] for entry in recall_tml])
    
    minimun = np.min(np.array([
    (r2_mean_gnn - r2_gnn_std).min(),
    (r2_mean_tml - r2_tml_std).min(),
    (accuracy_mean_gnn - accuracy_gnn_std).min(),
    (accuracy_mean_tml - accuracy_tml_std).min(),
    (precision_mean_gnn - precision_gnn_std).min(),
    (precision_mean_tml - precision_tml_std).min(),
    (recall_mean_gnn - recall_gnn_std).min(),
    (recall_mean_tml - recall_tml_std).min()
    ]))

    maximun = np.max(np.array([
    (r2_mean_gnn + r2_gnn_std).max(),
    (r2_mean_tml + r2_tml_std).max(),
    (accuracy_mean_gnn + accuracy_gnn_std).max(),
    (accuracy_mean_tml + accuracy_tml_std).max(),
    (precision_mean_gnn + precision_gnn_std).max(),
    (precision_mean_tml + precision_tml_std).max(),
    (recall_mean_gnn + recall_gnn_std).max(),
    (recall_mean_tml + recall_tml_std).max()
    ]))
    
    
    create_bar_plot(means=(r2_mean_gnn, r2_mean_tml), stds=(r2_gnn_std, r2_tml_std), min = minimun, max = maximun, metric = 'R2', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(accuracy_mean_gnn, accuracy_mean_tml), stds=(accuracy_gnn_std, accuracy_tml_std), min = minimun, max = maximun, metric = 'Accuracy', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(precision_mean_gnn, precision_mean_tml), stds=(precision_gnn_std, precision_tml_std), min = minimun, max = maximun, metric = 'Precision', save_path= save_dir, tml_algorithm=opt.tml_algorithm)
    create_bar_plot(means=(recall_mean_gnn, recall_mean_tml), stds=(recall_gnn_std, recall_tml_std), min = minimun, max = maximun, metric = 'Recall', save_path= save_dir, tml_algorithm=opt.tml_algorithm)

    results_all['Error'] = results_all['real_ddG'] - results_all['predicted_ddG']
    results_all['real_face'] = np.where(results_all['real_ddG'] > 0, 1, 0)
    results_all['predicted_face'] = np.where(results_all['predicted_ddG'] > 0, 1, 0)

    create_violin_plot(data=results_all, save_path= save_dir)
    create_strip_plot(data=results_all, save_path= save_dir)

    create_parity_plot(data=results_all, save_path= save_dir, tml_algorithm=opt.tml_algorithm)

    results_all = results_all.reset_index(drop=True)
    results_all.to_csv(f'{save_dir}/predictions_all.csv', index=False)


    print('All plots have been saved in the directory {}'.format(save_dir))

    gnn_predictions = results_all[results_all['Method'] == 'GNN']

    print('\n')

    print('Final metrics for GNN:')

    print('Accuracy: {:.3f}'.format(accuracy_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(gnn_predictions['real_face'], gnn_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(gnn_predictions['real_ddG'], gnn_predictions['predicted_ddG'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(gnn_predictions['real_ddG'], gnn_predictions['predicted_ddG'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(gnn_predictions['real_ddG'], gnn_predictions['predicted_ddG']))))
    
    tml_predictions = results_all[results_all['Method'] == opt.tml_algorithm]
    print(f'Final metrics for {opt.tml_algorithm}:')
    print('Accuracy: {:.3f}'.format(accuracy_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Precision: {:.3f}'.format(precision_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('Recall: {:.3f}'.format(recall_score(tml_predictions['real_face'], tml_predictions['predicted_face'])))
    print('R2: {:.3f}'.format(r2_score(tml_predictions['real_ddG'], tml_predictions['predicted_ddG'])))
    print('MAE: {:.3f}'.format(mean_absolute_error(tml_predictions['real_ddG'], tml_predictions['predicted_ddG'])))
    print('RMSE: {:.3f} \n'.format(sqrt(mean_squared_error(tml_predictions['real_ddG'], tml_predictions['predicted_ddG']))))