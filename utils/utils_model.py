import torch
import os
import csv
import re
import pickle
import numpy as np
from datetime import date, datetime
import streamlit as st
from copy import copy, deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, \
    accuracy_score, precision_score, recall_score, f1_score
from math import sqrt
from utils.plot_utils import *
from icecream import ic
from sklearn.preprocessing import RobustScaler


def calculate_metrics(y_true:list, y_predicted: list, task:str):

    if task == 'regression':
        r2 = r2_score(y_true=y_true, y_pred=y_predicted)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_predicted))  
        error = [(y_predicted[i]-y_true[i]) for i in range(len(y_true))]
        prctg_error = [ abs(error[i] / y_true[i]) for i in range(len(error))]
        mbe = np.mean(error)
        mape = np.mean(prctg_error)
        error_std = np.std(error)
        metrics = [r2, mae, rmse, mbe, mape, error_std]
        metrics_names = ['R2', 'MAE', 'RMSE', 'Mean Bias Error', 'Mean Absolute Percentage Error', 'Error Standard Deviation']

    elif task == 'classification':
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
        precision = precision_score(y_true=y_true, y_pred=y_predicted)
        recall = recall_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)
        metrics = [accuracy, precision, recall, f1]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    return np.array(metrics), metrics_names

######################################
######################################
######################################
###########  GNN functions ###########
######################################
######################################
######################################

def train_network(model, train_loader, device):

    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        model.optimizer.zero_grad()
        out = model(batch.x, 
                    batch.edge_index, 
                    batch.batch)
        loss = torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim=1)))
        loss.backward()
        model.optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)

def eval_network(model, loader, device):
    model.eval()
    loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x,
                    batch.edge_index,
                    batch.batch)
        loss += torch.sqrt(model.loss(out, torch.unsqueeze(batch.y, dim = 1) )).item() * batch.num_graphs
    return loss / len(loader.dataset)


def predict_network(model, loader, return_emb = False):
    model.to('cpu')
    model.eval()

    y_pred, y_true, idx, embeddings = [], [], [], []

    for batch in loader:
        batch = batch.to('cpu')
        out, emb = model(x = batch.x, 
                         edge_index = batch.edge_index, 
                         batch_index = batch.batch, 
                         return_graph_embedding = True)

        y_pred.append(out.cpu().detach().numpy())
        y_true.append(batch.y.cpu().detach().numpy())
        idx.append(batch.idx.cpu().detach().numpy())
        embeddings.append(emb.detach().numpy())

    y_pred = np.concatenate(y_pred, axis=0).ravel()
    y_true = np.concatenate(y_true, axis=0).ravel()
    idx = np.concatenate(idx, axis=0).ravel()
    embeddings = np.concatenate(embeddings, axis=0)

    embeddings = pd.DataFrame(embeddings)

    embeddings['ddG_exp'] = y_true
    embeddings['ddG_pred'] = y_pred
    embeddings['index'] = idx

    if return_emb == True:
        return y_pred, y_true, idx, embeddings
    else:
        return y_pred, y_true, idx


def generate_st_report(opt, 
                       loaders, 
                       model, 
                       model_params, 
                       inner = None,
                       outer =  None,):
    
    
    if opt.split_type == 'tvt':
        title = 'Training Report'
    elif opt.split_type == 'cv':
        title = 'Training Report Using Fold {} as Validation Set'.format(inner)
    elif opt.split_type == 'ncv':
        title = 'Report for Test Set Fold {} and Validation Set {}'.format(outer, inner)

    report = []

    report.append(f"{title}\n")
    report.append("====================\n\n")

    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]

    report.append("{}, {}\n".format(today_str, time))

    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]

    N_train, N_val, N_test = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    N_tot = N_train + N_val + N_test  

    model.load_state_dict(model_params)

    # Predict and get embeddings for training set
    y_pred_train, y_true_train, idx_train, emb_train = predict_network(model, train_loader, True)
    train_results = pd.DataFrame({f'real_{opt.target_variable_name}': y_true_train, f'predicted_{opt.target_variable_name}': y_pred_train, 'index': idx_train})
    train_results['set'] = 'Training'
    metrics_train, metrics_names = calculate_metrics(y_true_train, y_pred_train, task=opt.problem_type)

    report.append("Training set\n")
    report.append("Set size = {}\n".format(N_train))
    report.extend([f"{Metric} = {Value}\n" for Metric, Value in zip(metrics_names, metrics_train)])

    # Predict and get embeddings for validation set
    y_pred_val, y_true_val, idx_val, emb_val = predict_network(model, val_loader, True)
    val_results = pd.DataFrame({f'real_{opt.target_variable_name}': y_true_val, f'predicted_{opt.target_variable_name}': y_pred_val, 'index': idx_val})
    val_results['set'] = 'Validation'
    metrics_val, metrics_names = calculate_metrics(y_true_val, y_pred_val, task=opt.problem_type)

    report.append("Validation set\n")
    report.append("Set size = {}\n".format(N_val))
    report.extend([f"{Metric} = {Value}\n" for Metric, Value in zip(metrics_names, metrics_val)])

    # Predict and get embeddings for test set
    y_pred_test, y_true_test, idx_test, emb_test = predict_network(model, test_loader, True)
    test_results = pd.DataFrame({f'real_{opt.target_variable_name}': y_true_test, f'predicted_{opt.target_variable_name}': y_pred_test, 'index': idx_test})
    test_results['set'] = 'Test'
    metrics_test, metrics_names = calculate_metrics(y_true_test, y_pred_test, task=opt.problem_type)

    report.append("Test set\n")
    report.append("Set size = {}\n".format(N_test))
    report.extend([f"{Metric} = {Value}\n" for Metric, Value in zip(metrics_names, metrics_test)])

    report.append("\n\n")
    report.append("---------------------------------------------------------\n")
    report.append("\n\n")

    if opt.problem_type == 'regression':
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=y_true_train, 
                                 y=y_pred_train, 
                                 mode='markers', 
                                 name='Training set', 
                                 marker=dict(color='blue'),
                                 text=[f"Index: {idx}" for idx in idx_train],
                                 hoverinfo='text+x+y'))
        
        fig.add_trace(go.Scatter(x=y_true_val,
                                    y=y_pred_val,
                                    mode='markers',
                                    name='Validation set',
                                    marker=dict(color='orange'),
                                    text=[f"Index: {idx}" for idx in idx_val],
                                    hoverinfo='text+x+y'))
        
        fig.add_trace(go.Scatter(x=y_true_test,
                                    y=y_pred_test,
                                    mode='markers',
                                    name='Test set',
                                    marker=dict(color='green'),
                                    text=[f"Index: {idx}" for idx in idx_test],
                                    hoverinfo='text+x+y'))
        
        fig.add_trace(go.Scatter(x=[min(y_true_train), max(y_true_train)],
                                y=[min(y_true_train), max(y_true_train)],
                                mode='lines',
                                name='Parity Line',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(title='Parity Plot',
                            xaxis_title=f'Real {opt.target_variable_name} {opt.target_variable_units}',
                            yaxis_title=f'Predicted {opt.target_variable_name} {opt.target_variable_units}',
                            showlegend=True)
        
    
    
    if opt.show_all and opt.split_type != 'tvt':
        st.plotly_chart(fig)
        st.text("".join(report))

    # Combine embeddings for further use if needed
    emb_all = pd.concat([emb_train, emb_val, emb_test], axis=0)

    results_all = pd.concat([train_results, val_results, test_results], axis=0)

    if opt.split_type is not 'tvt':
        results_all['Inner_Fold'] = inner
        results_all['Outer_Fold'] = outer

    return results_all, report

    # Return the report as a string and the predictions/true values for plotting



def network_report(log_dir,
                   loaders,
                   outer,
                   inner, 
                   loss_lists,
                   save_all,
                   model, 
                   model_params,
                   best_epoch):


    #1) Create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    #2) Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    #3) Unfold loaders and save loaders and model
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val, N_test = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    N_tot = N_train + N_val + N_test     
    if save_all == True:
        torch.save(train_loader, "{}/train_loader.pth".format(log_dir))
        torch.save(val_loader, "{}/val_loader.pth".format(log_dir))
        torch.save(model, "{}/model.pth".format(log_dir))
        torch.save(model_params, "{}/model_params.pth".format(log_dir))
    torch.save(test_loader, "{}/test_loader.pth".format(log_dir)) 
    loss_function = 'RMSE_%'

    #4) loss trend during training
    train_list = loss_lists[0]
    val_list = loss_lists[1] 
    test_list = loss_lists[2]
    if train_list is not None and val_list is not None and test_list is not None:
        with open('{}/{}.csv'.format(log_dir, 'learning_process'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train_{}".format(loss_function), "Val_{}".format(loss_function), "Test_{}".format(loss_function)])
            for i in range(len(train_list)):
                writer.writerow([(i+1)*5, train_list[i], val_list[i], test_list[i]])
        create_training_plot(df='{}/{}.csv'.format(log_dir, 'learning_process'), save_path='{}'.format(log_dir))


    #5) Start writting report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Best epoch: {}\n".format(best_epoch))
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    model.load_state_dict(model_params)

    y_pred, y_true, idx, emb_train = predict_network(model, train_loader, True)
    emb_train['set'] = 'training'
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    
    file1.write("***************\n")
    y_pred, y_true, idx, emb_val = predict_network(model, val_loader, True)
    emb_val['set'] = 'val'
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx, emb_test = predict_network(model, test_loader, True)
    emb_test['set'] = 'test'

    emb_all = pd.concat([emb_train, emb_val, emb_test], axis=0)

    plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column='ddG_exp', set_column='set', fig_name='tsne_emb_exp', save_path=log_dir)
    #plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column='ddG_pred', set_column='set', fig_name='tsne_emb_pred', save_path=log_dir)
    emb_all.to_csv("{}/embeddings.csv".format(log_dir))

    pd.DataFrame({'real_ddG': y_true, 'predicted_ddG': y_pred, 'index': idx}).to_csv("{}/predictions_test_set.csv".format(log_dir))

    face_pred = np.where(y_pred > 0, 1, 0)
    face_true = np.where(y_true > 0, 1, 0)
    metrics, metrics_names = calculate_metrics(face_true, face_pred, task = 'c')

    correct_side_add = face_pred == face_true

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    #error = abs(y_pred-y_true)
    #y_true = y_true[correct_side_add]
    #y_pred = y_pred[correct_side_add]
    #idx = idx[correct_side_add]


    file1.write("Test Set Total Correct Face of Addition Predictions = {}\n".format(np.sum(correct_side_add)))

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))
    #create_it_parity_plot(real = y_true, predicted = y_pred, index = idx, figure_name='outer_{}_inner_{}.html'.format(outer, inner), save_path="{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")
    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
    abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []

    counter = 0

    for sample in range(len(y_pred)):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} kJ/mol    (index={})\n".format(counter, idx[sample], error_test[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.2f} kJ/mol    (index={})\n".format(counter, idx[sample], error_test[sample], sample))

    file1.close()

    return 'Report saved in {}'.format(log_dir)


def network_outer_report(log_dir: str,
                         outer: int,):
    
    
    accuracy, precision, recall, r2, mae, rmse = [], [], [], [], [], []

    files = [log_dir+f'Fold_{i}_val_set/performance.txt' for i in range(1, 11) if i != outer]

    # Define regular expressions to match metric lines
    accuracy_pattern = re.compile(r"Accuracy = (\d+\.\d+)")
    precision_pattern = re.compile(r"Precision = (\d+\.\d+)")
    recall_pattern = re.compile(r"Recall = (\d+\.\d+)")
    r2_pattern = re.compile(r"R2 = (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE = (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE = (\d+\.\d+)")
    
    for file in files:
        with open(os.path.join(file), 'r') as f:
            content = f.read()
        
        # Split the content by '*' to separate different sets
        sets = content.split('*')

        for set_content in sets:
            # Check if "Test set" is in the set content
            if "Test set" in set_content:
                # Extract metric values using regular expressions
                accuracy_match = accuracy_pattern.search(set_content)
                accuracy.append(float(accuracy_match.group(1)))
                precision_match = precision_pattern.search(set_content)
                precision.append(float(precision_match.group(1)))
                recall_match = recall_pattern.search(set_content)
                recall.append(float(recall_match.group(1)))
                r2_match = r2_pattern.search(set_content)
                try:
                    r2.append(float(r2_match.group(1)))
                except:
                    r2.append(0)
                mae_match = mae_pattern.search(set_content)
                mae.append(float(mae_match.group(1)))
                rmse_match = rmse_pattern.search(set_content)
                rmse.append(float(rmse_match.group(1)))

    # Calculate mean and standard deviation for each metric
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    precision_mean = np.mean(precision)
    precision_std = np.std(precision)
    recall_mean = np.mean(recall)
    recall_std = np.std(recall)
    r2_mean = np.mean(r2)
    r2_std = np.std(r2)
    mae_mean = np.mean(mae)
    mae_std = np.std(mae)
    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)

    # Write the results to the file
    file1 = open("{}/performance_outer_test_fold{}.txt".format(log_dir, outer), "w")
    file1.write("---------------------------------------------------------\n")
    file1.write("Test Set Metrics (mean ± std)\n")
    file1.write("Accuracy: {:.3f} ± {:.3f}\n".format(accuracy_mean, accuracy_std))
    file1.write("Precision: {:.3f} ± {:.3f}\n".format(precision_mean, precision_std))
    file1.write("Recall: {:.3f} ± {:.3f}\n".format(recall_mean, recall_std))
    file1.write("R2: {:.3f} ± {:.3f}\n".format(r2_mean, r2_std))
    file1.write("MAE: {:.3f} ± {:.3f}\n".format(mae_mean, mae_std))
    file1.write("RMSE: {:.3f} ± {:.3f}\n".format(rmse_mean, rmse_std))
    file1.write("---------------------------------------------------------\n")

    return 'Report saved in {}'.format(log_dir)

def extract_metrics(file):

    metrics = {'Accuracy': None, 'Precision': None, 'Recall': None, 'R2': None, 'MAE': None, 'RMSE': None}

    with open(file, 'r') as file:
            content = file.read()

    # Define regular expressions to match metric lines
    accuracy_pattern = re.compile(r"Accuracy: (\d+\.\d+) ± (\d+\.\d+)")
    precision_pattern = re.compile(r"Precision: (\d+\.\d+) ± (\d+\.\d+)")
    recall_pattern = re.compile(r"Recall: (\d+\.\d+) ± (\d+\.\d+)")
    r2_pattern = re.compile(r"R2: (\d+\.\d+) ± (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE: (\d+\.\d+) ± (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE: (\d+\.\d+) ± (\d+\.\d+)")

    accuracy_match = accuracy_pattern.search(content)
    precision_match = precision_pattern.search(content)
    recall_match = recall_pattern.search(content)
    r2_match = r2_pattern.search(content)
    mae_match = mae_pattern.search(content)
    rmse_match = rmse_pattern.search(content)

    # Update the metrics dictionary with extracted values
    if accuracy_match:
        metrics['Accuracy'] = {'mean': float(accuracy_match.group(1)), 'std': float(accuracy_match.group(2))}
    if precision_match:
        metrics['Precision'] = {'mean': float(precision_match.group(1)), 'std': float(precision_match.group(2))}
    if recall_match:
        metrics['Recall'] = {'mean': float(recall_match.group(1)), 'std': float(recall_match.group(2))}
    if r2_match:
        metrics['R2'] = {'mean': float(r2_match.group(1)), 'std': float(r2_match.group(2))}
    if mae_match:
        metrics['MAE'] = {'mean': float(mae_match.group(1)), 'std': float(mae_match.group(2))}
    if rmse_match:
        metrics['RMSE'] = {'mean': float(rmse_match.group(1)), 'std': float(rmse_match.group(2))}

    return metrics

######################################
######################################
######################################
######  traditional ML functions #####
######################################
######################################
######################################

def load_variables(path:str, descriptors:list):

    data = pd.read_csv(path)

    data = data.filter(descriptors)

    #remove erroneous data
    data = data.dropna(axis=0)


    X = data.drop(['ddG'], axis = 1)
    X = RobustScaler().fit_transform(np.array(X))
    y = data['ddG']
    print('Features shape: ', X.shape)
    print('Y target variable shape: ' , y.shape)

    return X, y, descriptors

def choose_model(best_params, algorithm):

    if best_params == None:
        if algorithm == 'rf':
            return RandomForestRegressor()
        if algorithm == 'lr':
            return LinearRegression()
        if algorithm == 'gb':
            return GradientBoostingRegressor()

    else:
        if algorithm == 'rf':
            return RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_leaf=best_params['min_samples_leaf'], 
                                     min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
        if algorithm == 'lr':
            return LinearRegression()
        if algorithm == 'gb':
            return GradientBoostingRegressor(loss = best_params['loss'], learning_rate=best_params['learning_rate'],n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                                             min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'], random_state=best_params['random_state'])
        

def hyperparam_tune(X, y, model, seed):

    np.random.seed(seed)

    print('ML algorithm to be tunned:', str(model))

    if str(model) == 'LinearRegression()':
        return None
    
    else: 
        if str(model) == 'RandomForestRegressor()':
            hyperP = dict(n_estimators=[100, 300, 500, 800], 
                        max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2, 5, 10, 15, 100],
                        min_samples_leaf=[1, 2, 5, 10],
                        random_state = [seed])
        elif str(model) == 'GradientBoostingRegressor()':
            hyperP = dict(loss=['squared_error'], learning_rate=[0.1, 0.2, 0.3],
                        n_estimators=[100, 300, 500, 800], max_depth=[None, 5, 8, 15, 25, 30],
                        min_samples_split=[2],
                        min_samples_leaf=[1, 2],
                        random_state = [seed])

        gridF = GridSearchCV(model, hyperP, cv=3, verbose=1, n_jobs=-1)
        bestP = gridF.fit(X, y)
        params = bestP.best_params_
        print('Best hyperparameters:', params, '\n')

        return params
    

def split_data(df:pd.DataFrame):

    '''
    splits a dataset in a given quantity of folds
    '''
        
    for outer in np.unique(df['fold']):
        proxy = copy(df)
        test = proxy[proxy['fold'] == outer]

        for inner in np.unique(df.loc[df['fold'] != outer, 'fold']):

            val = proxy.loc[proxy['fold'] == inner]
            train = proxy.loc[(proxy['fold'] != outer) & (proxy['fold'] != inner)]
            yield deepcopy((train, val, test))


def predict_tml(model, data:pd.DataFrame, descriptors:list):

    y_pred = model.predict(data[descriptors])
    y_true = list(data['ddG'])
    idx = list(data['index'])

    return np.array(y_pred), np.array(y_true), np.array(idx)


def tml_report(log_dir,
               outer, 
                inner,
               model, 
               data, 
               descriptors,
               save_all=True):
    
    #1) create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    #2) Get time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    #3) Unfold  train/val/test dataloaders
    train_data, val_data, test_data = data[0], data[1], data[2]
    N_train, N_val, N_test = len(train_data), len(val_data), len(test_data)
    N_tot = N_train + N_val + N_test 

    #4) Save dataframes for future use
    if save_all:
        train_data.to_csv("{}/train.csv".format(log_dir))
        val_data.to_csv("{}/val.csv".format(log_dir))
        pickle.dump(model, open("{}/model.sav".format(log_dir), 'wb'))

    test_data.to_csv("{}/test.csv".format(log_dir))

    #5) Performance Report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("Traditional ML algorithm Performance\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(model, train_data, descriptors)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    file1.write("***************\n")
    y_pred, y_true, idx = predict_tml(model, val_data, descriptors)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(model=model, data=test_data, descriptors = descriptors)

    pd.DataFrame({'real_ddG': y_true, 'predicted_ddG': y_pred, 'index': idx}).to_csv("{}/predictions_test_set.csv".format(log_dir))

    face_pred = np.where(y_pred > 0, 1, 0)
    face_true = np.where(y_true > 0, 1, 0)

    metrics, metrics_names = calculate_metrics(face_true, face_pred, task = 'c')

    correct_side_add = face_pred == face_true

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))
    
    #error = abs(y_pred-y_true)
    #y_true = y_true[correct_side_add]
    #y_pred = y_pred[correct_side_add]
    #idx = idx[correct_side_add]

    file1.write("Test Set Total Correct Face of Addition Predictions = {}\n".format(np.sum(correct_side_add)))

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))
    #create_it_parity_plot(real = y_true, predicted = y_pred, index = idx, figure_name='outer_{}_inner_{}.html'.format(outer, inner), save_path="{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")
    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
    abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []

    counter = 0

    for sample in range(len(y_pred)):
        if abs_error_test[sample] >= 3 * std_error_test:  
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write("0{}) {}    Error: {:.2f} kJ/mol    (index={})\n".format(counter, idx[sample], error_test[sample], sample))
            else:
                file1.write("{}) {}    Error: {:.2f} kJ/mol    (index={})\n".format(counter, idx[sample], error_test[sample], sample))

    file1.close()

    return 'Report saved in {}'.format(log_dir)


