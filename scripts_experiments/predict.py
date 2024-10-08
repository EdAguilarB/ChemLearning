import streamlit as st
import pandas as pd
from data.predict_unseen import predict_insilico
from utils.utils_model import predict_network
from torch_geometric.loader import DataLoader
import plotly.graph_objects as go
import plotly.express as px
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def predict_mols(opt, df:pd.DataFrame, model, model_params):

    """
    Predicts the target variable of the molecules in the in-silico library using the trained model.
    Args:
    opt: Namespace object with the options used in the training process.
    df: DataFrame with the in-silico library.
    model: List with models with the architecture chosen.
    model_params: List with the parameters of the models.
    Returns:
    results_insilico: DataFrame with the predictions of the target variable of the in-silico library.
    """

    graphs_insilico = predict_insilico(data = df, opt=opt)
    graphs_insilico = graphs_insilico.process()
    loader = DataLoader(graphs_insilico)

    results_insilico = pd.DataFrame(columns=[f'real_{opt.target_variable_name}', f'predicted_{opt.target_variable_name}', opt.mol_id_col_insilico, 'model'])

    for i in range(len(model_params)):
        model.load_state_dict(model_params[i])
        y_pred, y_true, idx, _ = predict_network(opt, model, loader, False)
        results_model = pd.DataFrame({f'real_{opt.target_variable_name}': y_true, f'predicted_{opt.target_variable_name}': y_pred,  opt.mol_id_col_insilico: idx, 'model': i})
        results_insilico = pd.concat([results_insilico, results_model], axis=0)

    if opt.split_type == 'tvt':
        results_insilico = results_insilico.drop(columns=['model'])

    else:
        mean_preds = results_insilico.groupby([opt.mol_id_col_insilico], as_index=False).mean()
        mean_preds['model'] = 'Mean'

        meadian_preds = results_insilico.groupby([opt.mol_id_col_insilico], as_index=False).median()
        meadian_preds['model'] = 'Median'
    
        if opt.split_type == 'cv':
            results_insilico['model'] += 2
            results_insilico['model'] = 'Inner Fold ' + results_insilico['model'].astype(str)
        
        elif opt.split_type == 'ncv':
            counter = 0
            for o in range(1, opt.folds+1):
                for i in range(1, opt.folds):
                    i += 1 if o <= i else i
                    results_insilico.loc[results_insilico['model'] == counter, 'model'] = f'Outer Fold {o} - Inner Fold {i}'
                    counter += 1
        
        results_insilico = pd.concat([results_insilico, mean_preds, meadian_preds], axis=0)

    if None not in y_true and opt.problem_type == 'regression':

        if opt.split_type == 'tvt':

            pre = px.scatter(results_insilico,
                            x=f'real_{opt.target_variable_name}',
                            y=f'predicted_{opt.target_variable_name}',
                            custom_data=[opt.mol_id_col_insilico],
                            title='In-Silico library property predictions',
                            )

        elif opt.split_type == 'cv' and opt.folds <= 5:

            pre = px.scatter(results_insilico,
                            x=f'real_{opt.target_variable_name}',
                            y=f'predicted_{opt.target_variable_name}',
                            color='model',
                            custom_data=[opt.mol_id_col_insilico],
                            title='In-Silico library property predictions',
                            )
            
        else:

            pre = px.scatter(results_insilico.loc[(results_insilico['model'] == 'Mean') | (results_insilico['model'] == 'Median')],
                            x=f'real_{opt.target_variable_name}',
                            y=f'predicted_{opt.target_variable_name}',
                            color='model',
                            custom_data=[opt.mol_id_col_insilico],
                            title='In-Silico library property predictions',
                            )

        pre.update_layout(xaxis_title=f'Real {opt.target_variable_name} / {opt.target_variable_units}',
                            yaxis_title=f'Predicted {opt.target_variable_name} / {opt.target_variable_units}',
                            width=800,)
            
    elif opt.problem_type == 'regression':

        results_insilico = results_insilico.dropna(axis=1)

        if opt.split_type == 'tvt':

            pre = px.violin(results_insilico,
                            y=f'predicted_{opt.target_variable_name}',
                            box=True,
                            points='all',
                            custom_data=[opt.mol_id_col_insilico],
                            title='In-Silico library property predictions',
                            )

        elif opt.split_type == 'cv' and opt.folds <= 5:

            pre = px.strip(results_insilico,
                           y=f'predicted_{opt.target_variable_name}',
                           color='model',
                           custom_data=[opt.mol_id_col_insilico],
                           title='In-Silico library property predictions',
                           )

        else:

            pre = px.strip(results_insilico.loc[(results_insilico['model'] == 'Mean') | (results_insilico['model'] == 'Median')],
                           y=f'predicted_{opt.target_variable_name}',
                           color='model',
                           custom_data=[opt.mol_id_col_insilico],
                           title='In-Silico library property predictions',
                           )
        
        pre.update_layout(yaxis_title=f'Predicted {opt.target_variable_name} / {opt.target_variable_units}',
                                          width=800,)

    
    if opt.problem_type == 'regression':
        pre.update_traces(hovertemplate='<br>ID: %{customdata[0]}<br>' + opt.target_variable_name +  ' Value: %{y}<extra></extra>',)
        st.plotly_chart(pre, use_container_width=True)
        
    st.write(results_insilico)

    return results_insilico