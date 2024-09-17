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

    graphs_insilico = predict_insilico(df).process()
    loader = DataLoader(graphs_insilico)

    results_insilico = pd.DataFrame(columns=[f'predictions_{opt.target_variable_name}', f'real_values_{opt.target_variable_name}', 'ID', 'model'])

    for i in range(len(model)):
        model[i].load_state_dict(model_params[i])
        y_pred, y_true, idx, embs = predict_network(model[i], loader, True)
        results_model = pd.DataFrame({f'predictions_{opt.target_variable_name}': y_pred, f'real_values_{opt.target_variable_name}': y_true, 'ID': idx, 'model': i})
        results_insilico = pd.concat([results_insilico, results_model], axis=0)

    st.write('Results in-Silico library')

    if None not in y_true:

        st.write(results_insilico)

        par = go.Figure()

        par.add_trace(go.Scatter(x=y_true,
                                 y=y_pred,
                                 mode='markers',
                                 marker=dict(color='blue'),
                                 text=idx,
                                 hoverinfo='text',))
        
        st.plotly_chart(par, use_container_width=True)
    
    else:
        results_insilico = results_insilico.dropna(axis=1)

        if results_insilico.nunique() == 1:
            results_insilico = results_insilico.drop(columns=['model'])

            st.write(results_insilico)

            vio = px.violin(y=y_pred,
                            box = True,
                            points='all', 
                            title='In-Silico library property predictions')
            
            st.plotly_chart(vio, use_container_width=True)

        elif opt.split_type == 'cv' and opt.folds <= 5:

            results_insilico['model'] += 2

            mean_preds = results_insilico.groupby(['ID'], as_index=False).mean()[[f'predictions_{opt.target_variable_name}', 'ID']]
            mean_preds['model'] = 'Mean'

            meadian_preds = results_insilico.groupby(['ID'], as_index=False).median()[[f'predictions_{opt.target_variable_name}', 'ID']]
            meadian_preds['model'] = 'Median'

            results_insilico = pd.concat([results_insilico, mean_preds, meadian_preds], axis=0)

            st.write(results_insilico)

            vio = px.strip(results_insilico,
                           y=f'predictions_{opt.target_variable_name} / {opt.target_variable_units}',
                           color='model',
                           custom_data=['ID'],
                           title='In-Silico library property predictions',
                           )
            
            vio.update_traces(hovertemplate='<br>ID: %{customdata[0]}<br>' + opt.target_variable_name +  ' Value: %{y}<extra></extra>',)

            vio.update_layout(yaxis_title=f'Predicted {opt.target_variable_name} / {opt.target_variable_units}',
                                          width=800,)
            
            st.plotly_chart(vio, use_container_width=True)

        else:

            if opt.split_type == 'cv':
                results_insilico['model'] += 2

            elif opt.split_type == 'ncv':
                counter = 0

                for o in range(1, opt.outer_folds+1):
                    for i in range(1, opt.inner_folds):
                        i += 1 if o <= i else i
                        results_insilico.loc[results_insilico['model'] == counter, 'model'] = f'Outer Fold {o} - Inner Fold {i}'
                        counter += 1

            mean_preds = results_insilico.groupby(['ID'], as_index=False).mean()[[f'predictions_{opt.target_variable_name}', 'ID']]
            mean_preds['model'] = 'Mean'

            meadian_preds = results_insilico.groupby(['ID'], as_index=False).median()[[f'predictions_{opt.target_variable_name}', 'ID']]
            meadian_preds['model'] = 'Median'

            results_insilico = pd.concat([results_insilico, mean_preds, meadian_preds], axis=0)

            st.write(results_insilico)

            vio = px.strip(results_insilico,
                           y=f'predictions_{opt.target_variable_name} / {opt.target_variable_units}',
                           color='model',
                           custom_data=['ID'],
                           title='In-Silico library property predictions',
                           )
            
            vio.update_traces(hovertemplate='<br>ID: %{customdata[0]}<br>' + opt.target_variable_name +  ' Value: %{y}<extra></extra>',)

            vio.update_layout(yaxis_title=f'Predicted {opt.target_variable_name} / {opt.target_variable_units}',
                                          width=800,)
            
            st.plotly_chart(vio, use_container_width=True)