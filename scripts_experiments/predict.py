import streamlit as st
import pandas as pd
from data.predict_unseen import predict_insilico
from utils.utils_model import predict_network
from torch_geometric.loader import DataLoader
import plotly.graph_objects as go

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
                                 hoverinfo='text',
                                 marginal_x='violin',
                                 marginal_y='violin',))
        
        st.plotly_chart(par, use_container_width=True)
        pass

    pass