import streamlit as st
import os
import pandas as pd
import json
import zipfile
import io
import torch

# Assuming BaseOptions and the training function are already defined as per your original code.
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


from options.base_options import BaseOptions
from scripts_experiments.train_GNN import train_GNNet
from scripts_experiments.predict import predict_mols

from data.predict_unseen import predict_insilico

from torch_geometric.loader import DataLoader

from utils.utils_model import predict_network


terms_dict = {
    "Train-Validation-Test Split": "tvt",
    "Cross Validation": "cv",
    "Nested Cross Validation": "ncv",
    "Stratified Split": "stratified",
    "Random Split": "random"
}
    

def main():

    # Initialize options
    opt = BaseOptions().parse()

    st.title("ChemLearning")

    st.sidebar.image("references/CL.png", use_column_width=True)

    st.sidebar.markdown("### **Train GNN**")


    st.write("This section allows you to train a GNN model on your own dataset.")

    st.sidebar.write("Please follow the steps below to train the model.")
    st.sidebar.write("Make sure to fill in all the fields before clicking the 'Apply'")


    # Step 1: Select CSV file
    with st.sidebar.expander("Step 1: Select CSV file"):
        csv_file = st.file_uploader("Choose a CSV file", type="csv")
        st.write(csv_file.name if csv_file else "No file selected")

    # Step 2: Data Splitting Options
    with st.sidebar.expander("Step 2: Set Data Splitting Options"):
        split_type = st.selectbox("Select Splitting Type", ["Train-Validation-Test Split", "Cross Validation", "Nested Cross Validation"])
        if split_type == "Train-Validation-Test Split":
            val_set_ratio = st.number_input("Validation set ratio", min_value= 0.001, max_value=1., value=0.2)
            test_set_ratio = st.number_input("Test set ratio", min_value= 0.001, max_value=1., value=0.2)

        elif split_type == "Cross Validation" or split_type == "Nested Cross Validation":
            folds = st.number_input("Number of folds", min_value=3, max_value=10, value=5)

        split_method = st.selectbox("Select Split Method", ["Stratified Split", "Random Split"])

        if split_method == "Stratified Split":
            # hacer lo de numero de tractos en que dividir la variable continua par el stratified
            pass


    # Step 2: Enter Directory Path Manually
    with st.sidebar.expander("Step 3: Enter the log directory path"):
        st.text("Please enter the full path to the directory where you want to save the log files.")
        log_dir = st.text_input("Log Directory Path", value=os.getcwd())
        st.text("Please enter the name of the directory where you want to save the log files.")
        log_name = st.text_input("Log directory name", value="my_experiment")

    # Step 5: Training Options
    with st.sidebar.expander("Step 4: Set Training Options", expanded=False):
        embedding_dim = st.number_input("Embedding size", min_value=1, max_value=1024, value=opt.embedding_dim)
        n_convolutions = st.number_input("Number of convolutions", min_value=1, max_value=10, value=opt.n_convolutions)
        readout_layers = st.number_input("Number of readout layers", min_value=1, max_value=10, value=opt.readout_layers)
        epochs = st.number_input("Number of epochs", min_value=1, max_value=1000, value=opt.epochs)
        batch_size = st.number_input("Training batch size", min_value=1, max_value=512, value=opt.batch_size)

    # Step 5: Training Options
    with st.sidebar.expander("Step 5: Predict Property of in silico library", expanded=False):

        in_silico_mols = st.file_uploader("Choose a CSV file", type="csv", key="in_silico_mols")
        if in_silico_mols:
            df_insilico = pd.read_csv(in_silico_mols)

        st.write(in_silico_mols.name if in_silico_mols else "No file selected")

        model_arch = st.file_uploader("Upload a model architecture file")
        model_pars = st.file_uploader("Upload the model best parameters")


    smiles_cols = []
    if csv_file:
        st.header("Preview CSV file")
        df = pd.read_csv(csv_file)
        st.write(df.head())
        smiles_cols = st.multiselect("Select SMILES columns", options=df.columns.tolist())
        st.write(f"Selected SMILES columns: {', '.join(smiles_cols)}" if smiles_cols else "No SMILES columns selected")

        graph_level_features = st.multiselect("Select graph-level features", options=df.columns.tolist())

        st.write("Select the molecules which have the following features:")

        graph_feats = {}

        for feature in graph_level_features:
            graph_feats[feature] = st.multiselect(feature, options=smiles_cols)

        one_hot_encode_feats = st.multiselect("Select features to one-hot encode", options=graph_level_features)

        identifier_col = st.selectbox("Select column with the ID of each molecule", options=df.columns.tolist())
        target_col = st.selectbox("Select target column", options=df.columns.tolist())
        target_name = st.text_input("Target variable name", value=target_col)
        target_units = st.text_input("Target variable units", value="kJ/mol")

        if target_col:
            st.header(f"Distribution of {target_name}")

            min_val_target = df[target_col].min()
            max_val_target = df[target_col].max()

            max_num_bins = 30
            min_num_bins = 5

            dif_max_min = max_val_target - min_val_target

            min_val_hist = dif_max_min / max_num_bins
            max_val_hist = dif_max_min / min_num_bins
            mid_val_hist = (max_val_hist + min_val_hist) / 2

            bin_size = st.slider("Bin size", min_value=min_val_hist, max_value=max_val_hist, value=mid_val_hist)
            
            # Create histogram
            fig = ff.create_distplot([df[target_col].dropna()], group_labels=[target_col], bin_size=bin_size)
            st.plotly_chart(fig)

            opt.root = log_dir
            opt.filename = os.path.basename(csv_file.name)
            opt.log_dir_results = os.path.join(log_dir, log_name)
            opt.experiment_name = log_name
            opt.mol_cols = smiles_cols
            opt.embedding_dim = embedding_dim
            opt.n_convolutions = n_convolutions
            opt.readout_layers = readout_layers
            opt.epochs = epochs
            opt.batch_size = batch_size

            opt.split_type = terms_dict[split_type]
            opt.split_method = terms_dict[split_method]

            opt.target_variable = target_col
            opt.target_variable_name = target_name
            opt.target_variable_units = target_units
            opt.mol_id_col = identifier_col

            opt.ohe_graph_feat = one_hot_encode_feats
            opt.graph_features = graph_feats


    if in_silico_mols:

        st.write(df_insilico)

        identifier_col = st.selectbox("Select column with the ID of each molecule for in silico library", options=df_insilico.columns.tolist())

        opt.mol_id_col_insilico = identifier_col


    # Apply button
    if csv_file and log_dir and log_name and smiles_cols:

        show_all = st.checkbox("Show all metrics for all training processes", value=True)

        if st.button("Apply and Train"):

            # Update opt with user inputs
            opt.show_all = show_all

            if split_type == "Train-Validation-Test Split":
                opt.val_size = val_set_ratio
                opt.test_size = test_set_ratio

            elif split_type == "Cross Validation" or split_type == "Nested Cross Validation":
                opt.folds = folds

            # Train the GNN model
            model_params, model, report = train_GNNet(opt, df)


            st.success(f"Training started successfully with the following configuration:\n\n"
                        f"Data File: {csv_file.name}\n"
                        f"Log Directory: {log_dir}\n"
                        f"Log Name: {log_name}\n"
                        f"Embedding Size: {embedding_dim}\n"
                        f"Number of Convolutions: {n_convolutions}\n"
                        f"Number of Readout Layers: {readout_layers}\n"
                        f"Number of Epochs: {epochs}\n"
                        f"Batch Size: {batch_size}\n"
                        f"Selected SMILES Columns: {', '.join(smiles_cols)}")
            



            if in_silico_mols:

                st.title("In-Silico Library Predictions")

                predict_mols(opt, df_insilico, model, model_params)


            json_data = vars(opt)        

            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                # Write the TXT file
                report_txt = "".join(report)  # Combine list of strings into a single string
                zip_file.writestr(f"report_all_{opt.experiment_name}.txt", report_txt)
                
                # Write the JSON file
                json_str = json.dumps(json_data, indent=4)
                zip_file.writestr(f"hyperparameters_{opt.experiment_name}.json", json_str)

                for model_idx, model in enumerate(model_params):
                    model_buffer = io.BytesIO()
                    torch.save(model, model_buffer)
                    model_buffer.seek(0)
                    zip_file.writestr(f"model_params_{model_idx}.pt", model_buffer.read())

                model_buffer = io.BytesIO()
                torch.save(model, model_buffer)
                model_buffer.seek(0)
                zip_file.writestr(f"model_architecture.pt", model_buffer.read())
            
            zip_buffer.seek(0)

            st.download_button(
                                    label="Download Detailed Report and JSON",
                                    data=zip_buffer,
                                    file_name=f"{opt.experiment_name}_results.zip",
                                    mime="application/zip"
                                )




    else:
        st.write("Please select all the training options to start the training.")


if __name__ == "__main__":
    main()




