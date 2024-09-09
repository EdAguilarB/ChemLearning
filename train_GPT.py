import streamlit as st
import os
import pandas as pd

# Assuming BaseOptions and the training function are already defined as per your original code.
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


from options.base_options import BaseOptions
from scripts_experiments.train_GNN import train_GNNet


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


    # Step 2: Enter Directory Path Manually
    with st.sidebar.expander("Step 2: Enter the log directory path"):
        st.text("Please enter the full path to the directory where you want to save the log files.")
        log_dir = st.text_input("Log Directory Path", value=os.getcwd())
        st.text("Please enter the name of the directory where you want to save the log files.")
        log_name = st.text_input("Log directory name", value="my_experiment")

    # Step 5: Training Options
    with st.sidebar.expander("Step 5: Set Training Options", expanded=False):
        embedding_dim = st.number_input("Embedding size", min_value=1, max_value=1024, value=opt.embedding_dim)
        n_convolutions = st.number_input("Number of convolutions", min_value=1, max_value=10, value=opt.n_convolutions)
        readout_layers = st.number_input("Number of readout layers", min_value=1, max_value=10, value=opt.readout_layers)
        epochs = st.number_input("Number of epochs", min_value=1, max_value=1000, value=opt.epochs)
        batch_size = st.number_input("Training batch size", min_value=1, max_value=512, value=opt.batch_size)

    smiles_cols = []
    if csv_file:
        st.header("Preview CSV file")
        df = pd.read_csv(csv_file)
        st.write(df.head())
        smiles_cols = st.multiselect("Select SMILES columns", options=df.columns.tolist())
        st.write(f"Selected SMILES columns: {', '.join(smiles_cols)}" if smiles_cols else "No SMILES columns selected")
        identifier_col = st.selectbox("Select column with the ID of each molecule", options=df.columns.tolist())
        target_col = st.selectbox("Select target column", options=df.columns.tolist())
        target_name = st.text_input("Target variable name", value=target_col)
        target_units = st.text_input("Target variable units", value="kJ/mol")

        if target_col:
            st.header(f"Distribution of {target_name}")

            min_val_target = df[target_col].min()
            max_val_target = df[target_col].max()

            max_num_bins = 20
            min_num_bins = 5

            dif_max_min = max_val_target - min_val_target

            min_val_hist = dif_max_min / max_num_bins
            max_val_hist = dif_max_min / min_num_bins
            mid_val_hist = (max_val_hist + min_val_hist) / 2

            bin_size = st.slider("Bin size", min_value=min_val_hist, max_value=max_val_hist, value=mid_val_hist)
            
            # Create histogram
            fig = ff.create_distplot([df[target_col].dropna()], group_labels=[target_col], bin_size=bin_size)
            st.plotly_chart(fig)

    # Apply button
    if csv_file and log_dir and log_name and smiles_cols:

        show_all = st.checkbox("Show all metrics for all training processes", value=True)

        if st.button("Apply and Train"):

            # Update opt with user inputs

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

            opt.show_all = show_all



            if split_type == "Train-Validation-Test Split":
                opt.val_size = val_set_ratio
                opt.test_size = test_set_ratio

            elif split_type == "Cross Validation" or split_type == "Nested Cross Validation":
                opt.folds = folds

            # Train the GNN model
            train_GNNet(opt, df)

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
        #else:
        #    st.error("Please ensure all fields are filled and SMILES columns are selected!")
    else:
        st.write("Please select all the training options to start the training.")

if __name__ == "__main__":
    main()




