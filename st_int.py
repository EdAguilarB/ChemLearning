import streamlit as st
import os
import pandas as pd
from options.base_options import BaseOptions
from scripts_experiments.train_GNN import train_network_nested_cv

# Initialize BaseOptions
opt = BaseOptions().parse()

# Main GUI
st.title("Train GNN Model")

# Step 1: Upload CSV file with the data
st.markdown("### Select the CSV file with the data")
gnn_data_file = st.file_uploader("Upload CSV", type=["csv"])

if gnn_data_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp", gnn_data_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(gnn_data_file.getbuffer())

    # Load the data for preview
    df = pd.read_csv(temp_file_path)
    st.write("File uploaded and saved successfully!")
    st.write("Preview of the data:")
    st.write(df.head())

    # Step 2: Select SMILES Columns
    st.markdown("### Select SMILES Columns")
    smiles_columns = st.multiselect("Choose columns containing SMILES strings", df.columns.tolist())

    # Step 3: Specify Log Directory and Name
    st.markdown("### Log Directory and Name")
    gnn_log_dir = st.text_input("Log directory path")
    gnn_log_name = st.text_input("Log directory name")

    # Step 4: Customize Training Options
    st.markdown("### Training Options")

    opt.embedding_dim = st.number_input("Embedding size", value=opt.embedding_dim)
    opt.n_convolutions = st.number_input("Number of convolutions", value=opt.n_convolutions)
    opt.readout_layers = st.number_input("Number of readout layers", value=opt.readout_layers)
    opt.epochs = st.number_input("Number of epochs", value=opt.epochs)
    opt.batch_size = st.number_input("Training batch size", value=opt.batch_size)

    # Run the training process
    if st.button("Run Train GNN"):
        # Update options with selected values
        opt.root = "temp"  # Set to the temporary directory
        opt.filename = gnn_data_file.name
        opt.mol_cols = smiles_columns
        opt.log_dir_results = os.path.join(gnn_log_dir, gnn_log_name)

        # Start training
        st.write("Starting training...")
        train_network_nested_cv(opt)
        st.success("Training complete!")


