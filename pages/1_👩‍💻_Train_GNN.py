import streamlit as st
import os
import pandas as pd
import json
import zipfile
import io
import torch

# Assuming BaseOptions and the training function are already defined as per your original code.
import plotly.figure_factory as ff

from options.base_options import BaseOptions
from scripts_experiments.train_GNN import train_GNNet


st.set_page_config(page_title="Train GNN", page_icon="🧪", layout="wide")

terms_dict = {
    "Train-Validation-Test Split": "tvt",
    "Cross Validation": "cv",
    "Nested Cross Validation": "ncv",
    "Stratified Split": "stratified",
    "Random Split": "random"
}
    



# Initialize options
opt = BaseOptions().parse()



# Set the section title
st.markdown("## 👋 Welcome to the GNN Training Section!")

# Instructions with visuals and formatting
st.markdown("""
### 🧪 Train Your GNN Model on Chemical Data
In this section, you can **train a Graph Neural Network (GNN)** model using your own chemical dataset! Simply upload a CSV file containing the SMILES representations of molecules and the target property you want to predict. The model will learn from the molecular structures to predict this target property.

### 📁 Upload Your Dataset
Make sure your dataset follows these guidelines:
- **SMILES columns**: Each molecule's SMILES representation should be in a separate column.
- **Target property**: The property you want to predict should be in a separate column as well.

#### Example:
| SMILES_1         | SMILES_2        | Target_Property |
|------------------|-----------------|-----------------|
| CCO              | CCC             | 12.5            |
| CCN              | C1=CC=CC=C1     | 8.4             |
| ...              | ...             | ...             |

### 💻 Predict from an In-Silico Library
Once your model is trained, you can also use it to predict properties for a new set of molecules in an **in-silico library**. Simply upload the new dataset for prediction.

### 🫡 Final Step: Complete All Fields Before Training
Before you hit the **'Apply'** button, double-check that all required fields are filled in correctly. This ensures smooth training and avoids errors.

### 📥 Download Your Results
After the training is complete:
- You will be able to **download a detailed report** of the training process.
- The **trained model files** will be available for download. You’ll need these files if you plan to use the **Predict** or **Explain** tabs later.

---

Once you're ready, upload your data and click **'Apply'** to start training your GNN model!
""")


# Step 1: Select CSV file

with st.expander('Step 1: Select CSV file with SMILES and target variable to model'):

    csv_file = st.file_uploader("Choose a CSV file", type="csv")

    smiles_cols = []
    if csv_file:
        st.markdown("**Preview CSV file**")
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
        ohe_pos_vals = {}

        for feature in one_hot_encode_feats:
            uni_vals = df[feature].unique().tolist()
            ohe_pos_vals[feature] = uni_vals
            
            st.write(f"Features to one-hot encode: {', '.join(one_hot_encode_feats)}")

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

# Step 2: Data Splitting Options


with st.expander("Step 2: Set Data Splitting Options", expanded=False):
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

# Step 5: Training Options
with st.expander("Step 3: Set Training Options", expanded=False):
    embedding_dim = st.number_input("Embedding size", min_value=1, max_value=1024, value=opt.embedding_dim)
    n_convolutions = st.number_input("Number of convolutions", min_value=1, max_value=10, value=opt.n_convolutions)
    readout_layers = st.number_input("Number of readout layers", min_value=1, max_value=10, value=opt.readout_layers)
    epochs = st.number_input("Number of epochs", min_value=1, max_value=1000, value=opt.epochs)
    batch_size = st.number_input("Training batch size", min_value=1, max_value=512, value=opt.batch_size)



log_name = st.text_input("Give this run an experiment name", value="my_experiment")




# Apply button
if csv_file and log_name and smiles_cols:

    show_all = st.checkbox("Show all metrics for all training processes", value=True)

    if st.button("Apply and Train"):

        # Update opt with user inputs
        opt.show_all = show_all
        opt.filename = os.path.basename(csv_file.name)
        opt.log_dir_results = os.path.join(log_name)
        opt.experiment_name = log_name
        opt.mol_cols = sorted(smiles_cols, key = str.lower) 
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
        opt.ohe_pos_vals = ohe_pos_vals
        opt.graph_features = graph_feats

        if split_type == "Train-Validation-Test Split":
            opt.val_size = val_set_ratio
            opt.test_size = test_set_ratio

        elif split_type == "Cross Validation" or split_type == "Nested Cross Validation":
            opt.folds = folds

        # Train the GNN model
        model_params, model, report, results_seen = train_GNNet(opt, df)

        model_arch = model[0]

        st.success(f"Training started successfully with the following configuration:\n\n"
                    f"Data File: {csv_file.name}\n"
                    f"Log Name: {log_name}\n"
                    f"Embedding Size: {embedding_dim}\n"
                    f"Number of Convolutions: {n_convolutions}\n"
                    f"Number of Readout Layers: {readout_layers}\n"
                    f"Number of Epochs: {epochs}\n"
                    f"Batch Size: {batch_size}\n"
                    f"Selected SMILES Columns: {', '.join(smiles_cols)}")
        

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
            torch.save(model_arch, model_buffer)
            model_buffer.seek(0)
            zip_file.writestr(f"model_architecture.pt", model_buffer.read())

            zip_file.writestr(f"results_{opt.experiment_name}.csv", results_seen.to_csv(index=False))
        
        zip_buffer.seek(0)

        st.download_button(
                                label="Download Detailed Report and JSON",
                                data=zip_buffer,
                                file_name=f"{opt.experiment_name}_results.zip",
                                mime="application/zip"
                            )
else:
    st.warning("Please select all the training options to start the training.")

