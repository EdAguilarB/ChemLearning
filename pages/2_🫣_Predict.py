import streamlit as st
import zipfile
import io
import json
import torch
import pandas as pd
import argparse
from scripts_experiments.predict import predict_mols

st.set_page_config(page_title="Predict Property", page_icon="🧪", layout="wide")


st.markdown("""
## 📊 Predict Molecular Properties with Your Trained GNN Model

Welcome to the **prediction** section! Here, you can upload a CSV file containing molecules for which you'd like to **predict properties** using the model you’ve already trained.

### 🧠 **Important: You Must Train a Model First!**

Before you use this feature, make sure you’ve already trained a GNN model. If you haven’t, please visit the **'Train GNN'** page to upload your dataset and train your model.

### 📝 **Prepare Your CSV for Prediction**

To make accurate predictions, your **CSV file** (in-silico library) must follow the same structure as the dataset you used for training. Please ensure the following:

- The **column names** in your new CSV **must match exactly** with the names from the training dataset, including the SMILES columns and any additional properties you may have included (e.g., molecular features or encoded values).
- Your CSV should contain the same number of **molecule columns** as in the original training dataset.
- If you included additional molecular features (e.g., stereochemistry, molecular descriptors) in the training file, those should also be included in your new CSV file and have the same column names.

#### Example Structure:
| SMILES_1         | SMILES_2        | ...             |
|------------------|-----------------|-----------------|
| CCO              | CCC             | ...             |
| CCN              | C1=CC=CC=C1     | ...             |

This ensures that the model can properly interpret the new dataset in the same way it did during training. Any discrepancies in column names or structure could lead to errors or inaccurate predictions.

Once your CSV is ready, simply upload it here, and the app will predict the properties of your molecules based on the trained model! 🚀
            
To start, upload your **experiment zip file** generated from your training experiments, and then upload your **in-silico library** for prediction.
""")


experiment = st.file_uploader("Upload your experiment zip file", type=['zip'])


if experiment is not None:
    # Read the uploaded zip file
    with zipfile.ZipFile(io.BytesIO(experiment.read())) as z:
        # Extract files
        st.write("Extracted the following files:")
        st.write(z.namelist())

        json_file = [file for file in z.namelist() if file.endswith('.json')][0]

        if json_file is not None:
            with z.open(json_file) as f:
                # Read the json file
                data = json.load(f)
                st.success("Loaded json file.")

        with z.open("model_architecture.pt") as f:
            # Load the model architecture
            model = torch.load(f)
            st.success("Loaded model architecture.")

        model_params_names = [file for file in z.namelist() if file.endswith('.pt') and not "model_architecture" in file]
        model_params = []

        for name in model_params_names:
            with z.open(name) as f:
                # Load the model parameters
                model_params.append(torch.load(f))
                st.success(f"Loaded model parameters for {name}.")

    if data is not None and model is not None and model_params is not None:

        # Upload the in-silico library
        insilico = st.file_uploader("Upload your in-silico library", type=['csv'])

        if insilico is not None:
            st.success("In-silico library uploaded successfully.")
            df = pd.read_csv(insilico)
            st.write(df)

            for mol in data["mol_cols"]:
                if mol not in df.columns:
                    st.error(f"Column {mol} not found in the in-silico library, please check the column names.")
                    st.stop()
            
            for graph_feat in data["graph_features"].keys():
                if graph_feat not in df.columns:
                    st.error(f"Column {graph_feat} not found in the in-silico library.")
                    st.stop()
            if data["ohe_graph_feat"]:
                for feat_ohe in data["ohe_graph_feat"]:
                    uni_vals = df[feat_ohe].unique()
                    for val in uni_vals:
                        if val not in data["ohe_pos_vals"][feat_ohe]:
                            st.error(f"Value {val} not found in the one-hot encoded values for {feat_ohe}.")
                            st.error("This means that this value has not been seen during training, and may cause mis-predictions.")

            st.title("In-Silico Library Prediction")
            args = argparse.Namespace(**data)

            mol_id_col_insilico = st.selectbox("Select the column with the molecule IDs", df.columns)

            if mol_id_col_insilico is not None:
                args.mol_id_col_insilico = mol_id_col_insilico

                if st.button("Predict Properties"):

                    results_insilico = predict_mols(args, df, model, model_params)


                    st.download_button(
                        label="Download Predictions",
                        data=results_insilico.to_csv(index=False),
                        file_name=f'predictions_{insilico.name}.csv',
                        mime='text/csv'
                    )

    



            





