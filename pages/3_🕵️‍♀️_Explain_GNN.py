import streamlit as st
import zipfile
import json
import torch
import io 
import argparse
import pandas as pd
from scripts_experiments.explain import explain_mols
from data.predict_unseen import predict_insilico
from torch_geometric.loader import DataLoader
from utils.utils_model import predict_network
from utils.plot_utils import plot_tsne_scatter_matrix
import numpy as np


st.set_page_config(page_title="Explain GNN", page_icon="🧪", layout="wide")


import streamlit as st

st.markdown("""
## 🔍 **Predict Molecular Properties and Explore Model Explainability**

Welcome to the **Prediction & Explainability** section! Here, you can upload a **zip file** generated from your previous training session to predict molecular properties for new molecules. Additionally, you can gain insights into how your model makes predictions using various explainability algorithms.

### 📁 **Step 1: Upload the Training Zip File**

To begin, upload the **zip file** you obtained after training your model. This file contains all the necessary data, such as the trained model and configuration files, to make predictions and run explainability analyses.

### 💻 **Step 2: Prepare Your CSV for Prediction**

Next, you’ll need a **CSV file** with the molecules you want to predict. Please follow these important guidelines to ensure the process works smoothly:

- The CSV must include at least one column with the **SMILES representations** of the molecules.
- The **column names** for the SMILES (and any other features) must exactly match those used during the training process.
- If additional features were included in the training dataset (e.g., molecular properties), these must also be present with the same column names in the new CSV.

### 🛠 **Step 3: Choose an Explainability Algorithm**

Once the predictions are ready, select an explainability algorithm to understand **how the model made its predictions**. The available options are:

- **t-SNE Analysis**: Visualizes how molecules are grouped in the model's latent space.
- **GNNExplainer**: Identifies important substructures (subgraphs) that contribute to the model's decision.
- **ShapleyValueSampling**: Provides a quantitative measure of how much each feature contributes to the prediction.
- *(More explainability algorithms will be added soon!)*

### 🆔 **Step 4: Select a Molecule for Explanation**

If your CSV contains **multiple molecules**, you can select a specific molecule for detailed explanation. Once a molecule is selected, the chosen algorithm will generate visualizations or other outputs to help you understand which features are most important in making the prediction.

### 🔧 **Step 5: Adjust Visualization Parameters**

To customize the output, you can modify the visualization settings, such as:

- **Scale factor**: Adjust the atom sizes in the visualizations.
- **Bond width**: Modify the width of the bonds between atoms.
- **Contrast threshold**: Set the contrast for highlighting important atomic features in the explanation.
- **Normalization**: Choose how to normalize the feature attributions (e.g., across all molecules or specific features).

### 🔑 **Interpreting Results**

The explainability outputs will provide insights into how the GNN model interprets molecular structures. For example, you might see which atoms or bonds are most influential for a particular property prediction. This can help you **better understand the molecular features** driving the model’s decisions, providing valuable guidance for both interpreting the predictions and further refining your molecular datasets.

""")



experiment = st.file_uploader("Upload your experiment zip file", type=['zip'])


if experiment is not None:
    # Read the uploaded zip file
    with zipfile.ZipFile(io.BytesIO(experiment.read())) as z:
        # Extract files
        json_file = [file for file in z.namelist() if file.endswith('.json')][0]

        if json_file is not None:
            with z.open(json_file) as f:
                # Read the json file
                data = json.load(f)
                st.success("Loaded json file.")
                args = argparse.Namespace(**data)


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

        split_type = data["split_type"]

        model_name = {}

        if split_type == "tvt":
            model_name["Train/Validation/Test Model"] = model_params[0]
        elif split_type == "cv":
            for i in range(data["folds"]-1):
                model_name[f"Inner Fold {i+2}"] = model_params[i]
        elif split_type == "ncv":
            for o in range(1, data["folds"]+1):
                for i in range(1, data["folds"]):
                    i += 1 if o <= i else i
                    model_name[f"Outer Fold {o} - Inner Fold {i}"] = model_params[i]
        
        explain_model = st.selectbox("Select the model to explain", list(model_name.keys()))
        model.load_state_dict(model_name[explain_model])

        mols_explain = st.file_uploader("Upload the molecules to explain", type=['csv'])

        if mols_explain is not None:
            st.success("Molecules uploaded successfully.")
            df = pd.read_csv(mols_explain)
            st.write(df)
            mol_id_col_insilico = st.selectbox("Select the column with the molecule IDs", df.columns)
            args.mol_id_col_insilico = mol_id_col_insilico

            for mol in data["mol_cols"]:
                if mol not in df.columns:
                    st.error(f"Column {mol} not found in the in-silico library, please check the column names.")
                    st.stop()
            
            for graph_feat in data["graph_features"].keys():
                if graph_feat not in df.columns:
                    st.error(f"Column {graph_feat} not found in the in-silico library.")
                    st.stop()

            ###


            if st.checkbox("Show t-SNE Analysis"):
                tsne_data = predict_insilico(df, args).process()
                loader_tsne = DataLoader(tsne_data)
                y_pred, y_true, idx, _, embeddings = predict_network(args, model, loader_tsne, True)
                embeddings = pd.concat([df, embeddings], axis=1)
                st.write(embeddings)
                color_by = st.selectbox("Select the column to color by", embeddings.columns, index=embeddings.columns.get_loc(f'predicted_{args.target_variable_name}'))
                num_components = st.radio("Select the number of components", [2, 3])
                perplexity = st.select_slider("Select the perplexity", np.arange(5, 50.2, 0.5).tolist(), value=25)

                args.color_by = color_by    
                args.num_components = num_components
                args.perplexity = perplexity

                plot_tsne_scatter_matrix(df = embeddings, feature_cols=[i for i in range(args.embedding_dim*2)], args=args)

            st.title("Explainability Analysis")
            
            

            if mol_id_col_insilico is not None:

                explain_mols_ID = st.multiselect("Select the molecules to explain", df[mol_id_col_insilico].unique())

                explain_mols_csv = df[df[mol_id_col_insilico].isin(explain_mols_ID)]
                explain_mols_csv = explain_mols_csv.reset_index(drop=True)

                st.write(explain_mols_csv)

                algorithm = st.selectbox("Select the explanation algorithm", ['GNNExplainer', 'ShapleyValueSampling'])

                if algorithm == 'ShapleyValueSampling':
                    st.warning("ShapleyValueSampling algorithm takes some time to compute. Please be patient.")


                molecule = st.selectbox("Select the molecule to explain", args.mol_cols)

                analysis = st.radio("Select the family of features to explain", ["Atom Identity", "Atom Degree", "Atom Hybridization", "Is Atom Aromatic?", "Is Atom in Ring?", "Atom Chirality"] +  ["All Features"])

                scale_factor = st.select_slider("Select the scale factor for atom size", options=np.arange(2, 10.9, 0.5).tolist(), value=5)

                bond_width = st.select_slider("Select the bond width", options=np.arange(1, 10.01, 0.5).tolist(), value=2)


                contrast_threshold = st.select_slider("Select the contrast threshold (this is a modulater of the contrast of how the atoms are displayed. Attributions larger than this number will be shown with stronger colours on the plot.)", options=np.arange(0, 1.001, 0.01).tolist(), value=.5)
                type_contrast = st.selectbox("Select the type of importance displayed. Continuous will show the continous values of importance while classification will display only the nodes considered important", ['Continous', 'Classification'], index=0)
                
                normalize_attributions = st.selectbox("Normalize the attributions based on", ['All', 'Molecule', 'Features'])

                if explain_mols_csv.shape[0] > 0 and algorithm is not None and molecule is not None and scale_factor is not None:
                    data = predict_insilico(explain_mols_csv, args).process()
                    args.algorithm = algorithm
                    args.explain_mol = molecule
                    args.scale_factor = scale_factor
                    args.normalize = normalize_attributions
                    args.analysis = analysis
                    args.contrast_threshold = contrast_threshold
                    args.type_contrast = type_contrast
                    args.bond_width = bond_width
                    explain_mols(opt=args, model=model, mol_graphs=data)




        
