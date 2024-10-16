import streamlit as st
import os
import pandas as pd
import plotly.figure_factory as ff
import zipfile
import io
from options.base_options import BaseOptions
from scripts_experiments.hyp_opt import run_tune
from utils.dicts import short_to_long, long_to_short
from ray import tune
import json

st.markdown('''
### 🛠️ **Hyperparameter Optimization**

**Hyperparameter Optimization** involves systematically searching for the optimal set of hyperparameters that produces the best model performance. This process is crucial as it can significantly enhance the model's accuracy and efficiency on unseen data.

#### **Key Considerations:**
- **Overfitting Prevention**: It's essential to ensure that the optimization process does not use the test data. Overfitting to the training or validation set can lead to misleading high performance during development but poor results on new, unseen data.
- **Computational Cost**: This process can be computationally intensive. It's advised to monitor resource utilization and adjust the complexity of the search space based on available computational resources.

#### **Dataset Splitting**:
To safeguard the integrity of the model evaluation, it's critical that hyperparameter optimization only involves your training and validation data. Please select how you plan to split your dataset:
- **Train/Validation/Test Split**: Separate your data into three sets. The test set will be strictly reserved for final evaluation.
- **Cross-Validation**: Use this method for a more robust estimation by rotating the validation set within the training data.
- **Nested Cross-Validation**: Ideal for smaller datasets, this method provides an unbiased evaluation by having two nested loops of cross-validation.

#### **Select Split Type**:
Please choose your preferred method of data splitting below. The software will automatically ensure that the test data remains untouched during the optimization process to provide a fair evaluation of the model.

[Add UI element here for split type selection]

By carefully selecting your dataset splitting strategy and rigorously tuning hyperparameters, you can achieve the best model performance while avoiding the pitfalls of data leakage or overfitting. 

            '''
)


opt = BaseOptions().parse() 

with st.expander('Step 1: Select CSV file with SMILES and target variable to model'):

    csv_file = st.file_uploader("Choose a CSV file", type="csv")

    smiles_cols = []
    if csv_file:
        st.markdown("**Preview CSV file**")
        df = pd.read_csv(csv_file)
        st.write(df.head())
        smiles_cols = st.multiselect("Select SMILES columns", options=df.columns.tolist())
        st.write(f"Selected SMILES columns: {', '.join(smiles_cols)}" if smiles_cols else "No SMILES columns selected")

        graph_feats = {}

        if st.checkbox("Include Graph-level features"):

            graph_level_features = st.multiselect("Select graph-level features", options=df.columns.tolist())

            st.write("Select the molecules which have the following features:")


            for feature in graph_level_features:
                graph_feats[feature] = st.multiselect(feature, options=smiles_cols)

            one_hot_encode_feats = st.multiselect("Select features to one-hot encode", options=graph_level_features)
            ohe_pos_vals = {}

            for feature in one_hot_encode_feats:
                uni_vals = df[feature].unique().tolist()
                ohe_pos_vals[feature] = uni_vals
                
                st.write(f"Features to one-hot encode: {', '.join(one_hot_encode_feats)}")
            
        else:
            st.write("No graph-level features selected")
            graph_level_features = None
            one_hot_encode_feats = None
            ohe_pos_vals = None


        identifier_col = st.selectbox("Select column with the ID of each molecule", options=df.columns.tolist(), index=None)
        
        target_col = st.selectbox("Select target column", options=df.columns.tolist(), index=None)

        if target_col and pd.api.types.is_numeric_dtype(df[target_col]):
            st.write(f"Target variable '{target_col}' is numeric")
            unique_vals_target = df[target_col].nunique()
            if unique_vals_target < 20:
                st.warning("This looks like a classification problem is to be modeled.")
                suggested_problem_type = "Classification"
            else:
                st.warning("This looks like a regression problem is to be modeled.")
                suggested_problem_type = "Regression"
        elif target_col is None:
            st.error("Please select a target variable.")
        else:
            st.error(f"Target variable '{target_col}' is not numeric. Please select a numeric column.")
            suggested_problem_type = None

        if target_col:
            problems = ["Classification", "Regression"]
            problem_type = st.selectbox("Select the problem type", problems, index=problems.index(suggested_problem_type) if suggested_problem_type in problems else None)

            if problem_type == "Classification":
                n_classes = df[target_col].nunique()
                st.write(f"Number of classes: {n_classes}")
            else:
                n_classes = 1

        target_name = st.text_input("Target variable name", value=target_col)
        target_units = st.text_input("Target variable units", value="kJ/mol")



        if target_col and problem_type == "Regression":
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

with st.expander('Step 2: Select Hyperparameter Search Space', expanded=False):

    st.subheader("Select Model Architecture Hyperparameter Search Space")

    emb_dim_range = st.multiselect("Select range of embedding dimensions to be explored", options=[32, 64, 128, 256], default=[64,128])

    conv_operator = st.selectbox("Select convolutional operators to be explored", options=['GCNConv', 'GATConv', 'SAGEConv', 'ChebConv'], index=2)

    n_convs_range = st.multiselect("Select range of convolutional layers to be explored", options=list(range(1, 6)), default=[1,2,3])

    readout_layers_range = st.multiselect("Select range of readout layers to be explored", options=list(range(1, 6)), default=[1,2,3])

    pooling_range = st.multiselect("Select range of pooling operations to be explored", options=['mean', 'max', 'sum', 'mean/max'], default=['mean', 'max'])


    st.subheader("Select Model Training Hyperparameter Search Space")

    epochs_range = st.multiselect("Select range of epochs to be explored", options=list(range(50, 301, 50)), default=[300])

    if len(epochs_range) > 3:
        st.warning("Selecting a large range of epochs can significantly increase computational time.")
    
    learning_rate_range = st.multiselect("Select range of learning rates to be explored", options=[0.001, 0.005, 0.01, 0.05, 0.1], default=[0.01, 0.005, 0.001])

    if len(learning_rate_range) > 3:
        st.warning("Selecting a large range of learning rates can significantly increase computational time.")

    early_stopping_range = st.multiselect("Select range of early stopping epochs to be explored", options=[1, 2, 3, 4, 5], default=[2, 3, 4])
    

    batch_size_range = st.multiselect("Select range of batch sizes to be explored", options=[16, 32, 64], default=[32, 64])

    if len(batch_size_range) > 3:
        st.warning("Selecting a large range of batch sizes can significantly increase computational time.")

    split_type = st.selectbox("Select Splitting Type", ["Train-Validation-Test Split", "Cross Validation", "Nested Cross Validation"])
    if split_type == "Train-Validation-Test Split":
        val_set_ratio = st.number_input("Validation set ratio", min_value= 0.001, max_value=1., value=0.2)
        test_set_ratio = st.number_input("Test set ratio", min_value= 0.001, max_value=1., value=0.2)

    elif split_type == "Cross Validation" or split_type == "Nested Cross Validation":
        folds = st.number_input("Number of folds", min_value=3, max_value=10, value=5)
    split_method = st.selectbox("Select Split Method", ["Stratified Split", "Random Split"])

log_name = st.text_input("Give this run an experiment name", value="my_experiment")



if st.button("Run Hyperparameter Optimization"):

    configs = {
        # network architecture params
        'embedding_dim': tune.choice(emb_dim_range),
        'n_convolutions': tune.choice(n_convs_range),
        'readout_layers': tune.choice(readout_layers_range),
        'pooling': tune.choice(pooling_range),
        'network_name': long_to_short[conv_operator],

        # training params
        'epochs': tune.choice(epochs_range),
        'lr': tune.choice(learning_rate_range),
        'early_stopping': tune.choice(early_stopping_range),
        'batch_size': tune.choice(batch_size_range),
    }

    opt.split_type = long_to_short[split_type] # not tunable
    opt.split_method = long_to_short[split_method] # not tunable

    opt.filename = os.path.basename(csv_file.name) # to generate graphs
    opt.log_dir_results = os.path.join(log_name) # not tunable
    opt.experiment_name = log_name # not tunable
    opt.mol_cols = sorted(smiles_cols, key = str.lower) # to generate graphs

    opt.target_variable = target_col # to generate graphs
    opt.target_variable_name = target_name # to generate graphs
    opt.target_variable_units = target_units # to generate graphs
    opt.mol_id_col = identifier_col # to generate graphs

    opt.ohe_graph_feat = one_hot_encode_feats # to generate graphs
    opt.ohe_pos_vals = ohe_pos_vals # to generate graphs
    opt.graph_features = graph_feats # to generate graphs

    opt.problem_type = problem_type.lower() # not tunable
    opt.n_classes = n_classes # not tunable
    opt.network_name = long_to_short[conv_operator] # not tunable

    if split_type == "Train-Validation-Test Split":
        opt.val_size = val_set_ratio
        opt.test_size = test_set_ratio

    elif split_type == "Cross Validation" or split_type == "Nested Cross Validation":
        opt.folds = folds

    
    best_hyp, all_runs = run_tune(opt, configs, df)

    st.success("Hyperparameter optimization completed successfully!")

    # add non tunable params to best_hyp
    # spliting options
    best_hyp['split_type'] = long_to_short[split_type]
    best_hyp['split_method'] = long_to_short[split_method]

    if split_type == "Train-Validation-Test Split":
        best_hyp['val_size'] = val_set_ratio
        best_hyp['test_size'] = test_set_ratio
    else:
        best_hyp['folds'] = folds

    # dataset options
    best_hyp['filename'] = os.path.basename(csv_file.name)
    best_hyp['mol_cols'] = sorted(smiles_cols, key = str.lower)
    best_hyp['target_variable'] = target_col
    best_hyp['target_variable_name'] = target_name
    best_hyp['target_variable_units'] = target_units
    best_hyp['mol_id_col'] = identifier_col

    # graph options
    best_hyp['ohe_graph_feat'] = one_hot_encode_feats
    best_hyp['ohe_pos_vals'] = ohe_pos_vals
    best_hyp['graph_features'] = graph_feats

    # training options 
    best_hyp['optimizer'] = opt.optimizer
    best_hyp['scheduler'] = opt.scheduler
    best_hyp['step_size'] = opt.step_size
    best_hyp['gamma'] = opt.gamma
    best_hyp['min_lr'] = opt.min_lr

    # problem type
    best_hyp['problem_type'] = problem_type.lower()
    best_hyp['n_classes'] = n_classes

    # experiment name
    best_hyp['experiment_name'] = log_name

    best_hyp['global_seed'] = opt.global_seed



    json_file = json.dumps(best_hyp, indent=4)

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as f:
        f.writestr(f"best_hyperparameters_{opt.experiment_name}.json", json_file)
        f.writestr(f"all_runs_{opt.experiment_name}.csv", all_runs.to_csv(index=False))
        f.writestr(f"{opt.filename}", df.to_csv(index=False))

    zip_buffer.seek(0)

    st.download_button(
        label="Download Best Hyperparameters",
        data=zip_buffer,
        file_name=f"hyp_opt_{opt.experiment_name}.zip",
        mime="application/zip"
    )
