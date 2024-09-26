import streamlit as st

# Set page title and layout
st.set_page_config(page_title="ChemGNN Trainer", page_icon="🧪", layout="wide")



# Main title
st.title("Welcome to ChemLearning Trainer!🤩")

# Load and display the logo
st.image("references/CL.png", width=200)

# Description of the app
st.markdown("""
This app allows chemists to **train Graph Neural Networks (GNNs)** on their own molecular datasets to predict properties of interest (yes, on your own dataset! 🥳). 
Explore the capabilities of this tool using your custom datasets, and generate insights through predictive modeling and explainability.

### How to Prepare Your Dataset
Before you get started, ensure that your dataset is properly structured in a CSV file. Here are the key requirements:
- **SMILES columns**: Each molecule's SMILES representation must be in a separate column.
- **Target property column**: The property you wish to predict (e.g., solubility, toxicity, etc.) must be in a separate column.
  
#### Example CSV Structure:
| SMILES_1         | SMILES_2        | Target_Property |
|------------------|-----------------|-----------------|
| CCO              | CCC             | 12.5            |
| CCN              | C1=CC=CC=C1     | 8.4             |
| ...              | ...             | ...             |

### Navigation
On the **left side menu**, you will find three main pages to guide you through the process:
1. **1_train_GNN**: Here, you can upload your dataset and begin training your GNN models. You'll be able to set your hyperparameters and monitor the training process.
2. **2_predict**: Once your model is trained, you can use this page to upload a new dataset for **prediction**. Simply upload a CSV file with the molecules of interest, and the app will predict the properties based on the trained model.
3. **3_explain_GNN**: To understand the inner workings of your GNN, this page provides **explainability plots**. You can explore how the model arrived at certain predictions, helping you gain insight into the molecular features driving the predictions.

### Get Started!
Select a page from the menu to start training your GNN models or make predictions.
""")

# Footer
st.markdown("""
---
This application was designed to empower chemists to leverage the power of **Graph Neural Networks** in their molecular studies. 
""")





