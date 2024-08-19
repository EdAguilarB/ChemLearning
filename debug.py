import streamlit as st
import pandas as pd

st.title("Train GNN Model")

# File uploader for the CSV data file
gnn_data_file = st.file_uploader("Select the CSV file with the data", type=["csv"])

if gnn_data_file is not None:
    # Display file details
    st.write("Filename:", gnn_data_file.name)
    st.write("Filetype:", gnn_data_file.type)
    st.write("Filesize:", gnn_data_file.size, "bytes")

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(gnn_data_file)
        st.write(df.head())  # Display the first few rows of the DataFrame
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
