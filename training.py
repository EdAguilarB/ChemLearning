import streamlit as st

# Step 4: Customize Training Options in the Sidebar
with st.sidebar.expander("Step 5: Set Training Options", expanded=True):
    st.header("Training Options")

    with st.expander("Basic Settings", expanded=True):
        embedding_dim = st.number_input("Embedding size", min_value=1, max_value=1024, value=5)
        n_convolutions = st.number_input("Number of convolutions", min_value=1, max_value=10, value=5)

    with st.expander("Advanced Settings"):
        readout_layers = st.number_input("Number of readout layers", min_value=1, max_value=10, value=5)
        epochs = st.number_input("Number of epochs", min_value=1, max_value=1000, value=5)
        batch_size = st.number_input("Training batch size", min_value=1, max_value=512, value=5)


