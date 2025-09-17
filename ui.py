import streamlit as st
from config import AVAILABLE_MODELS, AVAILABLE_SOURCES

def sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    top_k = st.sidebar.slider("Retriever: Top K", 1, 5, 3)
    model_choice = st.sidebar.selectbox("Model", AVAILABLE_MODELS)
    source_choice = st.sidebar.multiselect("Source(s)", AVAILABLE_SOURCES, default=["Local Docs"])
    return top_k, model_choice, source_choice
