import streamlit as st
import logging

logger = logging.getLogger(__name__)

def sidebar_settings():
    st.sidebar.header("⚙️ Settings")

    top_k = st.sidebar.slider("Retriever: Top K", 1, 5, 3)
    logger.info(f"Sidebar | top_k set to {top_k}")

    model_choice = st.sidebar.selectbox(
        "Model",
        [
            "google/flan-t5-small",
            "google/flan-t5-base",   # may 404
            "mistralai/Mistral-7B-Instruct-v0.2"
        ],
        index=0
    )
    logger.info(f"Sidebar | model_choice set to {model_choice}")

    source_choice = st.sidebar.multiselect(
        "Source(s)",
        ["Local Docs", "PubMed", "Web Search"],
        default=["Local Docs"]
    )
    logger.info(f"Sidebar | source_choice set to {source_choice}")

    return top_k, model_choice, source_choice
