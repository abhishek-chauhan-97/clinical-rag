import streamlit as st
import logging
from config import AVAILABLE_MODELS, AVAILABLE_SOURCES

logger = logging.getLogger(__name__)

def sidebar_settings():
    st.sidebar.header("⚙️ Settings")

    # Retriever default: 3 → you can change it
    top_k = st.sidebar.slider("Retriever: Top K", 1, 10, value=5)  
    logger.info(f"Sidebar | top_k set to {top_k}")

    # Default model → explicitly set instead of first in list
    default_model = "gemini-1.5-flash" if "gemini-1.5-flash" in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
    model_choice = st.sidebar.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(default_model))
    logger.info(f"Sidebar | model_choice set to {model_choice}")

    # Sources → default all instead of just local
    source_choice = st.sidebar.multiselect(
        "Source(s)",
        AVAILABLE_SOURCES,
        default=AVAILABLE_SOURCES  
    )
    logger.info(f"Sidebar | source_choice set to {source_choice}")

    return top_k, model_choice, source_choice
