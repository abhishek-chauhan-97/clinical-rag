import os
import streamlit as st

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found. Please add it in Hugging Face Secrets.")
    st.stop()

# Default models
AVAILABLE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large"
]

# Default sources
AVAILABLE_SOURCES = ["Local Docs", "PubMed", "Web Search"]
