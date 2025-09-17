import os
import streamlit as st

# Hugging Face Token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found. Please add it in Hugging Face Secrets.")
    st.stop()

# Default models
AVAILABLE_MODELS = [
    "google/flan-t5-base",   # ✅ works on free tier
    "google/flan-t5-large",  # ✅ free tier
    "google/flan-t5-xl"      # ✅ free tier
    # DO NOT add Mistral, Llama-2, or Claude here — they require paid endpoints.
]

# Default sources
AVAILABLE_SOURCES = ["Local Docs", "PubMed", "Web Search"]
