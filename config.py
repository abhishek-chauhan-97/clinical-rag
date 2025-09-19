import os
import streamlit as st

# Gemini Token
raw_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
GEMINI_API_KEY = raw_key.strip() if raw_key else None

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please add it in GitHub/Streamlit secrets.")
    st.stop()

# Default models
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

# Default sources
AVAILABLE_SOURCES = ["Local Docs", "PubMed", "Web Search"]
