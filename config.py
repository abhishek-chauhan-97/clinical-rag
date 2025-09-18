import os
import streamlit as st

# Gemini Token
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please add it in your GitHub/Streamlit secrets.")
    st.stop()

# Default models (Gemini only)
AVAILABLE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# Default sources
AVAILABLE_SOURCES = ["Local Docs", "PubMed", "Web Search"]
