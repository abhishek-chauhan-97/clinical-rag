import streamlit as st
import logging
from generator import AVAILABLE_MODELS

logger = logging.getLogger(__name__)

def sidebar_settings():
    st.sidebar.header("⚙️ Settings")

    top_k = st.sidebar.slider("Retriever: Top K", 1, 5, 3)
    logger.info(f"Sidebar | top_k set to {top_k}")

    model_choice = st.sidebar.selectbox("Model", AVAILABLE_MODELS) 
    logger.info(f"Sidebar | model_choice set to {model_choice}")

    source_choice = st.sidebar.multiselect(
        "Source(s)",
        ["Local Docs", "PubMed", "Web Search"],
        default=["Local Docs"]
    )
    logger.info(f"Sidebar | source_choice set to {source_choice}")

    return top_k, model_choice, source_choice

# ui.py (add near sidebar_settings or where appropriate)
from generator import check_model_availability

# After user selects model in sidebar_settings, optionally show:
if st.sidebar.button("Check model availability"):
    code, body = check_model_availability(model_choice)
    if code is None:
        st.sidebar.error("Check failed: see logs.")
    elif code == 200:
        st.sidebar.success(f"Model reachable (HTTP 200).")
    else:
        st.sidebar.warning(f"Model check: HTTP {code}")
        st.sidebar.text(body)

