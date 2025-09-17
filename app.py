import streamlit as st
from retriever import retrieve_top_k, DOCS
from generator import generate_answer
from ui import sidebar_settings
import traceback
import logging
import sys

st.set_page_config(page_title="Clinical GAI Prototype", layout="wide")
st.title("🧠 Clinical GAI Prototype")
st.caption("Disclaimer: Prototype only. Not for clinical use.")

# Sidebar
top_k, model_choice, source_choice = sidebar_settings()

# Input
user_query = st.text_input("Enter your medical question:")

if st.button("Search"):
    with st.spinner("Processing..."):
        try:
            ans, logs, refs = generate_answer(user_query, top_k, model_choice)
            if ans:
                st.success(ans)

                st.markdown("### 📚 References")
                for r in refs:
                    st.markdown(f"- [{r['id']}]({r['url']}) (score={r['score']:.3f})")

                st.markdown("### 🛠️ Logs")
                for l in logs:
                    st.text(l)
            else:
                st.error("❌ No answer generated. See logs below.")
        except Exception as e:
            st.error(f"❌ Exception: {e}")
            st.text(traceback.format_exc())

# ✅ Always log to both console and a file
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app_debug.log", mode="w")
    ]
)

logger = logging.getLogger(__name__)
logger.info("🚀 Logging initialized. If you don't see this line, logging is broken.")

try:
    logger.info("🔍 Running model...")
    response = llm(question)   # or however you're calling
    logger.debug(f"Raw response: {response}")
except Exception as e:
    logger.exception("❌ Model call failed")
    response = "⚠️ Error: model call failed. See logs."

