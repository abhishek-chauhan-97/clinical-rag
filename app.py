import streamlit as st
from retriever import retrieve_top_k, DOCS
from generator import generate_answer
from ui import sidebar_settings
import traceback
import logging
import sys

# ---- Logging ----
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app_debug.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)
logger.info("🚀 Logging initialized. If you don't see this line, logging is broken.")

# ---- Streamlit Config ----
st.set_page_config(page_title="Clinical GAI Prototype", layout="wide")
st.title("🧠 Clinical GAI Prototype")
st.caption("Disclaimer: Prototype only. Not for clinical use.")

# ---- Sidebar ----
top_k, model_choice, source_choice = sidebar_settings()

# ---- Input ----
st.subheader("Ask a medical question")
user_query = st.text_input("Enter your medical question:")

if not user_query:
    st.info(
        "💡 Example questions you can try:\n"
        "- What is the standard adult dosage of paracetamol?\n"
        "- How is anemia managed in pregnancy?\n"
        "- What are the side effects of ibuprofen?"
    )

if st.button("Search"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a question before searching.")
    else:
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
                    for l in logs:
                        st.text(l)
            except Exception as e:
                st.error(f"❌ Exception: {e}")
                st.text(traceback.format_exc())
                logger.exception("Unhandled exception in app")
