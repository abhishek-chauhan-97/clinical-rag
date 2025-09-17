import streamlit as st
from retriever import retrieve_top_k, DOCS
from generator import generate_answer
from ui import sidebar_settings
import traceback

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
