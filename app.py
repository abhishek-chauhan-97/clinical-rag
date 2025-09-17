import os, json, requests, numpy as np, streamlit as st
from sentence_transformers import SentenceTransformer

# ---- Secrets ----
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN not found. Please add it in Hugging Face Secrets.")
    st.stop()

# ---- Config ----
st.set_page_config(page_title="Clinical GAI Prototype", layout="wide")
st.title("🧠 Clinical GAI Prototype")
st.caption("Disclaimer: Prototype only. Not for clinical use.")

# ---- Sidebar Settings ----
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Retriever: Top K", 1, 5, 3)
model_choice = st.sidebar.selectbox("Model", ["google/flan-t5-small", "mistralai/Mistral-7B-Instruct-v0.2"])
source_choice = st.sidebar.selectbox("Source", ["Local Docs", "PubMed", "Web Search"])

# ---- Load Docs (local fallback) ----
def load_docs(path="data/docs.jsonl"):
    docs=[]
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    except:
        docs=[{"id":"sample_1","text":"Paracetamol is used for fever and pain. Standard adult dosing guidance is 500-1000 mg every 4-6 hours.","url":"local"}]
    return docs

DOCS = load_docs()
IDS = [d["id"] for d in DOCS]
TEXTS = [d["text"] for d in DOCS]

# ---- Embeddings ----
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBS = EMBED_MODEL.encode(TEXTS, convert_to_numpy=True)

def retrieve_top_k(query, k=3):
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(DOC_EMBS, q_emb) / (np.linalg.norm(DOC_EMBS, axis=1) * np.linalg.norm(q_emb) + 1e-10)
    idx = np.argsort(-sims)[:k]
    return [{"id": IDS[i], "text": TEXTS[i], "score": float(sims[i]), "url": DOCS[i].get("url","local")} for i in idx]

def hf_generate(prompt, model="google/flan-t5-small", max_tokens=256):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        return {"error": f"HF API error {r.status_code}: {r.text}"}
    data = r.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return {"text": data[0]["generated_text"]}
    return {"text": str(data)}

def answer(query, top_k=3):
    logs=[]
    logs.append(f"User query: {query}")

    # retrieval
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} documents")

    # build context
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"Use ONLY the context below to answer. Cite sources in [ID].\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    logs.append("Built prompt for model.")

    gen = hf_generate(prompt, model_choice)
    if "error" in gen:
        return None, logs, retrieved
    logs.append("Model responded successfully.")
    return gen["text"].strip(), logs, retrieved

# ---- UI ----
user_query = st.text_input("Enter your medical question:")
if st.button("Search"):
    with st.spinner("Processing..."):
        ans, logs, refs = answer(user_query, top_k)
        if ans:
            st.success(ans)
            st.markdown("### 📚 References")
            for r in refs:
                st.markdown(f"- [{r['id']}]({r['url']}) (score={r['score']:.3f})")
            st.markdown("### 🛠️ Logs")
            for l in logs:
                st.text(l)
        else:
            st.error("❌ Something went wrong.")
