# app.py — simple RAG prototype (Gradio + HF Inference)
import os, json, numpy as np, requests
from sentence_transformers import SentenceTransformer
import gradio as gr
import os
os.makedirs(".streamlit", exist_ok=True)


# -------- load docs
def load_docs(path="data/docs.jsonl"):
    docs=[]
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    except Exception as e:
        # fallback tiny doc
        docs = [{"id":"sample_1","text":"Paracetamol is used for fever and pain. Standard adult dosing guidance is 500-1000 mg every 4-6 hours."}]
    return docs

DOCS = load_docs()
IDS = [d["id"] for d in DOCS]
TEXTS = [d["text"] for d in DOCS]

# -------- embedding model (startup)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBS = EMBED_MODEL.encode(TEXTS, convert_to_numpy=True, show_progress_bar=False)

# -------- simple cosine retriever
def retrieve_top_k(query, k=3):
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)[0]
    # cosine similarities
    norms = np.linalg.norm(DOC_EMBS, axis=1) * np.linalg.norm(q_emb)
    sims = np.dot(DOC_EMBS, q_emb) / (norms + 1e-10)
    idx = np.argsort(-sims)[:k]
    results = [{"id": IDS[i], "text": TEXTS[i], "score": float(sims[i])} for i in idx]
    return results

# -------- HF Inference API call (simple)
def hf_generate(prompt, model="google/flan-t5-small", max_tokens=256):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"error":"HF_TOKEN not found in environment. Set HF_TOKEN in your Space secrets."}
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        return {"error": f"HF API error {r.status_code}: {r.text}"}
    data = r.json()
    # many text-generation models return [{'generated_text': "..."}]
    if isinstance(data, list) and "generated_text" in data[0]:
        return {"text": data[0]["generated_text"]}
    # fallback: return stringified response
    return {"text": str(data)}

# -------- answer function
def answer(query, top_k=3):
    retrieved = retrieve_top_k(query, k=top_k)
    context = "\n\n---\n\n".join([f"[{r['id']}]\n{r['text']}" for r in retrieved])
    prompt = f"Use ONLY the context below to answer the question concisely. Cite sources in square brackets using the source IDs.\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    gen = hf_generate(prompt)
    if "error" in gen:
        return gen["error"] + "\n\nContext used:\n" + context
    out = gen["text"].strip()
    # append provenance summary
    prov = "\n\n---\nSources used:\n" + "\n".join([f"{r['id']} (score={r['score']:.3f})" for r in retrieved])
    return out + prov

# -------- Streamlit UI
import streamlit as st

st.set_page_config(page_title="Clinical Key AI Prototype", layout="wide")

st.title("🧠 Clinical Key AI Prototype")
st.write("**Disclaimer:** Prototype only. Not for clinical use.")

# Input fields
user_query = st.text_input("Enter your medical question:")
top_k = st.slider("Retriever: top K", min_value=1, max_value=5, value=3)

# Run button
if st.button("Search"):
    if user_query.strip():
        with st.spinner("Running retrieval + generation..."):
            result = answer(user_query, top_k=top_k)
        st.success("Answer generated successfully!")
        st.write(result)
    else:
        st.warning("Please enter a question.")

