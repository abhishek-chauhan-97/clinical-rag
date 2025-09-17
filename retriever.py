import json, numpy as np
from sentence_transformers import SentenceTransformer

def load_docs(path="data/docs.jsonl"):
    docs=[]
    try:
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    except:
        docs=[{"id":"sample_1",
               "text":"Paracetamol is used for fever and pain. Standard adult dosing guidance is 500-1000 mg every 4-6 hours.",
               "url":"local"}]
    return docs

DOCS = load_docs()
IDS = [d["id"] for d in DOCS]
TEXTS = [d["text"] for d in DOCS]

# Embeddings
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DOC_EMBS = EMBED_MODEL.encode(TEXTS, convert_to_numpy=True)

def retrieve_top_k(query, k=3):
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)[0]
    sims = np.dot(DOC_EMBS, q_emb) / (np.linalg.norm(DOC_EMBS, axis=1) * np.linalg.norm(q_emb) + 1e-10)
    idx = np.argsort(-sims)[:k]
    return [{"id": IDS[i], "text": TEXTS[i], "score": float(sims[i]), "url": DOCS[i].get("url","local")} for i in idx]
