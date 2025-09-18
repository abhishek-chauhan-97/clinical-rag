# retriever.py (lightweight TF-IDF version)
import json
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---- Load Docs ----
def load_docs(path="data/docs.jsonl"):
    docs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        logger.info(f"📂 Loaded {len(docs)} docs from {path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load {path}, using fallback sample. Error: {e}")
        docs = [{
            "id": "sample_1",
            "text": "Paracetamol is used for fever and pain. Standard adult dosing guidance is 500-1000 mg every 4-6 hours.",
            "url": "local"
        }]
    return docs

DOCS = load_docs()
IDS = [d["id"] for d in DOCS]
TEXTS = [d["text"] for d in DOCS]

# ---- TF-IDF Embeddings ----
logger.info("🔧 Building TF-IDF vectorizer")
if TEXTS:
    VECTORIZER = TfidfVectorizer(stop_words="english")
    DOC_EMBS = VECTORIZER.fit_transform(TEXTS)
    logger.info(f"✅ Precomputed embeddings for {len(TEXTS)} documents")
else:
    VECTORIZER, DOC_EMBS = None, None
    logger.warning("⚠️ No documents to index")

# ---- Retrieval ----
def retrieve_top_k(query, k=3):
    logger.info(f"🔍 Retrieving top {k} docs for query='{query}'")

    if not DOCS or DOC_EMBS is None:
        logger.warning("⚠️ No documents available for retrieval")
        return []

    q_vec = VECTORIZER.transform([query])
    sims = cosine_similarity(q_vec, DOC_EMBS)[0]

    # rank by similarity
    idx = sims.argsort()[::-1][:k]

    results = [
        {"id": IDS[i], "text": TEXTS[i], "score": float(sims[i]), "url": DOCS[i].get("url", "local")}
        for i in idx
    ]

    logger.debug(f"Retrieved docs: {[r['id'] for r in results]}")
    return results
