# retriever.py (ultra-light, no external deps)
import json
import logging
import math
from collections import Counter

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

# ---- Tiny tokenizer ----
def tokenize(text):
    return [t.lower() for t in text.split() if t.isalnum()]

# ---- Build term frequencies ----
DOC_TOKS = [tokenize(t) for t in TEXTS]
DOC_COUNTS = [Counter(toks) for toks in DOC_TOKS]
VOCAB = set(w for toks in DOC_TOKS for w in toks)

# ---- Cosine similarity (count-based) ----
def cosine_sim(counter1, counter2):
    common = set(counter1.keys()) & set(counter2.keys())
    num = sum(counter1[w] * counter2[w] for w in common)
    denom1 = math.sqrt(sum(v*v for v in counter1.values()))
    denom2 = math.sqrt(sum(v*v for v in counter2.values()))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return num / (denom1 * denom2)

# ---- Retrieval ----
def retrieve_top_k(query, k=3):
    logger.info(f"🔍 Retrieving top {k} docs for query='{query}'")
    q_toks = tokenize(query)
    q_count = Counter(q_toks)

    sims = [cosine_sim(q_count, d) for d in DOC_COUNTS]
    idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]

    results = [
        {"id": IDS[i], "text": TEXTS[i], "score": float(sims[i]), "url": DOCS[i].get("url", "local")}
        for i in idx
    ]
    logger.debug(f"Retrieved docs: {[r['id'] for r in results]}")
    return results
