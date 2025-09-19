# retriever.py
import json
import logging
import math
import requests
from collections import Counter
from config import PUBMED_API_KEY

logger = logging.getLogger(__name__)

# ---- Load Local Docs ----
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

# ---- Cosine similarity ----
def cosine_sim(counter1, counter2):
    common = set(counter1.keys()) & set(counter2.keys())
    num = sum(counter1[w] * counter2[w] for w in common)
    denom1 = math.sqrt(sum(v*v for v in counter1.values()))
    denom2 = math.sqrt(sum(v*v for v in counter2.values()))
    if denom1 == 0 or denom2 == 0:
        return 0.0
    return num / (denom1 * denom2)

# ---- Local Retrieval ----
def retrieve_top_k(query, k=3):
    logger.info(f"🔍 Retrieving top {k} docs (local) for query='{query}'")
    q_count = Counter(tokenize(query))
    sims = [cosine_sim(q_count, d) for d in DOC_COUNTS]
    idx = sorted(range(len(sims)), key=lambda i: -sims[i])[:k]

    results = [
        {"id": IDS[i], "text": TEXTS[i], "score": float(sims[i]), "url": DOCS[i].get("url", "local")}
        for i in idx
    ]
    logger.debug(f"Retrieved docs: {[r['id'] for r in results]}")
    return results

# ---- PubMed Retrieval ----
def retrieve_pubmed(query, k=3):
    """Fetch top PubMed abstracts using Entrez API."""
    logger.info(f"🌐 Retrieving top {k} PubMed docs for query='{query}'")

    try:
        # Search PubMed
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": k,
            "retmode": "json",
        }
        if PUBMED_API_KEY:
            params["api_key"] = PUBMED_API_KEY

        search_res = requests.get(search_url, params=params, timeout=10).json()
        ids = search_res["esearchresult"]["idlist"]

        if not ids:
            return []

        # Fetch summaries
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "text",
        }
        fetch_res = requests.get(fetch_url, params=params, timeout=10).text

        # Split abstracts
        docs = []
        for i, pid in enumerate(ids):
            docs.append({
                "id": f"pubmed_{pid}",
                "text": f"PubMed ID {pid} abstract:\n{fetch_res}",
                "score": 1.0,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
            })
        return docs[:k]

    except Exception as e:
        logger.error(f"❌ PubMed retrieval failed: {e}")
        return []
