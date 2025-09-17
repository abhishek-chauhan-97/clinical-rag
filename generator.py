import requests
from config import HF_TOKEN
from retriever import retrieve_top_k
import logging

logger = logging.getLogger(__name__)

# ✅ Step 1: Define models that are actually supported by HF Inference API (free tier)
AVAILABLE_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl"
]

def hf_generate(prompt, model="google/flan-t5-base", max_tokens=256):
    """
    Calls Hugging Face inference API.
    Returns dict: { "text": "..."} or { "error": "..." }
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            logger.error(f"HF API error {r.status_code}: {r.text}")
            return {"error": f"HF API error {r.status_code}: {r.text}"}
        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return {"text": data[0]["generated_text"]}
        return {"text": str(data)}
    except Exception as e:
        logger.exception("HF API call failed")
        return {"error": str(e)}

def fallback_answer(query, retrieved):
    """
    Generates a dummy offline fallback answer using retrieved docs.
    """
    if not retrieved:
        return f"⚠️ Offline fallback: No docs retrieved for '{query}'."

    context_texts = [r['text'] for r in retrieved]
    answer = "⚠️ Offline fallback: Based on retrieved docs:\n\n"
    for i, t in enumerate(context_texts, start=1):
        answer += f"- {t}\n"
    return answer.strip()

def generate_answer(query, top_k, model_choice):
    logs = []
    logs.append(f"User query: {query}")

    # Step 1: retrieval
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} documents")

    # Step 2: build prompt
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"Use ONLY the context below to answer. Cite sources in [ID].\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    logs.append("Built prompt for model.")

    # Step 3: model call
    gen = hf_generate(prompt, model_choice)
    if "error" in gen:
        logs.append(f"Model error: {gen['error']}")
        # ✅ Fallback mode
        fallback = fallback_answer(query, retrieved)
        logs.append("⚠️ Returned offline fallback answer instead.")
        return fallback, logs, retrieved

    logs.append("Model responded successfully.")
    return gen["text"].strip(), logs, retrieved
