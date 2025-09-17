import requests
from config import HF_TOKEN
from retriever import retrieve_top_k
import logging
logger = logging.getLogger(__name__)

def hf_generate(prompt, model="google/flan-t5-small", max_tokens=256):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return {"error": f"HF API error {r.status_code}: {r.text}"}
        data = r.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return {"text": data[0]["generated_text"]}
        return {"text": str(data)}
    except Exception as e:
        return {"error": str(e)}

def generate_answer(query, top_k, model_choice):
    logs=[]
    logs.append(f"User query: {query}")

    # retrieval
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} documents")

    # context
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"Use ONLY the context below to answer. Cite sources in [ID].\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    logs.append("Built prompt for model.")

    gen = hf_generate(prompt, model_choice)
    if "error" in gen:
        logs.append(f"Model error: {gen['error']}")
        return None, logs, retrieved
    logs.append("Model responded successfully.")
    return gen["text"].strip(), logs, retrieved



def generate_answer(query, top_k, model_choice):
    logs = []
    refs = []

    try:
        logger.info(f"🔍 Generating answer | Model={model_choice} | Query='{query}'")
        # actual model call here
        response = llm(query)
        logger.debug(f"Raw response: {response}")
        logs.append("Model call succeeded.")
        return response, logs, refs
    except Exception as e:
        logger.exception("❌ Model call failed")
        logs.append(f"Exception: {str(e)}")
        return None, logs, refs


