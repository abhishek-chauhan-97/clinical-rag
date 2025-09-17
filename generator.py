# generator.py (replace existing)
import os, requests, logging
from retriever import retrieve_top_k
logger = logging.getLogger(__name__)

# Models we will surface in UI (HF inference may still fail for some)
AVAILABLE_MODELS = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl"
]

# -------- HF Inference call
HF_TOKEN = os.environ.get("HF_TOKEN")

def hf_generate(prompt, model="google/flan-t5-base", max_tokens=256):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    try:
        logger.info(f"Calling HF Inference: {model}")
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            logger.error(f"HF API error {r.status_code}: {r.text}")
            return {"error": f"HF API error {r.status_code}: {r.text}"}
        data = r.json()
        logger.debug(f"HF raw: {data}")
        if isinstance(data, list) and "generated_text" in data[0]:
            return {"text": data[0]["generated_text"]}
        # some models return {'generated_text': ...} or other shapes
        return {"text": str(data)}
    except Exception as e:
        logger.exception("HF API call failed")
        return {"error": str(e)}

# -------- Diagnostic: check model endpoint
def check_model_availability(model):
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://api-inference.huggingface.co/models/{model}"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        logger.info(f"Model check {model} -> {r.status_code}")
        return r.status_code, r.text[:2000]
    except Exception as e:
        logger.exception("Model availability check failed")
        return None, str(e)

# -------- Local transformers fallback (lazy load)
_local_tokenizer = None
_local_model = None
def load_local_model(model_name="google/flan-t5-small"):
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except Exception as e:
        logger.exception("transformers/torch import failed")
        raise

    logger.info(f"Loading local model {model_name} into CPU (this may take time)")
    _local_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _local_model.to("cpu")
    return _local_model, _local_tokenizer

def local_generate(prompt, model_name="google/flan-t5-small", max_new_tokens=128):
    try:
        model, tokenizer = load_local_model(model_name)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except Exception as e:
        logger.exception("Local generation failed")
        return None

# -------- Offline fallback (texts from retrieved docs)
def fallback_answer(query, retrieved):
    if not retrieved:
        return f"⚠️ Offline fallback: No docs retrieved for '{query}'."
    # conservative: return source text paragraphs verbatim
    context_texts = [r['text'] for r in retrieved]
    answer = "⚠️ Offline fallback: Based on retrieved docs:\n\n"
    for i, t in enumerate(context_texts, start=1):
        answer += f"- {t}\n"
    return answer.strip()

# -------- Orchestration
def generate_answer(query, top_k, model_choice):
    logs = []
    logs.append(f"User query: {query}")

    # retrieval
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} documents")

    # build prompt
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"Use ONLY the context below to answer. Cite sources in [ID].\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    logs.append("Built prompt for model.")

    # 1) Try HF Inference
    gen = hf_generate(prompt, model_choice)
    if "text" in gen:
        logs.append("Model responded successfully (HF Inference).")
        return gen["text"].strip(), logs, retrieved

    # 2) If HF fails -> try local transformers generation (best-effort)
    logs.append(f"Model error: {gen.get('error')}. Trying local model fallback.")
    local_text = local_generate(prompt, model_name="google/flan-t5-small", max_new_tokens=128)
    if local_text:
        logs.append("Local model generation succeeded.")
        return local_text.strip(), logs, retrieved

    # 3) Final: offline fallback
    logs.append("Local model failed or not available. Returning offline fallback.")
    fallback = fallback_answer(query, retrieved)
    return fallback, logs, retrieved
