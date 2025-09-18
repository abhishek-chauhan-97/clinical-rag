# generator.py (Gemini-first)
import logging
from retriever import retrieve_top_k
from config import GEMINI_API_KEY, AVAILABLE_MODELS

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

logger = logging.getLogger(__name__)

# -------- Gemini Inference
def gemini_generate(prompt, model="gemini-1.5-flash"):
    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        if response and response.text:
            return {"text": response.text}
        return {"error": "No text returned from Gemini"}
    except Exception as e:
        logger.exception("Gemini API call failed")
        return {"error": str(e)}

# -------- Offline fallback
def fallback_answer(query, retrieved):
    if not retrieved:
        return f"⚠️ Offline fallback: No docs retrieved for '{query}'."
    context_texts = [r['text'] for r in retrieved]
    answer = "⚠️ Offline fallback: Based on retrieved docs:\n\n"
    for i, t in enumerate(context_texts, start=1):
        answer += f"- {t}\n"
    return answer.strip()

# -------- Orchestration
def generate_answer(query, top_k, model_choice="gemini-1.5-flash"):
    logs = []
    logs.append(f"User query: {query}")

    # retrieval
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} documents")

    # build prompt
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"Use ONLY the context below to answer. Cite sources in [ID].\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    logs.append("Built prompt for model.")

    # Gemini call
    gen = gemini_generate(prompt, model_choice)
    if "error" in gen:
        logs.append(f"Model error: {gen['error']}")
        fallback = fallback_answer(query, retrieved)
        logs.append("⚠️ Returned offline fallback answer instead.")
        return fallback, logs, retrieved

    logs.append("Model responded successfully (Gemini).")
    return gen["text"].strip(), logs, retrieved
