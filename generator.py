# generator.py (Gemini-only, lightweight fallback)
import logging
from retriever import retrieve_top_k
from config import GEMINI_API_KEY
import google.generativeai as genai
from config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
logger = logging.getLogger(__name__)

# -------- Gemini Inference
def gemini_generate(prompt: str, model: str = "gemini-1.5-flash") -> dict:
    """Calls Gemini API with the given prompt and returns text output."""
    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        if response and response.text:
            return {"text": response.text}
        return {"error": "No text returned from Gemini"}
    except Exception as e:
        logger.exception("Gemini API call failed")
        return {"error": str(e)}

# -------- Offline text-only fallback
def fallback_answer(query: str, retrieved: list) -> str:
    """Lightweight fallback: surfaces retrieved chunks if Gemini call fails."""
    if not retrieved:
        return f"⚠️ Offline fallback: No docs retrieved for '{query}'."
    context_texts = [r["text"] for r in retrieved]
    answer = "⚠️ Offline fallback: Based on retrieved docs:\n\n"
    for i, t in enumerate(context_texts, start=1):
        answer += f"- {t}\n"
    return answer.strip()

# -------- Orchestration
def generate_answer(query: str, top_k: int, model_choice: str = "gemini-1.5-flash"):
    """Main entry: retrieves top-k docs, builds prompt, queries Gemini,
    and falls back to retrieved text if needed.
    """
    logs = []
    logs.append(f"User query: {query}")

    # retrieval
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} documents")

    # build prompt
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = (
        f"Use ONLY the context below to answer. Cite sources in [ID].\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer:"
    )
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

import logging
logger = logging.getLogger(__name__)
logger.info(f"🔑 Gemini API key loaded (length={len(GEMINI_API_KEY)})")

