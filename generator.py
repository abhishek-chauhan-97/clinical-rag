# generator.py
import logging
import google.generativeai as genai
from retriever import retrieve_top_k, retrieve_pubmed
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

# -------- Main Orchestration
def generate_answer(query: str, top_k: int, model_choice: str = "gemini-1.5-flash"):
    """
    Retrieves references (local or PubMed), sends prompt to Gemini,
    falls back to PubMed search if local docs are weak.
    Returns (answer, logs, references).
    """
    logs = []
    logs.append(f"User query: {query}")

    # 1. Try retrieving local docs
    retrieved = retrieve_top_k(query, k=top_k)
    logs.append(f"Retrieved {len(retrieved)} local documents")

    # 2. If local docs are fallback-only → switch to PubMed
    if not retrieved or (len(retrieved) == 1 and retrieved[0]["id"].startswith("sample_")):
        logs.append("⚠️ Local docs not useful → switching to PubMed API")
        retrieved = retrieve_pubmed(query, k=top_k)
        logs.append(f"Retrieved {len(retrieved)} PubMed documents")

    # 3. Build prompt
    context = "\n\n---\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"""
You are an advanced clinical AI assistant.
The following references may or may not be sufficient. 
If they are useful, incorporate them. If not, use your own knowledge.

Write your answer as if advising a seasoned professional doctor.
Be precise, evidence-based, and avoid generic explanations.

CONTEXT (references for your use):
{context}

QUESTION: {query}

Provide the answer below in clear medical language:
"""
    logs.append("Built professional doctor-facing prompt.")

    # 4. Gemini call
    gen = gemini_generate(prompt, model_choice)
    if "error" in gen:
        logs.append(f"Model error: {gen['error']}")
        return "⚠️ Gemini API call failed. Try again later.", logs, retrieved

    logs.append("Model responded successfully (Gemini).")
    return gen["text"].strip(), logs, retrieved
