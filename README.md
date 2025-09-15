# clinical-rag — prototype RAG chatbot (Gradio)

Flow:
- load small docs corpus (data/docs.jsonl)
- compute sentence embeddings (all-MiniLM-L6-v2)
- simple cosine-based retriever (top-K)
- call Hugging Face Inference API for generation (HF token required, set in Space secrets as HF_TOKEN)

Not for clinical use. Prototype only.
