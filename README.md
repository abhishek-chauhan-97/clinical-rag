---
title: Clinical RAG — Clinical GAI Prototype
emoji: "🩺"
colorFrom: "blue"
colorTo: "green"
sdk: streamlit
sdk_version: "latest"
app_file: app.py
pinned: false
---

# Clinical RAG — Clinical GAI Prototype

**Short description**  
Prototype RAG chatbot for clinical literature (demo only). NOT for clinical use.

## Quickstart (what this Space does)
- Runs a Streamlit front-end (`app.py`)
- Uses `sentence-transformers` to compute embeddings for local docs (`data/docs.jsonl`)
- Calls Hugging Face Inference API (requires `HF_TOKEN` secret) to generate answers
- Shows provenance and logs for RAG pipeline

## Required secrets / environment
- `HF_TOKEN` — Hugging Face API token (write or inference scope as needed). Add under *Settings → Variables and secrets*.

## Files & structure
