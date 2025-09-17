---
title: Clinical RAG — Clinical GAI Prototype
emoji: "🩺"
colorFrom: "blue"
colorTo: "green"
sdk: streamlit
sdk_version: "1.49.1"
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

# Files & structure


## How to update
- Edit files via GitHub (recommended) OR in this Space's **Files** UI.
- After commit to this repo, restart or factory rebuild the Space to apply changes.

## Troubleshooting
- If the app fails on startup with `PermissionError: '/.streamlit'`, ensure `.streamlit/config.toml` exists (see repo) and that `app.py` creates `.streamlit` directory before Streamlit imports runtime settings.
- If Hugging Face Inference returns 404/403, check the `model` name in `app.py` and the `HF_TOKEN` secret.

## License / Notes
Prototype for demonstration and interview purposes. Not intended for production or clinical decision-making.

