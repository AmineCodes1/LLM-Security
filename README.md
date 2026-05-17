# Project Aegis-RAG

## Goal
Demonstrate indirect prompt injection in a RAG pipeline, then compare vulnerable vs. defended behavior using a local stack (Ollama + ChromaDB + Streamlit).

## Architecture (High-Level)
1. Ingest documents (PDF/TXT), chunk, embed, store in ChromaDB.
2. Retrieve top-k chunks for a user question.
3. Optional Shield: scan/flag/sanitize retrieved context.
4. Generate an answer grounded in retrieved context.
5. Optional output validation blocks unsafe responses.
6. UI surfaces retrieved context and threat signals.

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Ensure Ollama is running locally and pull a model:
   `ollama pull mistral:7b`
4. (Optional) Copy `.env.example` to `.env` and adjust values.

## Run the App
`streamlit run streamlit_app.py`

In the UI:
- Upload PDF/TXT files and click "Add to knowledge base" to ingest.
- Use the chat box for queries.
- Toggle the Shield and sanitization policy in the sidebar.

## Example Attack Scenario
1. Generate a clean + poisoned document pair:
   `python scripts/generate_attack_documents.py --output-dir data/raw --overwrite`
2. Ingest documents:
   `python scripts/ingest_documents.py --input-dir data/raw`
3. Ask a neutral question in the UI.
   - With Shield OFF, the poisoned chunk may steer the model to request sensitive data.
   - With Shield ON, the context is flagged/sanitized and unsafe output is blocked.

## Evaluation (Optional)
Run a small evaluation loop to measure attack success rate, false positives, and latency impact:
`python scripts/evaluate_system.py --iterations 20 --poisoned-ratio 0.5 --dry-run`
