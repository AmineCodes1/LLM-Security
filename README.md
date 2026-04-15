# LLM-Security

Project Aegis-RAG is a local Retrieval-Augmented Generation scaffold focused on security experimentation.

## Why this structure

- `src/aegis_rag`: Core business logic (config, ingestion, retrieval/generation, guardrails). Keeping logic in `src` makes imports explicit and test-friendly.
- `scripts`: Operational commands, separated from core logic to keep the package reusable.
- `streamlit_app.py`: Thin UI layer so frontend changes do not affect backend services.
- `data/raw`: Input documents for ingestion.
- `data/chroma`: Persistent local vector store.

## Project layout

```
LLM-Security/
	data/
		chroma/
		raw/
	scripts/
		ingest_documents.py
	src/
		aegis_rag/
			__init__.py
			config.py
			guardrails.py
			ingestion.py
			logging_config.py
			rag_pipeline.py
	.env.example
	requirements.txt
	streamlit_app.py
```

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
	 `pip install -r requirements.txt`
3. Ensure Ollama is running locally and pull a model:
	 `ollama pull mistral:7b`
4. Copy `.env.example` to `.env` and adjust values if needed.
5. Add PDF/TXT files to `data/raw`.
6. Ingest documents:
	 `python scripts/ingest_documents.py --input-dir data/raw`
7. Launch UI:
	 `streamlit run streamlit_app.py`

## Notes for local 16GB RAM

- Default model is `mistral:7b` for lower memory pressure.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` for compact footprint.
- Retrieval defaults to top-k=3 to keep context and latency controlled.