# Medical RAG PoC

A compact retrieval-augmented generation proof-of-concept for medical content. Uses local Ollama LLMs, FAISS for retrieval, and a small embedding model.

See `STRUCTURE.md` for architecture and `QUICK_REFERENCE.md` for day-to-day commands.

## Requirements

- Python 3.10+ (3.11 recommended on Windows)
- Ollama running locally (default: `http://localhost:11434`)
- A pulled Ollama model (e.g., `llama3.1`)

## Quick setup

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Start Ollama if needed:

```bash
ollama pull llama3.1
ollama serve
```

## Next steps

- Run the web UI: `python app.py`
- Rebuild the index: `python -m src.utils.index_builder --rebuild`
- Run a CLI query: `python main.py --query "What are diabetes symptoms?"`

For full commands and troubleshooting, open `QUICK_REFERENCE.md`.
