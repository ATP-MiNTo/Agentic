# Medical RAG PoC - Project Structure

## Overview

This project is a local medical RAG demo built around three main flows:

- CLI queries through `main.py`
- Web UI queries through `app.py` and `src/ui/gradio_app.py`
- Index building and validation through `src/utils/index_builder.py`

The current implementation supports two corpus patterns:

- The CLI indexes everything under `data/raw_documents/`
- The Gradio UI can switch between disease corpora and rebuild the FAISS index for the selected corpus

## Directory layout

```text
medical_rag_poc/
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── GETTING_STARTED.md
├── QUICK_REFERENCE.md
├── STRUCTURE.md
├── data/
│   ├── raw_documents/
│   │   ├── migraine/
│   │   ├── diabetes/
│   │   └── cancer/
│   └── faiss_index/
├── logs/
└── src/
    ├── agent/
    ├── ui/
    └── utils/
```

## Data layer

### `data/raw_documents/`

This is the source corpus directory. The loader walks the tree recursively and accepts `.txt` and `.md` files.

Typical files include:

- `migraine/overview.txt`
- `migraine/symptoms.txt`
- `diabetes/management.txt`
- `cancer/treatment_options.txt`

### `data/faiss_index/`

This directory is generated at runtime and stores the saved FAISS artifacts.

Current files written by the index builder:

- `index.faiss`
- `metadata.json`

## Source code layer

### `src/agent/`

Core retrieval and generation logic.

- `embedding_model.py` loads and runs the embedding model.
- `llm_interface.py` talks to Ollama.
- `rag_agent.py` loads the FAISS index, retrieves chunks, builds prompts, and returns the final response.

### `src/ui/`

The Flask web app lives here.

- `flask_app.py` builds the web UI.
- The UI includes a corpus dropdown, top-k control, memory summary, retrieval details, reasoning output, and execution details.
- The UI keeps a short chat history summary so follow-up questions have context.

### `src/utils/`

Shared utilities.

- `data_loader.py` loads documents and chunks them.
- `index_builder.py` builds and validates the FAISS index.
- `logger.py` provides the logging helper.
- `prompt_templates.py` stores prompt text and formatting helpers.

## Configuration

`config.py` holds the runtime defaults.

Important settings:

- `LLM_MODEL = "llama3.1"`
- `LLM_ENDPOINT = "http://localhost:11434"`
- `EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"`
- `TOP_K = 5`
- `CHUNK_SIZE = 300`
- `CHUNK_OVERLAP = 50`
- `DEFAULT_DISEASE = "cancer"`
- `DISEASE_OPTIONS = ["cancer", "diabetes", "migraine", "all"]`
- `GRADIO_SERVER_NAME = "127.0.0.1"`
- `GRADIO_SERVER_PORT = 7860`

## Entry points

### `main.py`

CLI entry point.

### `app.py`

Starts the Gradio web UI.

## Logs

`logs/` stores runtime output from the logger.

Use it to inspect:

- query handling
- retrieval results
- reasoning output
- response generation

## Maintenance notes

- Add new medical content by placing another `.txt` or `.md` file in the right corpus folder.
- Rebuild the index after changing documents.
- Keep the CLI and web UI docs aligned with `config.py`, because the defaults define the current behavior.
