# Getting Started - Medical RAG PoC

## Quick setup

### 1. Prerequisites

- Python 3.10+
- Python 3.11 is recommended on Windows
- Ollama installed locally
- About 2 GB of free disk space for models and index files

### 2. Install Ollama

Download and install Ollama from https://ollama.ai.

Pull the model used by default in this project:

```bash
ollama pull llama3.1
```

Start the Ollama service:

```bash
ollama serve
```

The API should be available at `http://localhost:11434`.

### 3. Create a Python environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If Gradio fails to install cleanly on Windows, use Python 3.11 and reinstall the requirements in a fresh virtual environment.

### 5. Add medical documents

Place plain-text or Markdown files under `data/raw_documents/`.

Examples:

- `data/raw_documents/migraine/overview.txt`
- `data/raw_documents/diabetes/management.txt`
- `data/raw_documents/cancer/treatment_options.txt`

The CLI indexes everything under `data/raw_documents/`.
The web UI lets you load a single corpus folder or all documents.

### 6. Build or validate the FAISS index

Build the default index:

```bash
python -m src.utils.index_builder
```

Force a rebuild:

```bash
python -m src.utils.index_builder --rebuild
```

Validate the saved index:

```bash
python -m src.utils.index_builder --validate
```

The CLI also supports rebuilding:

```bash
python main.py --rebuild-index
```

### 7. Run the app

Web UI:

```bash
python app.py
```

CLI query:

```bash
python main.py --query "What are diabetes symptoms?"
```

Interactive CLI:

```bash
python main.py --interactive
```

## Current behavior

- `main.py` runs against the full corpus in `data/raw_documents/`.
- `app.py` opens the Gradio UI defined in `src/ui/gradio_app.py`.
- The Gradio UI loads a corpus based on the dropdown and rebuilds the FAISS index for that corpus.
- The default disease corpus is `cancer`.

## Troubleshooting

### Cannot connect to Ollama

- Make sure `ollama serve` is running.
- Confirm `http://localhost:11434/api/tags` responds.

### FAISS index missing

- Run `python -m src.utils.index_builder --rebuild`.

### No documents found

- Check `data/raw_documents/`.
- Make sure the files end in `.txt` or `.md`.

### Gradio import errors on Windows

- Use Python 3.11.
- Recreate the virtual environment and reinstall dependencies.

## Next step

Run the web UI:

```bash
python app.py
```
