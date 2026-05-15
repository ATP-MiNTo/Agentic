# Quick Reference - Commands

Copyable commands for common tasks.

### Build / validate FAISS index

```bash
python -m src.utils.index_builder        # build (if index missing)
python -m src.utils.index_builder --rebuild
python -m src.utils.index_builder --validate
```

### Run CLI

```bash
python main.py --query "What are migraine symptoms?"
python main.py --interactive
python main.py --rebuild-index
python main.py --top-k 3 --query "What causes migraines?"
```

### Run web UI

```bash
python app.py
```

### Config hints

- Edit `config.py` to change defaults like `TOP_K`, `DEFAULT_DISEASE`, and `GRADIO_SERVER_PORT`.

### Troubleshooting

- Ollama unreachable: ensure `ollama serve` is running and the model is pulled.
- FAISS index missing/errors: rebuild with `python -m src.utils.index_builder --rebuild`.
- Gradio import issues on Windows: try Python 3.11 and reinstall requirements in a fresh venv.

For more details about project structure, see `STRUCTURE.md`.
