# Getting Started - Medical RAG PoC

## Quick Setup Guide

### Prerequisites
- Python 3.10+
- 8GB RAM minimum
- Ollama (for running Llama model)
- ~2GB disk space

### Step 1: Install Ollama

Download and install from: https://ollama.ai

After installation, pull the Llama model:
```bash
ollama pull llama2:7b-instruct
# or
ollama pull mistral:7b-instruct
```

Start Ollama (it runs as a service):
```bash
ollama serve
```

The service will be available at `http://localhost:11434`

### Step 2: Set Up Python Environment

#### Option A: Using venv (Recommended)
```bash
cd medical_rag_poc

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using conda
```bash
conda create -n medical-rag python=3.10
conda activate medical-rag
cd medical_rag_poc
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (ML framework)
- Transformers (HuggingFace models)
- Sentence Transformers (embeddings)
- FAISS (vector search)
- Gradio (web UI)
- And other utilities

**First time**: This may take 3-5 minutes depending on internet speed.

### Step 4: Add Medical Documents

Add your medical documents to:
```
data/raw_documents/
├── migraine/
│   ├── overview.txt
│   ├── symptoms.txt
│   ├── treatment.txt
│   └── when_to_see_doctor.txt
├── diabetes/
│   └── [your content here]
└── cancer/
    └── [your content here]
```

Each `.txt` file should contain medical information in plain text format.

### Step 5: Build FAISS Index

```bash
python src/utils/index_builder.py
```

This will:
1. Load all documents from `data/raw_documents/`
2. Split them into chunks
3. Embed chunks using `bge-small-en-v1.5`
4. Create FAISS index
5. Save to `data/faiss_index/`

**Takes 30-60 seconds on first run** (downloads models)

### Step 6: Run the Agent

#### Option A: Web Interface (Recommended for Demo)
```bash
python app.py
```

Then open: http://localhost:7860

#### Option B: Command Line
```bash
# Single query
python main.py --query "What are diabetes symptoms?"

# Interactive mode
python main.py --interactive
```

## Troubleshooting

### ❌ "Cannot connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check endpoint: `http://localhost:11434/api/tags`

### ❌ "FAISS index not found"
- Run: `python src/utils/index_builder.py`

### ❌ "No documents found"
- Check `data/raw_documents/` directory
- Add `.txt` files to appropriate disease folders

### ❌ "Out of memory"
- Reduce `BATCH_SIZE` in `config.py`
- Use smaller model (e.g., `mistral:7b` instead of larger models)

### ❌ Slow embedding
- First run downloads models (~1GB)
- Subsequent runs are faster
- For faster embedding, use GPU (see advanced setup)

## Configuration

Edit `config.py` to customize:

```python
# Model
LLM_MODEL = "llama-3.1-8b-instruct"  # or "mistral:7b-instruct"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval
TOP_K = 5  # Number of documents to retrieve
CHUNK_SIZE = 300  # Words per chunk

# LLM
MAX_TOKENS = 512  # Max response length
TEMPERATURE = 0.7  # 0=deterministic, 1=creative

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

## Project Structure

```
medical_rag_poc/
├── data/                    # Medical documents and FAISS index
├── src/
│   ├── agent/              # RAG agent implementation
│   ├── ui/                 # Gradio web interface
│   └── utils/              # Utilities (logging, data loading)
├── logs/                   # Execution logs
├── config.py               # Settings
├── main.py                 # CLI entry point
├── app.py                  # Web UI entry point
├── README.md               # Full documentation
└── STRUCTURE.md            # Detailed architecture
```

## Common Commands

```bash
# Build index
python src/utils/index_builder.py --rebuild

# Single query via CLI
python main.py --query "What causes migraines?"

# Interactive chat
python main.py --interactive

# Web UI
python app.py

# Validate index
python src/utils/index_builder.py --validate

# View logs
cat logs/agent_*.log
```

## Next Steps

1. ✅ Install Ollama and pull a model
2. ✅ Run `pip install -r requirements.txt`
3. ✅ Add medical documents to `data/raw_documents/`
4. ✅ Run `python src/utils/index_builder.py`
5. ✅ Launch web UI: `python app.py`
6. ✅ Ask medical questions!

## Performance Tips

- **Faster**: Use smaller model (8B instead of 13B+)
- **Faster**: Use fewer documents (start with 5-10)
- **Faster**: Reduce `TOP_K` from 5 to 3
- **Better**: Add more detailed documents
- **Better**: Use GPU (install `torch-cuda` and `faiss-gpu`)

## For More Details

- [README.md](README.md) - Project overview and features
- [STRUCTURE.md](STRUCTURE.md) - Detailed architecture documentation
- [config.py](config.py) - All configurable settings

---

**Ready? Start with:**
```bash
python app.py
```

Happy exploring! 🚀
