# Quick Reference - File and Directory Guide

## 📂 Directory Structure at a Glance

```
medical_rag_poc/
│
├── 📄 STRUCTURE.md ................. Detailed architecture (400+ lines)
├── 📄 README.md .................... Project overview & features
├── 📄 GETTING_STARTED.md ........... Quick setup guide
├── 📄 config.py .................... Configuration (86 settings)
├── 📄 main.py ...................... CLI interface
├── 📄 app.py ....................... Web UI launcher
├── 📄 requirements.txt ............. Python dependencies
├── 📄 .env.example ................. Environment template
├── 📄 .gitignore ................... Git ignore rules
│
├── 📁 data/
│   ├── raw_documents/
│   │   ├── 📁 migraine/
│   │   │   ├── overview.txt ......... Placeholder
│   │   │   ├── symptoms.txt ........ Placeholder
│   │   │   ├── treatment.txt ....... Placeholder
│   │   │   └── when_to_see_doctor.txt ... Placeholder
│   │   │
│   │   ├── 📁 diabetes/
│   │   │   ├── overview.txt ........ Placeholder
│   │   │   ├── symptoms.txt ........ Placeholder
│   │   │   ├── management.txt ...... Placeholder
│   │   │   └── complications.txt ... Placeholder
│   │   │
│   │   └── 📁 cancer/
│   │       ├── overview.txt ........ Placeholder
│   │       ├── staging_classification.txt ... Placeholder
│   │       ├── symptoms.txt ........ Placeholder
│   │       └── treatment_options.txt ... Placeholder
│   │
│   └── 📁 faiss_index/ (generated at runtime)
│       ├── index.faiss ............ Vector database
│       ├── metadata.json .......... Document metadata
│       └── embeddings.npy ......... Embedding vectors
│
├── 📁 src/
│   ├── 📁 agent/ (Core RAG components)
│   │   ├── rag_agent.py ........... Main RAG agent class
│   │   ├── embedding_model.py ..... Embedding interface
│   │   ├── llm_interface.py ....... LLM interface
│   │   └── __init__.py
│   │
│   ├── 📁 ui/ (User interfaces)
│   │   ├── gradio_app.py .......... Gradio web interface
│   │   └── __init__.py
│   │
│   ├── 📁 utils/ (Utilities)
│   │   ├── logger.py ............. Logging system
│   │   ├── data_loader.py ........ Document loading
│   │   ├── prompt_templates.py ... LLM prompts
│   │   ├── index_builder.py ...... FAISS index builder
│   │   └── __init__.py
│   │
│   └── __init__.py
│
└── 📁 logs/ (generated at runtime)
    ├── agent_2024-05-13_14-23-45.log ... Execution logs
    ├── agent_2024-05-13_15-10-22.log ... Execution logs
    └── index_build_2024-05-13_09-00-01.log ... Index build logs
```

---

## 🔧 What Each File Does

### Core Agent Files

| File | Purpose |
|------|---------|
| `src/agent/rag_agent.py` | Main RAG agent: retrieval + reasoning + generation |
| `src/agent/embedding_model.py` | Embed text using `bge-small-en-v1.5` model |
| `src/agent/llm_interface.py` | Call Llama LLM via Ollama API |

### Utilities

| File | Purpose |
|------|---------|
| `src/utils/logger.py` | Centralized logging to console + files |
| `src/utils/data_loader.py` | Load documents and create chunks |
| `src/utils/prompt_templates.py` | LLM prompt templates (system, reasoning, generation) |
| `src/utils/index_builder.py` | Build FAISS index from documents |

### UI & Entry Points

| File | Purpose |
|------|---------|
| `src/ui/gradio_app.py` | Gradio web interface with foldable sections |
| `main.py` | CLI: single query or interactive chat |
| `app.py` | Launcher for web UI |

### Configuration & Setup

| File | Purpose |
|------|---------|
| `config.py` | All 86 configuration settings |
| `requirements.txt` | Python package dependencies |
| `STRUCTURE.md` | 400+ line detailed architecture |
| `README.md` | Project overview, examples, features |
| `GETTING_STARTED.md` | Step-by-step setup instructions |

---

## 📝 Data Placeholders

Each disease folder has placeholder .txt files ready for content:

### Migraine Placeholders
- `overview.txt` - Definition and types
- `symptoms.txt` - Symptoms and triggers
- `treatment.txt` - Treatment options
- `when_to_see_doctor.txt` - Emergency indicators

### Diabetes Placeholders
- `overview.txt` - Types and definitions
- `symptoms.txt` - Warning signs
- `management.txt` - Daily management
- `complications.txt` - Long-term effects

### Cancer Placeholders
- `overview.txt` - General cancer info
- `staging_classification.txt` - TNM staging
- `symptoms.txt` - Cancer warning signs
- `treatment_options.txt` - Available treatments

---

## 🚀 Common Tasks

### Add a New Document
1. Create `.txt` file in `data/raw_documents/[disease]/`
2. Fill with medical content
3. Run: `python src/utils/index_builder.py --rebuild`

### Query via CLI
```bash
python main.py --query "What causes migraines?"
```

### Interactive Chat
```bash
python main.py --interactive
```

### Launch Web UI
```bash
python app.py
```
Then visit: http://localhost:7860

### View Logs
```bash
cat logs/agent_*.log
```

### Check Index
```bash
python src/utils/index_builder.py --validate
```

---

## ⚙️ Configuration Quick Reference

Key settings in `config.py`:

```python
# Models
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval
TOP_K = 5  # Documents to retrieve
CHUNK_SIZE = 300  # Words per chunk

# LLM
MAX_TOKENS = 512  # Max response length
TEMPERATURE = 0.7  # Creativity level

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# UI
GRADIO_SERVER_PORT = 7860
```

---

## 🔄 Data Flow

```
1. Raw Document (user adds .txt)
            ↓
2. Data Loader (loads & chunks)
            ↓
3. Embedding Model (bge-small)
            ↓
4. FAISS Index (vector DB)
            ↓
5. User Query
            ↓
6. Embed Query → FAISS Search
            ↓
7. Retrieve Top-K Documents
            ↓
8. LLM Reasoning
            ↓
9. LLM Generation (with citations)
            ↓
10. Response (to CLI or Web UI)
            ↓
11. Logging (to file and console)
```

---

## ✅ Checklist Before Using

- [ ] Ollama installed and running
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Medical documents added to `data/raw_documents/`
- [ ] Index built: `python src/utils/index_builder.py`
- [ ] Ready to use!

---

## 📊 File Statistics

| Category | Count | Lines |
|----------|-------|-------|
| Python Code | 7 | ~1200 |
| Utilities | 4 | ~600 |
| UI/CLI | 3 | ~400 |
| Documentation | 4 | ~2000 |
| Data Placeholders | 12 | ~240 |
| Config | 1 | ~150 |
| **TOTAL** | **31** | **~4600** |

---

**For detailed information, see:**
- Architecture: [STRUCTURE.md](STRUCTURE.md)
- Usage: [README.md](README.md)
- Setup: [GETTING_STARTED.md](GETTING_STARTED.md)
