# Medical RAG PoC - Project Structure Documentation

## Overview

This document provides a **detailed breakdown** of the entire project structure, explaining the purpose, role, and relationships of each component. This is essential for understanding how the system works, maintaining the codebase, and extending functionality.

---

## Directory Tree

```
medical_rag_poc/
├── data/                           # All data-related files
│   ├── raw_documents/              # Original medical documents (placeholder)
│   │   ├── migraine/               # Migraine disease data
│   │   ├── diabetes/               # Diabetes disease data
│   │   └── cancer/                 # Cancer disease data
│   └── faiss_index/                # Vector database files (generated at runtime)
├── src/                            # Core source code
│   ├── agent/                      # RAG agent implementation
│   ├── ui/                         # Gradio web interface
│   └── utils/                      # Utility modules
├── logs/                           # Execution logs (generated at runtime)
├── requirements.txt                # Python dependencies
├── config.py                       # Configuration settings
├── main.py                         # Entry point for CLI
├── README.md                       # Project overview (user-facing)
└── STRUCTURE.md                    # This file (developer-facing)
```

---

## 1. Data Layer: `data/`

### 1.1 `data/raw_documents/`

**Purpose**: Stores all source medical documents that the RAG system will index and retrieve from.

**Structure**:
```
raw_documents/
├── migraine/
│   ├── overview.txt                # Basic info about migraine
│   ├── symptoms.txt                # Symptoms and triggers
│   ├── treatment.txt               # Treatment options
│   └── when_to_see_doctor.txt      # Emergency indicators
├── diabetes/
│   ├── overview.txt                # Definition and types
│   ├── symptoms.txt                # Warning signs
│   ├── management.txt              # Daily management
│   └── complications.txt           # Long-term effects
└── cancer/
    ├── overview.txt                # General cancer info
    ├── staging_classification.txt  # TNM staging system
    ├── symptoms.txt                # Cancer warning signs
    └── treatment_options.txt       # Available treatments
```

**Key Characteristics**:
- **Format**: Plain `.txt` files (human-readable, easy to maintain)
- **Naming**: Lowercase, underscore-separated, descriptive
- **Content Structure**: Each file has [PLACEHOLDER] section with clear content guidelines
- **Flexibility**: Users can easily add/edit documents without touching code

**How it's Used**:
1. Documents are loaded by `data_loader.py`
2. Split into chunks (~300 words each)
3. Embedded using `bge-small-en-v1.5`
4. Stored in FAISS index

**Maintainability Notes**:
- ✅ Easy to add new documents—just create `.txt` file in appropriate disease folder
- ✅ Easy to update content—modify text without code changes
- ✅ Easy to organize—disease folders keep related docs together
- ⚠️ Keep each file focused on ONE topic (e.g., "symptoms" not "symptoms-and-treatments")

---

### 1.2 `data/faiss_index/`

**Purpose**: Stores the vector embeddings and FAISS index for fast semantic search. **Generated at runtime** (not part of source control).

**Files Generated**:
```
faiss_index/
├── index.faiss                     # Binary FAISS index
├── metadata.json                   # Document IDs and source info
└── embeddings.npy                  # Numpy array of embeddings
```

**Technical Details**:
- **FAISS Index**: Fast Approximate Nearest Neighbor Search library
- **Embedding Dimension**: 384 (from `bge-small-en-v1.5`)
- **Purpose**: Enable O(log n) semantic search instead of O(n)
- **Creation**: Run `index_builder.py` to rebuild index

**How It's Used in RAG**:
1. User query is embedded using same model
2. Cosine similarity search against FAISS index
3. Top-K (default 5) documents retrieved with similarity scores
4. Metadata provides source information and document text

**Maintainability Notes**:
- ✅ Index is automatically rebuilt if documents change
- ✅ Reproducible—same documents → same index
- ⚠️ Rebuilding takes ~30-60 seconds for 12 documents

---

## 2. Source Code Layer: `src/`

### 2.1 `src/agent/`

**Purpose**: Core RAG agent implementation. Contains all logic for retrieval, reasoning, and response generation.

**Key Files**:

#### `src/agent/rag_agent.py` (Main Agent Class)

```python
class MedicalRAGAgent:
    - __init__()           # Initialize with LLM, embedding model, FAISS index
    - retrieve()           # Semantic search: query → ranked docs
    - reason()             # LLM reasoning: docs → intermediate thoughts
    - generate_response()  # Final response generation with citations
    - chat()               # Main entry point: query → full response
    - log_execution()      # Structured logging of all steps
```

**Key Methods Explained**:

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `retrieve(query, top_k=5)` | User query (string) | List of (doc_id, text, score) | Semantic search step |
| `reason(query, docs)` | Query + retrieved docs | Agent reasoning (string) | Show thinking process |
| `generate_response(query, docs, reasoning)` | Query + docs + thoughts | Final response with citations | User-facing answer |
| `log_execution()` | All execution data | Write to logs/ | Audit trail & debugging |

**Example Data Flow**:
```
User Query: "What causes migraines?"
    ↓ retrieve()
[("migraine/overview.txt", "content...", 0.92),
 ("migraine/symptoms.txt", "content...", 0.87),
 ...]
    ↓ reason()
"Retrieved 3 documents about migraine causes..."
    ↓ generate_response()
"Migraines are caused by... [Source: migraine/overview.txt]"
    ↓ log_execution()
Writes to logs/agent_YYYY-MM-DD_HH-MM-SS.log
```

**Maintainability Notes**:
- ✅ Single responsibility: handles only RAG logic
- ✅ Logging is built-in, not optional
- ✅ Easy to swap LLM (just change `self.llm` initialization)
- ⚠️ Reasoning quality depends on document quality

---

#### `src/agent/embedding_model.py`

**Purpose**: Manages embedding model (`bge-small-en-v1.5`). Handles model loading, caching, and inference.

**Key Methods**:
- `load_model()` — Load from HuggingFace (cached locally)
- `embed_text(text)` — Convert text to 384-dim vector
- `embed_batch(texts)` — Efficient batch embedding

**Why Separate File?**:
- ✅ Reusable across agent and indexing
- ✅ Easy to swap embedding model
- ✅ Easier to test independently

---

#### `src/agent/llm_interface.py`

**Purpose**: Abstraction layer for LLM (Llama-3.1-8B). Handles model loading, prompt engineering, and inference.

**Methods**:
- `load_model()` — Load Llama via Ollama or local
- `generate(prompt, max_tokens=512)` — LLM inference
- `stream_generate(prompt)` — Streaming responses

**Why Separate?**:
- ✅ Easy to swap Llama for other open-source models
- ✅ Encapsulates LLM-specific logic
- ✅ Can add caching, rate limiting here

---

### 2.2 `src/ui/`

**Purpose**: Gradio-based web interface for interactive agent testing.

#### `src/ui/gradio_app.py`

**Features**:
```python
gr.Interface:
  Inputs:
    - query_textbox        # User medical question
    - top_k_slider         # Number of documents to retrieve
  
  Outputs (Foldable Sections):
    - final_response       # Main answer
    - [FOLD] retrieval_details    # Retrieved docs + scores
    - [FOLD] agent_reasoning      # LLM thinking process
    - [FOLD] execution_logs       # Full execution trace
```

**Foldable Sections Implementation**:
- Uses HTML/CSS within Gradio for collapsible sections
- Initially hidden to reduce visual clutter
- Click to expand and see details (Agent reasoning, retrieval scores, etc.)

**Example UI Layout**:
```
┌─────────────────────────────────────────┐
│  Medical RAG Assistant                  │
├─────────────────────────────────────────┤
│ Query: [________________________________________]
│ Top-K: [5]  [Submit]
├─────────────────────────────────────────┤
│ RESPONSE:
│ Migraines are caused by neural and 
│ chemical changes in the brain...
├─────────────────────────────────────────┤
│ ▸ Retrieved Documents & Scores
│   [Click to expand]
├─────────────────────────────────────────┤
│ ▸ Agent Reasoning
│   [Click to expand]
├─────────────────────────────────────────┤
│ ▸ Execution Logs
│   [Click to expand]
└─────────────────────────────────────────┘
```

**Maintainability Notes**:
- ✅ Gradio handles rendering; no HTML/CSS needed
- ✅ Easy to add/remove output sections
- ✅ No external dependencies (Gradio is lightweight)

---

### 2.3 `src/utils/`

**Purpose**: Utility modules for data handling, logging, and common operations.

#### `src/utils/data_loader.py`

**Methods**:
- `load_documents(directory)` — Load all `.txt` files recursively
- `chunk_documents(docs, chunk_size=300, overlap=50)` — Split into overlapping chunks
- `validate_documents()` — Check for empty files, encoding issues

**Why Separate?**:
- ✅ Data loading logic is independent of agent
- ✅ Easy to add new formats (e.g., PDF, JSON)
- ✅ Reusable for index building and validation

---

#### `src/utils/logger.py`

**Purpose**: Centralized logging system. Logs both to terminal and files.

**Features**:
```python
Logger:
  - log_query(query)           # When user submits query
  - log_retrieval(docs, scores) # Retrieved documents + scores
  - log_reasoning(thoughts)    # Agent reasoning
  - log_response(response)     # Final response
  - save_to_file()            # Persist to logs/
```

**Log Format Example**:
```
[2024-05-13 14:23:45] QUERY: "What are migraine triggers?"
[2024-05-13 14:23:46] RETRIEVAL:
  - migraine/symptoms.txt (score: 0.89)
  - migraine/overview.txt (score: 0.85)
[2024-05-13 14:23:47] REASONING: "User asked about triggers..."
[2024-05-13 14:23:48] RESPONSE: "Common triggers include..."
```

**Maintainability Notes**:
- ✅ Centralized logging makes debugging easy
- ✅ Easy to add new log levels
- ✅ File logs persist for audit trail

---

#### `src/utils/index_builder.py`

**Purpose**: Build and rebuild FAISS index from raw documents.

**Methods**:
- `build_index()` — Full pipeline: load → chunk → embed → FAISS
- `save_index()` — Serialize index to disk
- `load_index()` — Deserialize from disk

**When Run**:
- During initial setup: `python index_builder.py`
- After adding/editing documents: `python index_builder.py --rebuild`

**Execution Time**: ~30-60 seconds (one-time cost)

---

#### `src/utils/prompt_templates.py`

**Purpose**: Store and manage LLM prompts. Keeps prompts versioned and maintainable.

**Prompts Stored**:
```python
SYSTEM_PROMPT = """You are a medical information assistant...
Provide accurate, evidence-based information..."""

RETRIEVAL_PROMPT = """Based on these documents, answer the question..."""

REASONING_PROMPT = """Show your thinking process step by step..."""
```

**Why Separate?**:
- ✅ Easy to refine prompts without touching agent code
- ✅ Easy to A/B test different prompts
- ✅ Prompts are versioned with code

---

## 3. Configuration: `config.py`

**Purpose**: Centralized configuration for easy customization without code changes.

**Key Settings**:
```python
# Model Configuration
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# RAG Configuration
TOP_K = 5                    # Documents to retrieve
CHUNK_SIZE = 300             # Words per chunk
CHUNK_OVERLAP = 50           # Overlap between chunks

# LLM Parameters
MAX_TOKENS = 512
TEMPERATURE = 0.7            # Creativity (0=deterministic, 1=creative)

# Paths
DATA_DIR = "data/raw_documents"
INDEX_DIR = "data/faiss_index"
LOG_DIR = "logs"

# Logging
LOG_LEVEL = "INFO"           # DEBUG, INFO, WARNING, ERROR
SAVE_LOGS_TO_FILE = True
```

**Maintainability Notes**:
- ✅ Change settings without editing code
- ✅ Easy to test different configurations
- ✅ Single source of truth for all constants

---

## 4. Entry Points

### `main.py` — CLI Interface

**Purpose**: Command-line interface for running the agent.

**Usage**:
```bash
python main.py --query "What causes diabetes?"
python main.py --interactive          # Chat mode
python main.py --rebuild-index        # Rebuild FAISS
```

**Code Structure**:
```python
def main():
    # Load config
    # Initialize agent
    # Process CLI arguments
    # Run query
    # Display results
```

---

### `app.py` — Web Interface

**Purpose**: Launch Gradio web UI.

**Usage**:
```bash
python app.py
# Opens http://localhost:7860
```

---

## 5. Logs: `logs/`

**Purpose**: Stores execution logs for audit trail and debugging. **Generated at runtime**.

**Log File Naming**:
```
logs/
├── agent_2024-05-13_14-23-45.log    # One file per execution
├── agent_2024-05-13_15-10-22.log
├── index_build_2024-05-13_09-00-01.log
└── errors_2024-05-13.log            # Errors only
```

**Log Contents**:
```
[timestamp] [level] [component] message

Example:
[2024-05-13 14:23:45] [INFO] [RAGAgent] Query: "What causes migraines?"
[2024-05-13 14:23:46] [DEBUG] [EmbeddingModel] Embedding query...
[2024-05-13 14:23:46] [INFO] [Retrieval] Retrieved 5 docs (top score: 0.92)
[2024-05-13 14:23:47] [INFO] [LLM] Generating response...
[2024-05-13 14:23:48] [INFO] [Response] Response generated (185 tokens)
```

**Maintainability Notes**:
- ✅ Logs are auto-rotated (old logs archived)
- ✅ Use logs to diagnose issues
- ✅ Share logs for debugging without exposing code

---

## 6. Dependencies: `requirements.txt`

**Purpose**: Lists all Python dependencies and versions.

**Key Dependencies**:
```
torch==2.0.1              # Deep learning framework
transformers==4.30.0      # Hugging Face models
sentence-transformers==2.2.2  # Embedding models
faiss-cpu==1.7.3          # Vector search
gradio==3.50.0            # Web UI
ollama==0.1.0             # LLM interface
python-dotenv==1.0.0      # Environment variables
```

**Why Each Dependency?**:
| Dependency | Purpose | Alternative |
|------------|---------|-------------|
| torch | DL framework (required by transformers) | TensorFlow (not used) |
| transformers | Load HF models | Calling API (rejected) |
| sentence-transformers | Embedding model wrapper | Manual implementation (slow) |
| faiss-cpu | Vector search | ScaNN, Annoy (slower) |
| gradio | Web UI | Streamlit, Flask (more complex) |
| ollama | Local LLM interface | LM Studio, vLLM (more setup) |

---

## 7. Typical Workflow & File Interactions

### Scenario 1: User Adds Medical Document

```
User edits data/raw_documents/migraine/side_effects.txt
    ↓
Run: python index_builder.py --rebuild
    ↓
- data_loader.py loads NEW document
- Chunks the content
- embedding_model.py embeds chunks
- FAISS index saved to data/faiss_index/
    ↓
Next query uses updated index
```

### Scenario 2: User Asks Query via CLI

```
User: python main.py --query "How is diabetes treated?"
    ↓
main.py initializes MedicalRAGAgent
    ↓
rag_agent.retrieve()
  - embedding_model.py embeds query
  - FAISS finds top-5 similar docs
    ↓
rag_agent.reason()
  - Formats retrieval prompt
  - llm_interface.py calls Llama
    ↓
rag_agent.generate_response()
  - LLM generates answer with citations
    ↓
logger.py writes execution trace to logs/
    ↓
main.py displays response + logs to terminal
```

### Scenario 3: User Accesses Web UI

```
User: python app.py
    ↓
Gradio starts at http://localhost:7860
    ↓
User enters query in UI
    ↓
gradio_app.py calls agent.chat(query)
    ↓
Same flow as Scenario 2
    ↓
Results displayed in collapsible sections
```

---

## 8. Maintainability Principles

### 8.1 Readability

- **File Organization**: By function (agent/, ui/, utils/) not by type
- **Naming**: Descriptive, lowercase, underscores (Python convention)
- **Comments**: Docstrings on every module and function
- **Type Hints**: Python 3.8+ type annotations where possible

### 8.2 Extensibility

- **Configuration**: Central `config.py` for all settings
- **Abstractions**: Each component is isolated (swap embedding model = 1 change)
- **Logging**: Built-in; easy to add new logs without touching core logic
- **Modularity**: Utils are reusable; agent is LLM-agnostic

### 8.3 Testability

- **Unit Tests** (optional but recommended):
  - Test `data_loader.py` with sample documents
  - Test `embedding_model.py` with known vectors
  - Test prompt templates independently
  
- **Integration Tests**:
  - Full query → response pipeline
  - Index rebuild and validation

### 8.4 Debugging

- **Logs**: Check `logs/` directory for step-by-step execution
- **Config**: Change `LOG_LEVEL = "DEBUG"` for verbose output
- **Isolated Components**: Can test each part independently

---

## 9. Key Decisions & Rationale

| Decision | Why | Alternatives |
|----------|-----|-------------|
| Plain text `.txt` documents | Human-readable, version-controllable | PDF, JSON (less editable) |
| FAISS for vector search | Fast, lightweight, no server needed | Pinecone, Weaviate (cloud) |
| Gradio for UI | Zero-config, auto-responsive | Flask, Streamlit (more setup) |
| Chunking documents | Improves retrieval precision | Full document search (less precise) |
| Separate utils modules | Reusability and testability | Monolithic agent (harder to maintain) |
| Centralized logging | Audit trail, debugging | Print statements (lost on crash) |

---

## 10. Getting Started for New Developers

**To understand the codebase**:
1. Read this file first (you're doing it!)
2. Look at `config.py` (understand all settings)
3. Review `src/agent/rag_agent.py` (core logic)
4. Check `src/ui/gradio_app.py` (user interface)
5. Read `logs/` from a test run (see real execution flow)

**To add a new feature**:
1. Decide which layer: agent, ui, or utils
2. Check if it's already in a utility module
3. Create/edit the appropriate file
4. Update logging if relevant
5. Test and document

**To debug an issue**:
1. Check recent logs in `logs/`
2. Set `LOG_LEVEL = "DEBUG"` in `config.py`
3. Run again and examine output
4. Trace through the relevant source file

---

## Summary

This project is designed for **clarity and maintainability**:
- ✅ Clear directory structure (data/ src/ logs/)
- ✅ Single-responsibility modules
- ✅ Comprehensive logging at every step
- ✅ Configuration-driven customization
- ✅ Easy to extend (swap models, add docs, change prompts)

The organization prioritizes making it easy for developers to:
1. Understand how queries flow through the system
2. Add new documents without touching code
3. Modify behavior through configuration
4. Debug issues by reading logs
5. Extend functionality by adding new modules
