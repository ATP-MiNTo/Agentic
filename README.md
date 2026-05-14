# Medical RAG PoC: AI Agent with Retrieval-Augmented Generation

**A Proof of Concept demonstrating Agentic RAG for medical information retrieval and reasoning.**

---

## 🎯 What is This?

This is a **working demonstration** of an AI agent that uses **Retrieval-Augmented Generation (RAG)** to answer medical questions accurately. The agent:

1. **Retrieves** relevant medical documents based on semantic similarity
2. **Reasons** over the retrieved information to understand context
3. **Generates** evidence-based responses with source citations
4. **Logs** every step for transparency and debugging

Think of it as a **smart medical Q&A system** that knows where its answers come from and shows its work.

---

## 🏗️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | llama3.1:8b (open-source) | Runs locally, no API costs, customizable |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Lightweight, accurate semantic search |
| **Vector DB** | FAISS | Fast semantic search, no server needed |
| **Interface** | Gradio (web) + CLI | Interactive, zero-config, collapsible sections |
| **Framework** | Python 3.10+ | Well-established, extensive ML libraries |

---

## 🎓 How RAG Works (High Level)

```
User Query: "What causes migraines?"
    ↓
[1. RETRIEVAL] Search vector database for relevant docs
    Results: "Migraine Overview" (score: 0.92), "Migraine Triggers" (score: 0.87)
    ↓
[2. REASONING] Agent analyzes retrieved docs
    Thought: "Top docs discuss causes and triggers. These are relevant."
    ↓
[3. GENERATION] LLM generates response using docs + context
    Response: "Migraines are caused by neural and chemical changes...
              [Source: Migraine Overview]"
    ↓
User sees: Answer + Retrieved Documents + Agent Reasoning + Logs
```

**Key Difference from Simple Search**: RAG doesn't just list documents—it **synthesizes** information into coherent answers backed by evidence.

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                       │
│  ┌──────────────────┐          ┌──────────────────┐    │
│  │  Web UI (Gradio) │          │  CLI (Terminal)  │    │
│  └────────┬─────────┘          └────────┬─────────┘    │
└───────────┼────────────────────────────┼────────────────┘
            │                            │
            └────────────────┬───────────┘
                             │
    ┌────────────────────────▼────────────────────────┐
    │         Medical RAG Agent                       │
    │  ┌──────────────┐  ┌──────────────────────┐    │
    │  │  Retrieval   │  │   LLM Reasoning &    │    │
    │  │  (Semantic   │→ │   Response Gen       │    │
    │  │   Search)    │  │  (Llama-3.1-8B)     │    │
    │  └──────────────┘  └──────────────────────┘    │
    └────────────────────────┬───────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
    ┌────────────┐    ┌────────────────┐    ┌─────────┐
    │   FAISS    │    │  Embedding     │    │  Logs   │
    │   Index    │    │  Model         │    │  Files  │
    │ (Vector DB)│    │(bge-small)     │    │         │
    └────────────┘    └────────────────┘    └─────────┘
        │                    │
        └────────────────────┼────────────────┐
                             │                │
                    ┌────────▼────────┐      │
                    │ Raw Documents   │◄─────┘
                    │ (migraine,      │
                    │  diabetes,      │
                    │  cancer)        │
                    └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB RAM (minimum)
- CPU: 4+ cores (GPU optional)
- ~2GB disk space for models

### Installation

1. **Clone/download the project**:
   ```bash
   cd medical_rag_poc
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Build the FAISS index** (first time only):
   ```bash
   python src/utils/index_builder.py
   ```
   (Takes ~30-60 seconds, downloads models on first run)

### Running the Agent

**Option 1: Web Interface (Recommended for Demo)**
```bash
python app.py
# Opens http://localhost:7860 in browser
```

**Option 2: Command Line**
```bash
python main.py --query "What are symptoms of diabetes?"
python main.py --interactive        # Chat mode
```

---

## 📋 Medical Topics Covered

The PoC includes detailed medical documents on **3 diseases**:

### 1. **Migraine**
- Symptoms and triggers
- Treatment options
- Prevention strategies
- When to seek emergency care

### 2. **Diabetes**
- Type 1 vs Type 2 differences
- Management and monitoring
- Complications
- Lifestyle modifications

### 3. **Cancer**
- TNM staging system
- Common warning signs
- Treatment modalities
- Support resources

Each topic includes 3-4 detailed documents that the agent retrieves and reasons over.

---

## 📝 Example Interactions

### Example 1: Symptom Question
```
Query: "I have a throbbing headache with sensitivity to light. What could it be?"

[RETRIEVAL]
  - migraine/symptoms.txt (score: 0.91)
  - migraine/overview.txt (score: 0.87)

[REASONING]
  "Query mentions headache + light sensitivity, which are classic migraine 
   symptoms. Retrieved docs match perfectly."

[RESPONSE]
  "Based on the symptoms you describe—throbbing headache with light 
  sensitivity—this is consistent with migraine. Migraines often involve 
  throbbing pain and sensitivity to sensory stimuli. 
  [Source: migraine/symptoms.txt]
  
  ⚠️ IMPORTANT: This is not medical advice. Please consult a doctor 
  for proper diagnosis."
```

### Example 2: Management Question
```
Query: "How should I manage my diabetes on a daily basis?"

[RETRIEVAL]
  - diabetes/management.txt (score: 0.94)
  - diabetes/overview.txt (score: 0.81)

[REASONING]
  "User asking about daily diabetes management. Retrieved management doc 
   directly addresses this."

[RESPONSE]
  "Daily diabetes management typically includes:
  1. Blood sugar monitoring (2-4 times daily)
  2. Medication adherence (insulin or oral meds)
  3. Balanced diet (controlled carbs, fibers)
  4. Regular physical activity
  5. Stress management
  
  [Source: diabetes/management.txt]"
```

---

## 🔍 Observability: Seeing What the Agent Does

This is a **key feature** of the PoC. Every query shows:

### 1. **Retrieved Documents**
```
[RETRIEVAL] Top 5 documents retrieved:
  1. migraine/symptoms.txt          Score: 0.92  Distance: 0.08
  2. migraine/overview.txt          Score: 0.85  Distance: 0.15
  3. migraine/when_to_see_doctor    Score: 0.79  Distance: 0.21
```

### 2. **Agent Reasoning**
```
[AGENT REASONING]
"I retrieved 3 relevant documents about migraines. The query asks about 
causes, and the top doc (score 0.92) discusses triggers and underlying 
mechanisms. The agent will synthesize this into a coherent response."
```

### 3. **Execution Logs**
```
[2024-05-13 14:23:45] Query received: "What causes migraines?"
[2024-05-13 14:23:46] Embedding query vector (384 dimensions)
[2024-05-13 14:23:46] FAISS search completed (0.2ms)
[2024-05-13 14:23:46] Retrieved 5 documents, top score: 0.92
[2024-05-13 14:23:47] LLM inference started
[2024-05-13 14:23:48] Response generated (185 tokens)
[2024-05-13 14:23:48] Total execution time: 1.2 seconds
```

**All logs saved to** `logs/agent_YYYY-MM-DD_HH-MM-SS.log` for later review.

---

## 📚 Logs & Debugging

### Viewing Logs

**In Terminal** (automatically printed):
```
[INFO] Query: "What are diabetes symptoms?"
[DEBUG] Embedding model loaded
[DEBUG] FAISS index loaded
[INFO] Retrieved 5 documents
[INFO] Reasoning step completed
[INFO] Response generated
```

**In Files** (stored for audit trail):
```bash
ls logs/
# Output:
# agent_2024-05-13_14-23-45.log
# agent_2024-05-13_15-10-22.log
# index_build_2024-05-13_09-00-01.log
```

### Debugging Mode

To see **more detailed logs**:
```python
# Edit config.py
LOG_LEVEL = "DEBUG"  # Instead of "INFO"
```

---

## 🧠 Model Specifications

### LLM: llama3.1:8b
- **Parameters**: 8 Billion (fits in CPU)
- **Architecture**: Transformer-based
- **Quantization**: Supported (reduces memory)
- **License**: Meta AI (open-source)
- **Inference Time**: ~2-5 seconds per query (CPU)

### Embedding Model: BAAI/bge-small-en-v1.5
- **Dimensions**: 384
- **Size**: ~133 MB
- **Specialized**: Semantic similarity
- **Speed**: ~10ms per query embedding

### FAISS: Facebook AI Similarity Search
- **Type**: Approximate Nearest Neighbor Search
- **Complexity**: O(log n) instead of O(n)
- **Index Size**: ~500KB for 12 documents
- **Search Latency**: ~1-2ms for top-5

---

## 🔧 Configuration & Customization

All settings in `config.py`:

```python
# LLM Parameters
LLM_MODEL = "llama3.1:8b"
MAX_TOKENS = 512
TEMPERATURE = 0.7  # 0=deterministic, 1=creative

# Retrieval
TOP_K = 5                    # Documents to retrieve
CHUNK_SIZE = 300             # Words per chunk
CHUNK_OVERLAP = 50           # Overlap between chunks

# Logging
LOG_LEVEL = "INFO"           # DEBUG, INFO, WARNING, ERROR
SAVE_LOGS_TO_FILE = True
```

---

## 📊 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Index Build Time** | 30-60s | One-time cost |
| **Query Embedding** | 10ms | CPU |
| **FAISS Search** | 1-2ms | Fast |
| **LLM Inference** | 2-5s | CPU, depends on response length |
| **Total Query Time** | 3-8s | End-to-end |
| **Memory Usage** | ~2GB | Llama + embedding model |
| **Disk Usage** | ~500MB | Models + index + data |

---

## 🤝 Data Management

### Adding New Medical Documents

**Step 1**: Create a `.txt` file in the appropriate folder:
```
data/raw_documents/migraine/new_document.txt
```

**Step 2**: Follow the template structure:
```
[DOCUMENT TITLE]
Source: [Where you got this info]
Last Updated: [Date]

---

[Your medical content here...]
```

**Step 3**: Rebuild the index:
```bash
python src/utils/index_builder.py --rebuild
```

The agent will immediately use the new document.

---

## ⚠️ Important Disclaimers

This is a **Proof of Concept**. For production medical applications:

1. **Not for Medical Decisions**: Use only for educational purposes
2. **Consult Healthcare Professionals**: Always see a doctor for actual medical concerns
3. **Data Accuracy**: Verify all information from official health sources
4. **Legal Compliance**: HIPAA, GDPR, and medical regulations apply in production
5. **Liability**: Developers are not liable for medical misuse

---

## 🧪 Testing the Agent

### Test Query 1: Direct Symptom Match
```bash
python main.py --query "I have severe headaches and light sensitivity, what is it?"
```
**Expected**: High retrieval scores, clear migraine identification

### Test Query 2: Specific Management
```bash
python main.py --query "How often should I check my blood sugar?"
```
**Expected**: Retrieves diabetes/management.txt, provides actionable answer

### Test Query 3: Complex Comparison
```bash
python main.py --query "What's the difference between Type 1 and Type 2 diabetes?"
```
**Expected**: Multiple docs retrieved, nuanced comparison

---

## 📁 Project Structure Summary

For detailed breakdown, see [STRUCTURE.md](STRUCTURE.md).

```
medical_rag_poc/
├── data/raw_documents/      # Medical documents (add yours here!)
│   ├── migraine/
│   ├── diabetes/
│   └── cancer/
├── src/
│   ├── agent/               # RAG logic
│   ├── ui/                  # Gradio interface
│   └── utils/               # Helpers & logging
├── logs/                    # Execution logs
├── config.py                # Settings
├── main.py                  # CLI entry point
├── app.py                   # Web UI entry point
└── requirements.txt         # Dependencies
```

---

## 🚦 Next Steps

1. **Install & run** (see Quick Start above)
2. **Add medical documents** to `data/raw_documents/`
3. **Test queries** via CLI or web UI
4. **Review logs** to understand agent behavior
5. **Customize** via `config.py`
6. **Deploy** or extend (see STRUCTURE.md for details)

---

## 📖 Documentation

- **[STRUCTURE.md](STRUCTURE.md)** — Detailed breakdown of every file and how they work
- **[README.md](README.md)** — This file (project overview)
- **[config.py](config.py)** — Configuration reference
- **[logs/](logs/)** — Execution logs for debugging

---

## ❓ FAQ

**Q: Why is the first query slow?**  
A: Models are being downloaded and initialized. Subsequent queries are faster.

**Q: Can I use a different LLM?**  
A: Yes! Edit `config.py` and `src/agent/llm_interface.py`.

**Q: How do I improve answer quality?**  
A: Add more detailed medical documents, refine prompts in `src/utils/prompt_templates.py`.

**Q: Can this replace a real doctor?**  
A: Absolutely not. This is a learning tool, not medical advice.

**Q: How do I add custom medical knowledge?**  
A: Add `.txt` files to `data/raw_documents/`, then rebuild the index.

---

## 📄 License & Attribution

- **Llama**: Meta AI (open-source)
- **FAISS**: Facebook AI (open-source)
- **Medical Data**: Public health sources (WHO, CDC, etc.)

---

## 🎓 Learning Outcomes

By studying this PoC, you'll understand:

✅ How RAG combines retrieval + generation  
✅ How semantic search works with embeddings  
✅ How LLMs reason over retrieved context  
✅ How to build production-ready logging  
✅ How to structure maintainable ML projects  
✅ How to make AI systems transparent & debuggable  

---

**Ready to explore? Run:**
```bash
python app.py
```

Enjoy! 🚀
