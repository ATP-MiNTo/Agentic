# Quick Reference

## Run Commands

- Build/rebuild vector index:
  - `python src/utils/index_builder.py`
- Web UI:
  - `python app.py`
- CLI single query:
  - `python main.py --query "What are migraine symptoms?"`
- CLI interactive chat:
  - `python main.py --interactive`
- CLI with custom top-k:
  - `python main.py --query "..." --top-k 10`
- CLI verbose mode:
  - `python main.py --query "..." --verbose`

## Agent Loop Actions (Planner Selects)

1. `retrieve_top_k` - Initial semantic search retrieval
2. `reretrieve_refined_query` - Query rewriting + re-retrieve
3. `decompose_query` - Break complex question into subqueries
4. `retrieve_multi_query` - Execute multiple focused retrievals
5. `extract_evidence_by_topic` - Filter/focus retrieved evidence by topic
6. `clarify_user` - Ask user for clarification on vague queries
7. `finalize_answer` - Generate response from accumulated evidence

## High-Value Config Keys (`config.py`)

**Retrieval:**
- `TOP_K` (default: 5) - docs per retrieval
- `MIN_RETRIEVAL_SCORE` (default: 0.5) - similarity threshold
- `CHUNK_SIZE`, `CHUNK_OVERLAP` - document chunking

**LLM:**
- `LLM_MODEL` - Ollama model name
- `MAX_TOKENS`, `MIN_TOKENS` - response length
- `TEMPERATURE` - 0.0=deterministic, 1.0=creative
- `LLM_TIMEOUT` - seconds before timeout

**Agentic Loop:**
- `AGENT_MAX_STEPS` (default: 4) - max iterations
- `AGENT_CONFIDENCE_THRESHOLD` (default: 0.7) - stop when score ≥ this
- `AGENT_MIN_EVIDENCE_DOCS` (default: 2) - minimum docs needed
- `AGENT_FORCE_CLARIFY_ON_LOW_EVIDENCE` - ask for clarification if low score
- `AGENT_MAX_SUBQUERIES` (default: 3) - max decomposed queries
- `ENABLE_SESSION_MEMORY` - track conversation history

**Logging:**
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR, CRITICAL
- `SAVE_LOGS_TO_FILE` - persist logs
- `DEBUG_MODE` - keep intermediate data

## Key Files

- `src/agent/rag_agent.py` - planner/tool/critic orchestration
- `src/agent/llm_interface.py` - Ollama interface + JSON extraction
- `src/agent/embedding_model.py` - embedding generation (HuggingFace)
- `src/utils/prompt_templates.py` - all prompts (planner, critic, generation, reasoning)
- `src/utils/session_memory.py` - multi-turn conversation memory
- `src/ui/gradio_app.py` - response, retrieval, reasoning, execution details
- `src/utils/index_builder.py` - FAISS indexing pipeline
- `src/utils/data_loader.py` - document loading
- `src/utils/logger.py` - structured logging

## Data Paths

- Raw documents: `data/raw_documents/` (organized by medical topic)
- FAISS index: `data/faiss_index/`
- Logs: `logs/`
- Cache: `.cache/` (embeddings)

## Medical Topics in Database

- **Migraine**: symptoms, triggers, treatments, prevention
- **Diabetes**: types, symptoms, management, complications
- **Cancer**: overview, staging, symptoms, treatments

## Typical Debug Checks

1. ✓ Ollama reachable: `curl http://localhost:11434/api/tags`
2. ✓ Model available: check in Ollama UI or logs
3. ✓ FAISS files exist: `ls data/faiss_index/`
4. ✓ Query shows `retrieved_documents` in details
5. ✓ Agent trace has multiple actions (sign of iteration)
6. ✓ Critic score trends upward (0.0→1.0) across steps
7. ✓ Session memory enabled: check `logs/` for turn history
