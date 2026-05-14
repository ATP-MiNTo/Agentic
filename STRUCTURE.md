# Project Structure

## Overview

This project is a medical Retrieval-Augmented Generation assistant with an iterative agentic control loop. The loop is:

1. plan next action
2. execute tool
3. observe results
4. critique evidence quality
5. stop or continue

## Workspace Layout

```
455Agentic/
|-- app.py
|-- main.py
|-- config.py
|-- README.md
|-- GETTING_STARTED.md
|-- QUICK_REFERENCE.md
|-- STRUCTURE.md
|-- requirements.txt
|-- data/
|   |-- raw_documents/
|   `-- faiss_index/
|-- logs/
`-- src/
    |-- agent/
    |   |-- embedding_model.py
    |   |-- llm_interface.py
    |   `-- rag_agent.py
    |-- ui/
    |   `-- gradio_app.py
    `-- utils/
        |-- data_loader.py
        |-- index_builder.py
        |-- logger.py
        `-- prompt_templates.py
```

## Core Modules

### `src/agent/rag_agent.py`

Main orchestrator. Responsibilities:

- maintain iterative step loop
- request planner JSON action
- execute selected action/tool
- run critic JSON evaluation
- decide stop/continue from confidence and evidence thresholds
- synthesize final response with citations and disclaimer

Supported actions (LLM planner chooses one per step):

1. `retrieve_top_k` - Initial semantic search with configured top-k
2. `reretrieve_refined_query` - Rewrite query for better retrieval
3. `decompose_query` - Break complex question into focused subqueries
4. `retrieve_multi_query` - Execute parallel retrievals on subqueries
5. `extract_evidence_by_topic` - Filter evidence by topic/keyword
6. `clarify_user` - Ask user to clarify an ambiguous question
7. `finalize_answer` - Generate final response from evidence

### `src/agent/llm_interface.py`

LLM gateway for Ollama.

- `generate(...)` for regular text
- `generate_json(...)` for structured planner/critic outputs
- robust first-JSON-object extraction from model text

### `src/utils/prompt_templates.py`

Central prompt definitions:

- `PLANNER_PROMPT`
- `QUERY_REWRITE_PROMPT`
- `CRITIC_PROMPT`
- `REASONING_PROMPT`
- `GENERATION_PROMPT`
- `SYSTEM_PROMPT` and disclaimer

### `src/ui/gradio_app.py`

Web interface with sections for:

- final answer
- retrieved documents
- reasoning summary
- execution details (iterations + critic score)
- action trace

## Data Layer

### `data/raw_documents/`

Topic-based source documents (migraine, diabetes, cancer). These are authoritative context inputs for retrieval.

### `data/faiss_index/`

Generated vector artifacts used at runtime for semantic search.

## Configuration (`config.py`)

Important agentic controls:

- `AGENT_MAX_STEPS`
- `AGENT_MIN_EVIDENCE_DOCS`
- `AGENT_CONFIDENCE_THRESHOLD`
- `AGENT_FORCE_CLARIFY_ON_LOW_EVIDENCE`

These settings define loop depth, evidence requirements, and low-confidence behavior.

## Runtime Flow

1. user submits query
2. planner selects action in JSON
3. action runs and updates working evidence state
4. critic scores evidence sufficiency
5. loop exits on action finalization or confidence threshold
6. final response generation with citations and safety disclaimer

## Logging and Observability

`src/utils/logger.py` and runtime logs in `logs/` provide step-level traceability:

- query received
- retrieval results and scores
- planned actions
- critic score and confidence
- final response timing

## Updating Knowledge

1. edit or add files under `data/raw_documents/`
2. rebuild index via `python src/utils/index_builder.py`
3. run app again
