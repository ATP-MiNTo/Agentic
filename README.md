# Medical Agentic RAG PoC

A local-first medical QA assistant that now runs an iterative agentic loop:

1. Planner chooses the next action.
2. Tool executes retrieval or evidence focusing.
3. Critic scores evidence quality and decides whether to continue.
4. Final answer is generated with citations and safety disclaimer.

## Agentic Architecture

Unlike basic RAG, this system runs an **iterative planner-tool-critic loop**:

- **Planner**: LLM chooses next action from 7 tools (retrieve, rewrite, decompose, etc.)
- **Tools**: Execute retrieval or evidence refinement, updating state
- **Critic**: Evaluates evidence quality and confidence after each step
- **Stopping Logic**: Exits when confidence ≥ threshold OR max steps reached

**Key Capabilities:**
- Dynamic tool selection (not fixed pipeline)
- Query decomposition for complex questions
- Multi-query retrieval for broader coverage
- Confidence-based early stopping
- Clarifying-question fallback for ambiguous queries
- Structured JSON planning and critique outputs
- Persistent session memory across turns
- Agent trace and critic scores exposed in UI

## Key Features

- Local LLM via Ollama (`llama-3.1-8b-instruct` by default)
- Semantic retrieval via FAISS + BGE embeddings
- Iterative plan-act-observe-critic orchestration
- Medical safety disclaimers and source-cited responses
- Detailed logs for each query and step

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Ensure Ollama is running and model is available.
4. Build the index:
   - `python src/utils/index_builder.py`
5. Run app:
   - Web UI: `python app.py`
   - CLI single query: `python main.py --query "What are early symptoms of diabetes?"`
   - CLI interactive: `python main.py --interactive`

## Agentic Runtime Parameters (`config.py`)

- `ENABLE_AGENTIC_LOOP = True`
- `AGENT_MAX_STEPS = 4`
- `AGENT_MIN_EVIDENCE_DOCS = 2`
- `AGENT_CONFIDENCE_THRESHOLD = 0.7`
- `AGENT_FORCE_CLARIFY_ON_LOW_EVIDENCE = True`
- `ENABLE_SESSION_MEMORY = True`
- `AGENT_MAX_SUBQUERIES = 3`

## Important Files

- `src/agent/rag_agent.py`: Agent loop orchestration
- `src/agent/llm_interface.py`: LLM calls + JSON output parsing
- `src/utils/prompt_templates.py`: Planner, rewrite, critic, and generation prompts
- `src/ui/gradio_app.py`: UI formatting of action trace and confidence score
- `config.py`: Runtime knobs and validation

## Safety Notes

This system is educational and not a diagnostic or treatment tool.
Always consult a qualified healthcare professional for medical decisions.

## Documentation

- `GETTING_STARTED.md`: installation and first run
- `QUICK_REFERENCE.md`: commands and key files
- `STRUCTURE.md`: architecture and module map
