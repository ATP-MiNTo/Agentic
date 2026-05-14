# Getting Started

## Prerequisites

- Python 3.10+
- Ollama installed and running
- Local model available in Ollama (default config uses `llama-3.1-8b-instruct`)

## Setup

1. Open project root:
   - `e:/Work/455Agentic`
2. Create virtual environment:
   - `python -m venv .venv`
3. Activate it:
   - Windows: `.venv\\Scripts\\activate`
4. Install dependencies:
   - `pip install -r requirements.txt`

## Prepare Data Index

1. Ensure source docs exist in `data/raw_documents/`.
2. Build FAISS index:
   - `python src/utils/index_builder.py`

If you update raw documents later, rebuild the index again.

## Run

- Web UI:
  - `python app.py`
- CLI one-shot:
  - `python main.py --query "What are warning signs of cancer?"`
- CLI chat:
  - `python main.py --interactive`

## Verify Agentic Behavior

In query details (UI accordion or CLI details), confirm you can see:

- step-wise action trace
- multiple tool actions when needed
- critic score and iteration count

## Agentic Config

Tune in `config.py` or `.env`:

- `AGENT_MAX_STEPS` (default: 4) - max iterations before forced finalization
- `AGENT_MIN_EVIDENCE_DOCS` (default: 2) - minimum documents before answer
- `AGENT_CONFIDENCE_THRESHOLD` (default: 0.7) - critic score to stop early
- `AGENT_FORCE_CLARIFY_ON_LOW_EVIDENCE` (default: true) - ask for clarification if score < threshold
- `AGENT_MAX_SUBQUERIES` (default: 3) - max subqueries when decomposing
- `ENABLE_SESSION_MEMORY` (default: true) - remember past queries in conversation

## Troubleshooting

- "Cannot connect to Ollama": start service with `ollama serve`.
- "FAISS index not found": run `python src/utils/index_builder.py`.
- Very low confidence responses: increase `TOP_K` or improve source documents.
