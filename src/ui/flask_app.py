# Ensure .env is loaded before anything else
import src.utils.load_env
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request

import config
from src.agent.rag_agent import MedicalRAGAgent
from src.utils.index_builder import FAISSIndexBuilder
from src.utils.logger import get_logger

logger = get_logger()

app_root = Path(__file__).resolve().parents[2]

current_disease = config.DEFAULT_DISEASE
agent: MedicalRAGAgent = None


def _build_memory_summary(history: List[Tuple[str, str]], max_turns: int = 4) -> str:
    if not history:
        return ""

    recent_turns = history[-max_turns:]
    lines: List[str] = []
    for index, (user_message, assistant_message) in enumerate(recent_turns, 1):
        lines.append(f"Turn {index} User: {user_message}")
        lines.append(f"Turn {index} Assistant: {assistant_message}")

    return "\n".join(lines)


def _build_memory_panel(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "Memory\n\nNo conversation memory yet. Ask a question to start a chat thread."

    return f"Memory\n\n{_build_memory_summary(history)}"


def _format_retrieval_details(result: Dict) -> List[Dict]:
    return [
        {
            "id": doc["id"],
            "score": doc["score"],
            "preview": doc["preview"],
        }
        for doc in result.get("retrieved_documents", [])
    ]


def _build_logs_text(history: List[Tuple[str, str]], top_k: int, execution_time: float = None) -> str:
    lines = [
        f"- Chat Turns in Memory: {len(history)}",
        f"- Memory Used: {'Yes' if len(history) > 0 else 'No'}",
        f"- Top-K: {top_k}",
    ]
    if execution_time is not None:
        lines.append(f"- Last query time: {execution_time:.2f}s")
    return "\n".join(lines)


def _get_data_dir_for_disease(disease: str) -> Path:
    disease = (disease or "").strip().lower()
    if disease not in config.DISEASE_OPTIONS:
        raise ValueError(f"Unsupported disease '{disease}'")
    return config.DATA_DIR if disease == "all" else config.DATA_DIR / disease


def _load_agent_for_disease(disease: str) -> Tuple[MedicalRAGAgent, str]:
    data_dir = _get_data_dir_for_disease(disease)
    builder = FAISSIndexBuilder(data_dir=data_dir)
    if not builder.build_index():
        raise RuntimeError(f"Failed to build index for {disease}")

    agent_instance = MedicalRAGAgent()
    return agent_instance, f"Loaded corpus: {disease}"


def create_app() -> Flask:
    global agent, current_disease

    static_folder = str(app_root / "static")
    template_folder = str(app_root / "templates")
    app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)

    try:
        agent, current_disease = _load_agent_for_disease(config.DEFAULT_DISEASE)
        logger.info(f"Initialized web agent with default corpus: {current_disease}")
    except Exception as e:
        logger.error(f"Failed to initialize default corpus: {str(e)}")
        agent = None

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            current_disease=current_disease,
            disease_options=config.DISEASE_OPTIONS,
            default_top_k=config.TOP_K,
        )

    @app.route("/disclaimer")
    def disclaimer():
        return render_template("disclaimer.html")

    @app.route("/api/chat", methods=["POST"])
    def chat():
        if agent is None:
            return jsonify({"success": False, "error": "Agent is not initialized."}), 500

        payload = request.get_json(force=True)
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        top_k = int(payload.get("top_k", config.TOP_K))

        if not message:
            return jsonify({"success": False, "error": "Message cannot be empty."}), 400

        clean_history: List[Tuple[str, str]] = []
        for item in history:
            if isinstance(item, list) and len(item) == 2:
                clean_history.append((str(item[0]), str(item[1])))

        conversation_context = _build_memory_summary(clean_history)
        result = agent.chat(
            message,
            top_k=top_k,
            return_details=True,
            conversation_context=conversation_context,
        )

        response_text = result.get("response", "")
        new_history = clean_history + [(message, response_text)]

        return jsonify(
            success=result.get("success", False),
            response=response_text,
            memory=_build_memory_panel(new_history),
            retrieved_documents=_format_retrieval_details(result),
            reasoning=result.get("reasoning", "No reasoning yet."),
            logs=_build_logs_text(new_history, top_k, result.get("execution_time")),
            history=new_history,
            error=result.get("error", None),
        )

    @app.route("/api/load_corpus", methods=["POST"])
    def load_corpus():
        global agent, current_disease
        payload = request.get_json(force=True)
        disease = (payload.get("disease") or config.DEFAULT_DISEASE).strip().lower()

        try:
            agent_instance, status = _load_agent_for_disease(disease)
            agent = agent_instance
            current_disease = disease
            return jsonify(
                success=True,
                status=status,
                memory="Memory\n\nConversation memory cleared after loading a new corpus.",
                retrieved_documents=[],
                reasoning="No reasoning yet.",
                logs="No queries have been run yet.",
            )
        except Exception as e:
            logger.error(f"Error loading corpus '{disease}': {str(e)}")
            return jsonify(success=False, error=str(e)), 500

    return app
