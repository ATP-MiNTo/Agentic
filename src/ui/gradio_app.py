# Gradio web interface for Medical RAG
try:
    import gradio as gr
except Exception as e:
    raise RuntimeError(
        "Failed to import gradio (or its audio dependencies). On Windows, this often requires extra audio packages or compatible wheels.\n"
        "Quick fixes: install audio support in the venv (e.g. 'pip install pyaudio') or use a Python version with compatible wheels (3.11).\n"
        f"Original import error: {e}"
    )
from typing import Dict, List, Tuple
import socket
import config
from src.agent.rag_agent import MedicalRAGAgent
from src.utils.index_builder import FAISSIndexBuilder
from src.utils.logger import get_logger

logger = get_logger()


def _patch_gradio_template_response() -> None:
    """Patch Gradio's template response helper for Starlette API compatibility."""
    try:
        import gradio.routes as gradio_routes
    except Exception:
        return

    templates = getattr(gradio_routes, "templates", None)
    if templates is None or getattr(templates, "_copilot_template_response_patched", False):
        return

    original_template_response = templates.TemplateResponse

    def compat_template_response(*args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict) and "request" in args[1]:
            template_name = args[0]
            context = args[1]
            request = context["request"]
            return original_template_response(request, template_name, context, **kwargs)

        return original_template_response(*args, **kwargs)

    templates.TemplateResponse = compat_template_response
    templates._copilot_template_response_patched = True


def _find_free_port(start_port: int, max_attempts: int = 10) -> int:
    """Find the first available port at or above the requested port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((config.GRADIO_SERVER_NAME, port)) != 0:
                return port
    return start_port


class GradioApp:
    """Gradio interface for Medical RAG Agent."""
    
    def __init__(self, default_disease: str = None):
        """Initialize Gradio app."""
        logger.info("Initializing Gradio interface...")
        self.agent = None
        self.current_disease = None
        try:
            self.load_disease_corpus(default_disease or config.DEFAULT_DISEASE)
            logger.info("✓ Agent loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def load_disease_corpus(self, disease: str) -> str:
        """Build and load the FAISS index for a selected disease corpus."""
        disease = (disease or "").strip().lower()
        if disease not in config.DISEASE_OPTIONS:
            raise ValueError(f"Unsupported disease '{disease}'")

        disease_dir = config.DATA_DIR if disease == "all" else config.DATA_DIR / disease
        logger.info(f"Loading disease corpus: {disease_dir}")

        builder = FAISSIndexBuilder(data_dir=disease_dir)
        if not builder.build_index():
            raise RuntimeError(f"Failed to build index for {disease}")

        self.agent = MedicalRAGAgent()
        self.current_disease = disease
        return f"Loaded corpus: **{disease}**"

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 260) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"

    def _build_memory_summary(self, history: List[Tuple[str, str]], max_turns: int = 4) -> str:
        """Create a compact memory summary from recent chat turns."""
        if not history:
            return ""

        recent_turns = history[-max_turns:]
        lines: List[str] = []
        for index, (user_message, assistant_message) in enumerate(recent_turns, 1):
            lines.append(f"Turn {index} User: {self._truncate_text(user_message, 200)}")
            lines.append(f"Turn {index} Assistant: {self._truncate_text(assistant_message, 260)}")

        summary = "\n".join(lines)
        return summary[:1500]

    def _format_retrieval_text(self, result: Dict) -> str:
        retrieval_text = "### Retrieved Documents\n\n"
        retrieved_documents = result.get("retrieved_documents", [])
        if retrieved_documents:
            for i, doc in enumerate(retrieved_documents, 1):
                retrieval_text += f"**Document {i}: {doc['id']}**\n"
                retrieval_text += f"- Score: {doc['score']:.1%}\n"
                retrieval_text += f"- Preview: {doc['preview']}...\n\n"
        else:
            retrieval_text += "No documents retrieved."
        return retrieval_text

    def _format_reasoning_text(self, result: Dict) -> str:
        reasoning = result.get("reasoning", "No reasoning available.")
        return f"### Agent Reasoning\n\n{reasoning}"

    def _format_logs_text(self, result: Dict, chat_history: List[Tuple[str, str]]) -> str:
        logs_text = "### Execution Details\n\n"
        logs_text += f"- Query: {result.get('query', '')}\n"
        logs_text += f"- Documents Retrieved: {len(result.get('retrieved_documents', []))}\n"
        logs_text += f"- Chat Turns in Memory: {len(chat_history)}\n"
        logs_text += f"- Execution Time: {result.get('execution_time', 0.0):.2f}s\n"
        if result.get("conversation_context"):
            logs_text += f"- Memory Used: Yes\n"
        else:
            logs_text += f"- Memory Used: No\n"
        return logs_text

    def _build_memory_panel(self, history: List[Tuple[str, str]]) -> str:
        summary = self._build_memory_summary(history)
        if not summary:
            return "**Memory**\n\nNo conversation memory yet. Ask a question to start a chat thread."

        return (
            "**Memory**\n\n"
            "The assistant uses the recent turns below to resolve follow-up questions.\n\n"
            f"```text\n{summary}\n```"
        )

    def _chat_turn(
        self,
        message: str,
        history: List[Tuple[str, str]],
        top_k: int,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str, str, str, str, str]:
        """Process a single chat turn and return updated UI state.
        
        Returns:
            (updated_history_for_state, chatbot_display, memory_md, retrieval_md, reasoning_md, logs_md, cleared_input)
        """
        history = history or []
        message = (message or "").strip()

        if not message:
            return (
                history,
                history,
                self._build_memory_panel(history),
                "### Retrieved Documents\n\nPlease enter a medical question.",
                "### Agent Reasoning\n\nNo question was entered.",
                "### Execution Details\n\n- No query submitted.",
                "",
            )

        conversation_context = self._build_memory_summary(history)

        logger.info(f"Processing query from web UI: {message[:50]}...")

        try:
            result = self.agent.chat(
                message,
                top_k=top_k,
                return_details=True,
                conversation_context=conversation_context,
            )

            if not result.get("success"):
                response_text = result.get("response", "An error occurred.")
                updated_history = history + [(message, response_text)]
                return (
                    updated_history,
                    updated_history,
                    self._build_memory_panel(updated_history),
                    "### Retrieved Documents\n\nNo documents available because the answer failed.",
                    "### Agent Reasoning\n\nThe request could not be completed.",
                    f"### Execution Details\n\n- Error: {result.get('error', 'Unknown error')}",
                    "",
                )

            response_text = result["response"]
            updated_history = history + [(message, response_text)]

            return (
                updated_history,
                updated_history,
                self._build_memory_panel(updated_history),
                self._format_retrieval_text(result),
                self._format_reasoning_text(result),
                self._format_logs_text(result, updated_history),
                "",
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            error_text = f"An error occurred: {str(e)}"
            updated_history = history + [(message, error_text)]
            return (
                updated_history,
                updated_history,
                self._build_memory_panel(updated_history),
                "### Retrieved Documents\n\nAn error occurred before retrieval details could be produced.",
                "### Agent Reasoning\n\nAn error occurred before reasoning could be produced.",
                f"### Execution Details\n\n- Error: {str(e)}",
                "",
            )

    def _load_corpus_and_reset_chat(self, disease: str):
        """Load a new corpus and clear chat memory so the new topic starts clean."""
        status = self.load_disease_corpus(disease)
        return (
            status,
            [],
            [],
            "**Memory**\n\nConversation memory cleared after loading a new corpus.",
            "### Retrieved Documents\n\nNo documents retrieved yet.",
            "### Agent Reasoning\n\nNo reasoning yet.",
            "### Execution Details\n\nNo queries have been run yet.",
            "",  # input cleared
        )
    
    def process_query(self, query: str, top_k: int) -> Dict:
        """Process query and return formatted results.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary with formatted outputs for UI
        """
        if not query or not query.strip():
            return (
                "Please enter a medical question.",
                "",
                "",
                ""
            )

        if self.agent is None:
            return (
                "No disease corpus is loaded. Please load one first.",
                "",
                "",
                "No active agent"
            )
        
        logger.info(f"Processing query from web UI: {query[:50]}...")
        
        try:
            result = self.agent.chat(query, top_k=top_k, return_details=True)
            
            if not result["success"]:
                return (
                    result["response"],
                    "",
                    "",
                    f"Error: {result.get('error', 'Unknown error')}"
                )
            
            # Format retrieval details
            retrieval_text = "### Retrieved Documents\n\n"
            if result["retrieved_documents"]:
                for i, doc in enumerate(result["retrieved_documents"], 1):
                    retrieval_text += f"**Document {i}: {doc['id']}**\n"
                    retrieval_text += f"- Score: {doc['score']:.1%}\n"
                    retrieval_text += f"- Preview: {doc['preview']}...\n\n"
            else:
                retrieval_text += "No documents retrieved."
            
            # Format reasoning
            reasoning_text = f"### Agent Reasoning\n\n{result['reasoning']}"
            
            # Format execution info
            logs_text = f"### Execution Details\n\n"
            logs_text += f"- Query: {result['query']}\n"
            logs_text += f"- Documents Retrieved: {len(result['retrieved_documents'])}\n"
            logs_text += f"- Execution Time: {result['execution_time']:.2f}s\n"
            
            return (
                result["response"],
                retrieval_text,
                reasoning_text,
                logs_text
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return (
                f"An error occurred: {str(e)}",
                "",
                "",
                f"Error: {str(e)}"
            )
    
    def build_ui(self):
        """Build Gradio interface."""
        
        # Custom CSS for chat-style styling
        css = """
        .gradio-container {
            background: radial-gradient(circle at top, rgba(255,255,255,0.95), rgba(241,245,249,0.98));
        }
        .chat-shell {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            overflow: hidden;
        }
        .chat-header {
            padding: 18px 22px 8px 22px;
        }
        .chat-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: rgba(248, 250, 252, 0.9);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }
        .sidebar-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            background: rgba(255,255,255,0.9);
            padding: 16px;
        }
        .memory-box pre {
            white-space: pre-wrap;
            word-break: break-word;
        }
        """
        
        with gr.Blocks(
            title="Medical Chat RAG",
            theme=gr.themes.Soft(
                primary_hue="emerald",
                secondary_hue="slate",
                neutral_hue="slate"
            ),
            css=css
        ) as app:
            chat_history_state = gr.State([])

            with gr.Column(elem_classes=["chat-shell"]):
                with gr.Column(elem_classes=["chat-header"]):
                    gr.Markdown("# 🏥 Medical Chat Assistant")
                    gr.Markdown(
                        "Chat with the RAG assistant about migraine, diabetes, or cancer. "
                        "The last few turns stay in memory so you can ask follow-up questions naturally. "
                        "This is for educational purposes only."
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            value=[],
                            height=540,
                            bubble_full_width=False,
                            show_copy_button=True,
                        )
                        query_input = gr.Textbox(
                            label="Message",
                            placeholder="Ask a question, then follow up with things like 'what about the symptoms you mentioned?'",
                            lines=1,
                            show_label=False,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary")
                            # Clear button moved to chat header (beside conversation bubble)

                    with gr.Column(scale=1, elem_classes=["sidebar-card"]):
                        disease_selector = gr.Dropdown(
                            choices=config.DISEASE_OPTIONS,
                            value=self.current_disease or config.DEFAULT_DISEASE,
                            label="Corpus",
                            info="Switch between disease collections"
                        )
                        load_button = gr.Button("Load corpus", variant="secondary")
                        corpus_status = gr.Markdown(
                            value=f"Loaded corpus: **{self.current_disease or 'none'}**"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-K",
                            info="How many chunks to retrieve"
                        )
                        memory_output = gr.Markdown(
                            value="**Memory**\n\nNo conversation memory yet."
                        )

                with gr.Accordion("Retrieved documents", open=False):
                    retrieval_output = gr.Markdown(value="No retrieval details yet.")

                with gr.Accordion("Reasoning", open=False):
                    reasoning_output = gr.Markdown(value="No reasoning yet.")

                with gr.Accordion("Execution logs", open=False):
                    logs_output = gr.Markdown(value="No queries have been run yet.")

                gr.Examples(
                    examples=[
                        ["What are the early symptoms of diabetes?", 5],
                        ["How is migraine different from a regular headache?", 5],
                        ["What are cancer risk factors?", 5],
                        ["When should I see a doctor for a migraine?", 5],
                        ["How is Type 2 diabetes managed?", 5],
                    ],
                    inputs=[query_input, top_k_slider],
                    label="Example questions"
                )

            def _submit_from_ui(message, history, top_k):
                return self._chat_turn(message, history, top_k)

            def _clear_history():
                return [], [], "**Memory**\n\nConversation cleared.", "### Retrieved Documents\n\nNo retrieval details yet.", "### Agent Reasoning\n\nNo reasoning yet.", "### Execution Details\n\nNo queries have been run yet.", ""

            send_outputs = [
                chat_history_state,
                chatbot,
                memory_output,
                retrieval_output,
                reasoning_output,
                logs_output,
                query_input,
            ]

            send_btn.click(
                fn=_submit_from_ui,
                inputs=[query_input, chat_history_state, top_k_slider],
                outputs=send_outputs,
            )

            # Allow Enter key in the textbox to submit the same handler as the Send button
            query_input.submit(
                fn=_submit_from_ui,
                inputs=[query_input, chat_history_state, top_k_slider],
                outputs=send_outputs,
            )

            # New chat button placed in header; define click binding below after header created
            clear_btn = gr.Button("New chat", variant="secondary")
            clear_btn.click(
                fn=_clear_history,
                inputs=[],
                outputs=[chat_history_state, chatbot, memory_output, retrieval_output, reasoning_output, logs_output, query_input],
            )

            load_button.click(
                fn=self._load_corpus_and_reset_chat,
                inputs=[disease_selector],
                outputs=[corpus_status, chat_history_state, chatbot, memory_output, retrieval_output, reasoning_output, logs_output, query_input],
            )
        
        return app


def main():
    """Main entry point for web UI."""
    try:
        logger.info("Starting Gradio web interface...")
        _patch_gradio_template_response()
        
        # Patch the broken schema helpers to handle bool schema values.
        try:
            from gradio_client import utils as gradio_utils
            original_json_schema_to_python_type = gradio_utils.json_schema_to_python_type
            original__json_schema_to_python_type = gradio_utils._json_schema_to_python_type
            original_get_type = gradio_utils.get_type

            def patched__json_schema_to_python_type(schema, defs=None):
                """Patched version that handles bool schema gracefully."""
                if isinstance(schema, bool):
                    return "unknown"
                return original__json_schema_to_python_type(schema, defs)

            def patched_json_schema_to_python_type(schema):
                """Patched wrapper that preserves the original public API."""
                if isinstance(schema, bool):
                    return "unknown"
                return patched__json_schema_to_python_type(schema, schema.get("$defs"))
            
            def patched_get_type(schema):
                """Patched version that handles bool schema gracefully."""
                # If schema is a bool, return "unknown"
                if isinstance(schema, bool):
                    return "unknown"
                # Otherwise use original
                return original_get_type(schema)
            
            gradio_utils.json_schema_to_python_type = patched_json_schema_to_python_type
            gradio_utils._json_schema_to_python_type = patched__json_schema_to_python_type
            gradio_utils.get_type = patched_get_type
        except Exception as e:
            logger.warning(f"Failed to patch schema generator: {e}")
        
        app = GradioApp(default_disease=config.DEFAULT_DISEASE)
        ui = app.build_ui()

        launch_port = _find_free_port(config.GRADIO_SERVER_PORT)
        if launch_port != config.GRADIO_SERVER_PORT:
            logger.warning(
                f"Port {config.GRADIO_SERVER_PORT} is busy, falling back to {launch_port}"
            )
        
        logger.info(f"Launching at http://{config.GRADIO_SERVER_NAME}:{launch_port}")
        
        ui.launch(
            server_name=config.GRADIO_SERVER_NAME,
            server_port=launch_port,
            share=config.GRADIO_SHARE,
            debug=config.GRADIO_DEBUG
        )
        
    except KeyboardInterrupt:
        logger.info("Web interface stopped by user")
    except Exception as e:
        logger.error(f"Error in web interface: {str(e)}")
        raise


if __name__ == "__main__":
    main()
