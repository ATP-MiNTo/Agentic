# Gradio web interface for Medical RAG
import gradio as gr
from typing import Dict
import config
from src.agent.rag_agent import MedicalRAGAgent
from src.utils.logger import get_logger

logger = get_logger()


class GradioApp:
    """Gradio interface for Medical RAG Agent."""
    
    def __init__(self):
        """Initialize Gradio app."""
        logger.info("Initializing Gradio interface...")
        try:
            self.agent = MedicalRAGAgent()
            logger.info("✓ Agent loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise
    
    def process_query(self, query: str, top_k: int) -> Dict:
        """Process query and return formatted results.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary with formatted outputs for UI
        """
        if not query or not query.strip():
            return {
                "response": "Please enter a medical question.",
                "retrieval": "",
                "reasoning": "",
                "logs": ""
            }
        
        logger.info(f"Processing query from web UI: {query[:50]}...")
        
        try:
            result = self.agent.chat(query, top_k=top_k, return_details=True)
            
            if not result["success"]:
                return {
                    "response": result["response"],
                    "retrieval": "",
                    "reasoning": "",
                    "logs": f"Error: {result.get('error', 'Unknown error')}"
                }
            
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
            
            return {
                "response": result["response"],
                "retrieval": retrieval_text,
                "reasoning": reasoning_text,
                "logs": logs_text
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"An error occurred: {str(e)}",
                "retrieval": "",
                "reasoning": "",
                "logs": f"Error: {str(e)}"
            }
    
    def build_ui(self):
        """Build Gradio interface."""
        
        # Custom CSS for styling
        css = """
        .response-box {
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0;
            background-color: #f0f8f0;
        }
        .accordion-section {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 10px 0;
        }
        .info-badge {
            display: inline-block;
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(
            title="Medical RAG PoC",
            theme=gr.themes.Soft(),
            css=css
        ) as app:
            
            # Header
            gr.Markdown("# 🏥 Medical Information RAG Assistant")
            gr.Markdown("""
            This is a **Proof of Concept** demonstrating Retrieval-Augmented Generation for medical information.
            
            **Topics Covered**: Migraine, Diabetes, Cancer
            
            **⚠️ DISCLAIMER**: This is for educational purposes only. Always consult healthcare professionals 
            for medical decisions.
            """)
            
            # Input section
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="Ask a Medical Question",
                        placeholder="Example: What are the symptoms of diabetes?",
                        lines=2,
                        info="Ask about migraine, diabetes, or cancer"
                    )
                
                with gr.Column(scale=1):
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top-K Documents",
                        info="Number of documents to retrieve"
                    )
            
            # Submit button
            submit_btn = gr.Button(
                "🔍 Search",
                variant="primary",
                size="lg"
            )
            
            # Output sections
            with gr.Column():
                # Main response
                response_output = gr.Markdown(
                    label="Response",
                    value="*Response will appear here*"
                )
                
                # Accordion sections (foldable)
                with gr.Accordion("📊 Retrieved Documents & Scores", open=False):
                    retrieval_output = gr.Markdown(
                        value="*Document retrieval details will appear here*"
                    )
                
                with gr.Accordion("🧠 Agent Reasoning Process", open=False):
                    reasoning_output = gr.Markdown(
                        value="*Agent reasoning will appear here*"
                    )
                
                with gr.Accordion("📋 Execution Logs", open=False):
                    logs_output = gr.Markdown(
                        value="*Execution details will appear here*"
                    )
            
            # Connect button to processing function
            submit_btn.click(
                fn=self.process_query,
                inputs=[query_input, top_k_slider],
                outputs={
                    "response": response_output,
                    "retrieval": retrieval_output,
                    "reasoning": reasoning_output,
                    "logs": logs_output
                }
            )
            
            # Allow Enter key to submit
            query_input.submit(
                fn=self.process_query,
                inputs=[query_input, top_k_slider],
                outputs={
                    "response": response_output,
                    "retrieval": retrieval_output,
                    "reasoning": reasoning_output,
                    "logs": logs_output
                }
            )
            
            # Example queries
            gr.Examples(
                examples=[
                    ["What are the early symptoms of diabetes?", 5],
                    ["How is migraine different from a regular headache?", 5],
                    ["What are cancer risk factors?", 5],
                    ["When should I see a doctor for a migraine?", 5],
                    ["How is Type 2 diabetes managed?", 5],
                ],
                inputs=[query_input, top_k_slider],
                label="Example Questions"
            )
        
        return app


def main():
    """Main entry point for web UI."""
    try:
        logger.info("Starting Gradio web interface...")
        
        app = GradioApp()
        ui = app.build_ui()
        
        logger.info(f"Launching at http://{config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}")
        
        ui.launch(
            server_name=config.GRADIO_SERVER_NAME,
            server_port=config.GRADIO_SERVER_PORT,
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
