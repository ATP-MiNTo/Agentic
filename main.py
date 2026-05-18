# Ensure .env is loaded before anything else
import src.utils.load_env
# CLI entry point for Medical RAG
import argparse
import time
from src.agent.rag_agent import MedicalRAGAgent
from src.utils.logger import get_logger

logger = get_logger()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Medical RAG PoC - AI Agent with Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "What are migraine symptoms?"
  python main.py --interactive
  python main.py --rebuild-index
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single medical query to process"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild FAISS index from documents"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle rebuild index
    if args.rebuild_index:
        from src.utils.index_builder import FAISSIndexBuilder
        logger.info("Rebuilding FAISS index...")
        builder = FAISSIndexBuilder()
        success = builder.build_index()
        exit(0 if success else 1)
    
    # Initialize agent
    logger.info("Initializing Medical RAG Agent...")
    try:
        agent = MedicalRAGAgent()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        logger.info("Tip: Run 'python src/utils/index_builder.py' to build the index")
        exit(1)
    
    # Handle single query
    if args.query:
        logger.info(f"Processing query: {args.query}")
        start_time = time.time()
        
        result = agent.chat(args.query, top_k=args.top_k, return_details=True)
        
        print("\n" + "="*80)
        print("RESPONSE")
        print("="*80)
        print(result["response"])
        
        print("\n" + "="*80)
        print("RETRIEVED DOCUMENTS")
        print("="*80)
        for i, doc in enumerate(result["retrieved_documents"], 1):
            print(f"\n[{i}] {doc['id']}")
            print(f"    Score: {doc['score']:.1%}")
            print(f"    Preview: {doc['preview'][:150]}...")
        
        print("\n" + "="*80)
        print("AGENT REASONING")
        print("="*80)
        print(result["reasoning"])
        
        elapsed = time.time() - start_time
        print(f"\nExecution time: {elapsed:.2f}s")
        
    # Handle interactive mode
    elif args.interactive:
        logger.info("Starting interactive mode (type 'exit' to quit)")
        print("\n" + "="*80)
        print("Medical RAG Interactive Chat")
        print("="*80)
        print("Type your medical questions. Commands:")
        print("  'exit' - Exit interactive mode")
        print("  'help' - Show this help message")
        print("="*80 + "\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() == "exit":
                    logger.info("Exiting interactive mode")
                    break
                
                if query.lower() == "help":
                    print("Commands:")
                    print("  'exit' - Exit interactive mode")
                    print("  Any medical question - Get answer from RAG agent")
                    continue
                
                if not query:
                    continue
                
                print("\nProcessing...\n")
                start_time = time.time()
                
                result = agent.chat(query, top_k=args.top_k)
                elapsed = time.time() - start_time
                
                print(f"Assistant: {result['response']}")
                print(f"\n(Response time: {elapsed:.2f}s)\n")
                
            except KeyboardInterrupt:
                logger.info("Chat interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print("An error occurred. Please try again.\n")
    
    else:
        # Default: show help
        parser.print_help()
        print("\nQuick start:")
        print("  python main.py --query \"What are diabetes symptoms?\"")
        print("  python main.py --interactive")


if __name__ == "__main__":
    main()
