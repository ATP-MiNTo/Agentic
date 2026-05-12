# Logger utility for structured logging
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import config

class RayLogger:
    """Centralized logger for the Medical RAG system."""
    
    def __init__(self, name: str = "MedicalRAG", log_file: Optional[Path] = None):
        """Initialize logger with console and file handlers.
        
        Args:
            name: Logger name
            log_file: Path to log file. If None, auto-generates one.
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Log format
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if config.SAVE_LOGS_TO_FILE:
            if log_file is None:
                log_file = self._generate_log_filepath()
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file = log_file
        else:
            self.log_file = None
    
    @staticmethod
    def _generate_log_filepath() -> Path:
        """Generate log file path with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        return config.LOG_DIR / f"agent_{timestamp}.log"
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_separator(self):
        """Log a separator line."""
        self.logger.info("-" * 80)
    
    def log_query(self, query: str):
        """Log user query."""
        self.log_separator()
        self.info(f"QUERY: {query}")
    
    def log_retrieval(self, docs: list, scores: list):
        """Log retrieved documents with scores.
        
        Args:
            docs: List of (doc_id, content) tuples
            scores: List of similarity scores
        """
        self.info("RETRIEVAL RESULTS:")
        for i, (doc_id, score) in enumerate(zip(docs, scores), 1):
            self.info(f"  [{i}] {doc_id:<50} Score: {score:.4f}")
    
    def log_reasoning(self, reasoning: str):
        """Log agent reasoning."""
        self.info(f"AGENT REASONING:\n{reasoning}")
    
    def log_response(self, response: str, tokens: int = None):
        """Log final response."""
        self.info(f"RESPONSE:\n{response}")
        if tokens:
            self.info(f"Tokens generated: {tokens}")
    
    def log_execution_time(self, component: str, elapsed_seconds: float):
        """Log execution time for a component."""
        self.info(f"{component} execution time: {elapsed_seconds:.2f}s")
    
    def log_error_with_context(self, error: Exception, context: str):
        """Log error with context."""
        self.error(f"Error in {context}: {str(error)}")
        self.debug(f"Error type: {type(error).__name__}")
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get path to log file."""
        return self.log_file


# Global logger instance
_logger_instance: Optional[RayLogger] = None

def get_logger(name: str = "MedicalRAG") -> RayLogger:
    """Get or create global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = RayLogger(name)
    return _logger_instance


# Convenience functions
def log_info(message: str):
    """Log info message using global logger."""
    get_logger().info(message)

def log_debug(message: str):
    """Log debug message using global logger."""
    get_logger().debug(message)

def log_error(message: str):
    """Log error message using global logger."""
    get_logger().error(message)

def log_warning(message: str):
    """Log warning message using global logger."""
    get_logger().warning(message)
