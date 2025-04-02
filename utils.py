import logging

# Create logger
logger = logging.getLogger('rag_debug')
logger.addHandler(logging.NullHandler())  # Default to no-op handler

def setup_debug_logging(debug_log_path: str = None) -> None:
    """Configure debug logging.
    
    Args:
        debug_log_path: Path to the debug log file. If None, debug logging will be disabled.
    """
    if debug_log_path:
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(debug_log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Disable logging if no debug file specified
        logger.setLevel(logging.ERROR)

def log_debug(message: str) -> None:
    """Log a debug message."""
    logger.debug(message)

def log_error(message: str) -> None:
    """Helper function to log error messages regardless of debug setting."""
    logger.error(message)
