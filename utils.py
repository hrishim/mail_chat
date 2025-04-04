import logging
import argparse

# Create logger
logger = logging.getLogger('rag_debug')
logger.addHandler(logging.NullHandler())  # Default to no-op handler

# Create parser for debug flag
parser = argparse.ArgumentParser(description='RAG Debug Settings')
parser.add_argument('--debugLog', action='store_true', help='Enable debug logging')
parser.add_argument('--debugLogPath', type=str, help='Optional path to save debug logs to file')
args = parser.parse_args([])  # Default to no args

def setup_debug_logging(debug_enabled: bool = False, debug_log_path: str = None) -> None:
    """Configure debug logging.
    
    Args:
        debug_enabled: Whether to enable debug logging
        debug_log_path: Optional path to save debug logs to file
    """
    global args
    args.debugLog = debug_enabled
    
    if debug_enabled:
        logger.setLevel(logging.DEBUG)
        
        # Always add console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if path is provided
        if debug_log_path:
            file_handler = logging.FileHandler(debug_log_path)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    else:
        # Disable logging if debug not enabled
        logger.setLevel(logging.ERROR)

def log_debug(message: str, *args) -> None:
    """Log a debug message.
    
    Args:
        message: The message to log
        *args: Additional arguments for string formatting
    """
    if args:
        message = message % args
    logger.debug(message)

def log_error(message: str, *args) -> None:
    """Helper function to log error messages regardless of debug setting.
    
    Args:
        message: The message to log
        *args: Additional arguments for string formatting
    """
    if args:
        message = message % args
    logger.error(message)
