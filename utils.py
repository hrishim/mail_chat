import logging
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Email Chatbot with RAG')
parser.add_argument('--debugLog', type=str, help='Path to the debug log file. If not specified, debug logging will be disabled.')
args = parser.parse_args()

# Configure logging only if debug log file is specified
logger = logging.getLogger('rag_debug')
if args.debugLog:
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(args.debugLog)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
else:
    # Disable logging if no debug file specified
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())

def log_debug(message: str) -> None:
    """Helper function to log debug messages only if debug logging is enabled."""
    if args.debugLog:
        logger.debug(message)

def log_error(message: str) -> None:
    """Helper function to log error messages regardless of debug setting."""
    logger.error(message)
