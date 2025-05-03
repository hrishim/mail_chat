#!/usr/bin/env python3
import os
import sys
import types
import argparse
import time

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from email_searcher import EmailSearcher, SearchConfig, SearchResult
from utils import setup_debug_logging, log_debug
from langchain_core.documents import Document

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test email searcher token limiting functionality")
    parser.add_argument("--logFile", type=str, required=True, help="Path to log file for debug logging and document information")
    return parser.parse_args()

def test_token_limiting(log_file=None):
    """Test token limiting functionality."""
    # Ensure log file is specified
    if not log_file:
        print("ERROR: logFile is required")
        sys.exit(1)
        
    # Setup debug logging with the same log file
    setup_debug_logging(debug_enabled=True, debug_log_path=log_file)
    log_debug("Starting test email searcher")
    
    # Force DB_TYPE to use Chroma for this test
    os.environ["DB_TYPE"] = "langchain_chroma"
    
    # Initialize email searcher with debug logging and log file
    searcher = EmailSearcher(debug_log=True, log_file=log_file)
    
    # Debug Chroma database
    log_debug("\n=== CHROMA DATABASE DEBUG ===")
    if hasattr(searcher.vectorstore, '_client'):
        # List all collections
        try:
            collections = searcher.vectorstore._client.list_collections()
            log_debug(f"Available collections: {[c.name for c in collections]}")
        except Exception as e:
            log_debug(f"Error listing collections: {str(e)}")
            
    if hasattr(searcher.vectorstore, '_collection'):
        count = searcher.vectorstore._collection.count()
        log_debug(f"Chroma collection count: {count}")
        
        # Try to get some sample documents to check metadata
        try:
            sample_ids = searcher.vectorstore._collection.get(limit=5)
            log_debug(f"Sample document IDs: {sample_ids['ids']}")
            log_debug(f"Sample document metadatas: {sample_ids['metadatas']}")
        except Exception as e:
            log_debug(f"Error getting sample documents: {str(e)}")
    else:
        log_debug("Chroma collection not available for inspection")
        
    # Debug SQLite database
    log_debug("\n=== SQLITE DATABASE DEBUG ===")
    log_debug(f"SQLite DB path: {searcher.sqlite_db_path}")
    log_debug(f"EmailStore initialized: {searcher.email_store is not None}")
    if searcher.email_store:
        try:
            # Try to get a count of documents in the SQLite database
            count = searcher.email_store.get_document_count()
            log_debug(f"SQLite document count: {count}")
        except Exception as e:
            log_debug(f"Error getting SQLite document count: {str(e)}")
    
    # Test token estimation using tiktoken
    text = "This is a test string for token estimation."
    log_debug(f"\n=== TOKEN ESTIMATION TEST ===")
    log_debug(f"Text: '{text}'")
    log_debug(f"Text length: {len(text)} characters")
    
    # Get token count from tiktoken
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4's encoding
    tokens = len(encoding.encode(text))
    log_debug(f"Actual tokens (tiktoken): {tokens}")
    log_debug(f"Tokens per char: {tokens/len(text):.3f}")
    
    # Compare with estimator
    estimated_tokens = searcher.estimate_tokens(text)
    log_debug(f"Estimated tokens (4 chars/token): {estimated_tokens}")
    log_debug(f"Difference: {abs(tokens - estimated_tokens)} tokens\n")
    
    log_debug("=== EMAIL SEARCHER TEST LOGS ===")
    log_debug(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test semantic search with token limiting
    query = "Where did I pay FASTag Toll on 2 Dec 2024?"
    log_debug(f"Testing semantic search with query: {query}")
    
    config = SearchConfig(
        num_docs=20,
        rerank_multiplier=3,
        rerank_method="Cosine Similarity",
        max_total_tokens=500  # Set a lower limit to test token limiting
    )
    log_debug(f"Created search config: num_docs={config.num_docs}, rerank={config.rerank_method}")
    log_debug(f"Config: num_docs={config.num_docs}, rerank_multiplier={config.rerank_multiplier}, "
             f"rerank_method={config.rerank_method}, max_total_tokens={config.max_total_tokens}\n")
    
    # Perform the search
    log_debug("Executing semantic search...")
    result = searcher.semantic_search(query, config)
    log_debug("Search completed")
    log_debug(f"Search returned {len(result.documents)} documents with {result.total_tokens} tokens out of {result.total_docs} total matches")
    
    # Log document details
    if result.documents:
        log_debug("\n=== DOCUMENT DETAILS ===")
        for i, doc in enumerate(result.documents):
            log_debug(f"\nDocument {i+1} of {len(result.documents)}:")
            
            # Show similarity score if available
            if 'similarity_score' in doc.metadata:
                log_debug(f"Similarity Score: {doc.metadata['similarity_score']:.4f}")
            
            # Show split information if this is a split document
            if doc.metadata.get('is_split'):
                split_index = doc.metadata.get('split_index', 0)
                total_splits = doc.metadata.get('total_splits', 1)
                log_debug(f"[Part {split_index + 1} of {total_splits}]")
            
            log_debug(f"  Content preview: {doc.page_content[:100]}...")
            log_debug(f"  Metadata: {doc.metadata}")
            
            # Try to get full thread if available
            thread_id = doc.metadata.get('thread_id')
            message_index = doc.metadata.get('message_index')
            if thread_id is not None and message_index is not None:
                log_debug(f"\nRetrieving full thread for document {i+1}...")
                full_thread = searcher.get_full_thread(thread_id, message_index)
                if full_thread:
                    log_debug(f"Retrieved full thread with {len(full_thread.page_content)} characters")
                    log_debug(f"Thread ID: {thread_id}")
                    log_debug(f"Message Index: {message_index}")
                    if doc.metadata.get('is_split'):
                        split_index = doc.metadata.get('split_index', 0)
                        total_splits = doc.metadata.get('total_splits', 1)
                        log_debug(f"[Part {split_index + 1} of {total_splits}]")
                    log_debug(f"Content:\n{full_thread.page_content}\n")
                    log_debug("Metadata:")
                    for key, value in full_thread.metadata.items():
                        log_debug(f"  {key}: {value}")
                    log_debug(f"Estimated tokens: {searcher.estimate_tokens(full_thread.page_content)}\n")
                else:
                    log_debug("Failed to retrieve full thread\n")
    
    # Log process docs result
    log_debug("\n=== SEARCH RESULTS SUMMARY ===")
    log_debug(f"Total documents found: {len(result.documents)}")
    log_debug(f"Total tokens: {result.total_tokens}")
    log_debug(f"Total matched docs: {result.total_docs}")
    if result.split_docs:
        log_debug(f"Documents that were split: {len(result.split_docs)}")
        for doc_id, splits in result.split_docs.items():
            log_debug(f"  {doc_id}: {len(splits)} splits")
    
    log_debug("\n=== TEST COMPLETED ===")
    log_debug(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test results saved to: {log_file}")

if __name__ == "__main__":
    args = parse_args()
    test_token_limiting(args.logFile)
