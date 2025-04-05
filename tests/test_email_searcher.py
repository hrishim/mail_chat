#!/usr/bin/env python3
import os
from email_searcher import EmailSearcher, SearchConfig
from utils import setup_debug_logging, log_debug

def test_thread_retrieval():
    """Test retrieving full email threads."""
    # Setup debug logging
    setup_debug_logging(True)
    
    # Initialize email searcher
    searcher = EmailSearcher(debug_log=True)
    
    # Get some documents
    config = SearchConfig(
        num_docs=2,
        rerank_method="Cosine Similarity",
        return_full_threads=True
    )
    docs = searcher.semantic_search("flight booking", config)
    
    # Test thread retrieval for each document
    for doc in docs:
        thread_id = doc.metadata.get('thread_id')
        if not thread_id:
            continue
            
        log_debug(f"\nTesting thread retrieval for ID: {thread_id}")
        thread_doc = searcher.get_full_thread(thread_id)
        
        if thread_doc:
            log_debug(f"Thread metadata:")
            for key, value in thread_doc.metadata.items():
                log_debug(f"  {key}: {value}")
            log_debug("\nThread content:")
            log_debug(thread_doc.page_content)
        else:
            log_debug("No thread found")

if __name__ == "__main__":
    test_thread_retrieval()
