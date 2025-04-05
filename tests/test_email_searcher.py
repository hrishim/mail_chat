#!/usr/bin/env python3
import os
import sys
import types
import argparse

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from email_searcher import EmailSearcher, SearchConfig, SearchResult
from utils import setup_debug_logging, log_debug
from langchain_core.documents import Document

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test email searcher token limiting functionality")
    parser.add_argument("--debugLog", type=str, help="Path to debug log file. If not specified, debug logging will be disabled.")
    parser.add_argument("--outputDir", type=str, default="test_results", help="Directory to save test results")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode without vector database")
    return parser.parse_args()

def test_token_limiting(output_dir="test_results", testing_mode=False):
    """Test token limiting functionality."""
    # Setup debug logging
    log_debug("Starting token limiting test")
    
    # Initialize email searcher
    searcher = EmailSearcher(debug_log=True, testing=testing_mode)
    
    # Test token estimation
    text = "This is a test string for token estimation."
    tokens = searcher.estimate_tokens(text)
    log_debug(f"Text: '{text}'")
    log_debug(f"Estimated tokens: {tokens}")
    
    # If we're in testing mode, we need to test with mock data
    if testing_mode:
        # Test document splitting with mock data
        long_doc = Document(
            page_content="This is a very long document. " * 100,  # Repeat to make it long
            metadata={"test_key": "test_value"}
        )
        
        # Test _split_document method
        splits = searcher._split_document(long_doc, 50)  # Split into ~50 token chunks
        log_debug(f"Split document into {len(splits)} parts")
        for i, split in enumerate(splits[:2]):  # Show first 2 splits
            log_debug(f"Split {i} content: {split.page_content[:50]}...")
            log_debug(f"Split {i} metadata: {split.metadata}")
        
        # Test _process_docs_with_limit method with mock data
        docs = [
            Document(page_content="Short document 1", metadata={"id": "1"}),
            Document(page_content="Short document 2", metadata={"id": "2"}),
            Document(page_content="Short document 3", metadata={"id": "3"}),
            Document(page_content="Short document 4", metadata={"id": "4"}),
            Document(page_content="Short document 5", metadata={"id": "5"})
        ]
        
        # Process with a limit that should include only 3 documents
        token_limit = searcher.estimate_tokens("Short document 1Short document 2Short document 3")
        result = searcher._process_docs_with_limit(docs, token_limit)
    else:
        # Test with real data from the vector database
        log_debug("Testing with real data from vector database")
        
        # Test semantic search with token limiting
        query = "FASTag Toll payment"
        config = SearchConfig(
            num_docs=10,
            rerank_multiplier=3,
            rerank_method="Cosine Similarity",
            max_total_tokens=500  # Set a lower limit to test pagination
        )
        
        # Perform search
        result = searcher.semantic_search(query, config)
        log_debug(f"Search returned {len(result.documents)} documents with {result.total_tokens} tokens")
        log_debug(f"Has more: {result.has_more}, Next doc index: {result.next_doc_index}, Total docs: {result.total_docs}")
        
        # Test pagination if there are more results
        if result.has_more:
            log_debug("Testing pagination")
            config.start_index = result.next_doc_index
            next_result = searcher.semantic_search(query, config)
            log_debug(f"Next page returned {len(next_result.documents)} documents with {next_result.total_tokens} tokens")
            log_debug(f"Has more: {next_result.has_more}, Next doc index: {next_result.next_doc_index}")
        
        # Test get_full_thread
        if result.documents:
            doc = result.documents[0]
            if 'thread_id' in doc.metadata and 'full_content_id' in doc.metadata:
                thread_id = doc.metadata['thread_id']
                message_index = int(doc.metadata['full_content_id'].split('_')[-1]) if '_' in doc.metadata['full_content_id'] else 0
                log_debug(f"Testing get_full_thread with thread_id: {thread_id}, message_index: {message_index}")
                full_thread = searcher.get_full_thread(thread_id, message_index)
                if full_thread:
                    log_debug(f"Retrieved full thread with {len(full_thread.page_content)} characters")
                else:
                    log_debug("Failed to retrieve full thread")
    
    # Log results
    log_debug(f"Process docs result:")
    log_debug(f"  Documents: {len(result.documents)}")
    log_debug(f"  Total tokens: {result.total_tokens}")
    log_debug(f"  Has more: {result.has_more}")
    log_debug(f"  Next doc index: {result.next_doc_index}")
    log_debug(f"  Total docs: {result.total_docs}")
    
    # Test pagination if in testing mode
    if testing_mode and result.has_more:
        next_result = searcher._process_docs_with_limit(docs, token_limit, result.next_doc_index)
        log_debug(f"Next page result:")
        log_debug(f"  Documents: {len(next_result.documents)}")
        log_debug(f"  Total tokens: {next_result.total_tokens}")
        log_debug(f"  Has more: {next_result.has_more}")
    
    log_debug("Token limiting test completed successfully")
    
    # Save test results to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "token_limiting_results.txt"), "w") as f:
        f.write(f"Token estimation test:\n")
        f.write(f"  Text: '{text}'\n")
        f.write(f"  Estimated tokens: {tokens}\n\n")
        
        if testing_mode:
            f.write(f"Document splitting test (mock data):\n")
            f.write(f"  Split document into {len(splits)} parts\n")
            for i, split in enumerate(splits[:2]):
                f.write(f"  Split {i} content: {split.page_content[:50]}...\n")
                f.write(f"  Split {i} metadata: {split.metadata}\n\n")
        else:
            f.write(f"Semantic search test (real data):\n")
            f.write(f"  Query: '{query}'\n")
            f.write(f"  Search returned {len(result.documents)} documents with {result.total_tokens} tokens\n")
            f.write(f"  Has more: {result.has_more}, Next doc index: {result.next_doc_index}, Total docs: {result.total_docs}\n\n")
            
            if result.documents:
                f.write(f"First document:\n")
                f.write(f"  Content preview: {result.documents[0].page_content[:100]}...\n")
                f.write(f"  Metadata: {result.documents[0].metadata}\n\n")
            
            if result.has_more:
                f.write(f"Pagination test:\n")
                f.write(f"  Next page returned {len(next_result.documents)} documents with {next_result.total_tokens} tokens\n")
                f.write(f"  Has more: {next_result.has_more}, Next doc index: {next_result.next_doc_index}\n\n")
        
        f.write(f"Process docs result:\n")
        f.write(f"  Documents: {len(result.documents)}\n")
        f.write(f"  Total tokens: {result.total_tokens}\n")
        f.write(f"  Has more: {result.has_more}\n")
        f.write(f"  Next doc index: {result.next_doc_index}\n")
        f.write(f"  Total docs: {result.total_docs}\n\n")
        
        if testing_mode and result.has_more:
            f.write(f"Pagination test (mock data):\n")
            f.write(f"  Documents: {len(next_result.documents)}\n")
            f.write(f"  Total tokens: {next_result.total_tokens}\n")
            f.write(f"  Has more: {next_result.has_more}\n\n")
            
        f.write("Test completed successfully")
    
    print(f"Test results saved to: {os.path.join(output_dir, 'token_limiting_results.txt')}")

if __name__ == "__main__":
    args = parse_args()
    setup_debug_logging(args.debugLog)
    test_token_limiting(args.outputDir, args.testing)
