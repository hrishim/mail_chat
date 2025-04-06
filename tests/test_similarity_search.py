#!/usr/bin/env python3
import os
import sys
import json
import re
import argparse
import types
from dotenv import load_dotenv

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from langchain_core.documents import Document
from email_searcher import EmailSearcher, SearchConfig, SearchResult
from utils import log_debug, setup_debug_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test similarity search functionality")
    parser.add_argument("--debugLog", type=str, help="Path to debug log file. If not specified, debug logging will be disabled.")
    parser.add_argument("--outputDir", type=str, default="test_results", help="Directory to save test results")
    parser.add_argument("--testing", action="store_true", help="Run in testing mode without vector database")
    return parser.parse_args()

def test_similarity_search(query: str, num_docs: int = 10, rerank_multiplier: int = 3, 
                          rerank_method: str = "Cosine Similarity", max_total_tokens: int = 3000,
                          output_dir: str = "test_results", testing_mode: bool = False):
    """Test similarity search with different parameters and log results."""
    log_debug(f"Testing similarity search with query: {query}")
    log_debug(f"Parameters: num_docs={num_docs}, rerank_multiplier={rerank_multiplier}, rerank_method={rerank_method}, max_total_tokens={max_total_tokens}")
    
    # Ensure environment variables are loaded from the correct .env file
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(dotenv_path)
    
    # Log vector database path
    vector_db_path = os.getenv("VECTOR_DB", "./mail_vectordb")
    log_debug(f"Vector database path from .env: {vector_db_path}")
    log_debug(f"Dotenv path: {dotenv_path}")
    
    # Initialize searcher with debug logging
    searcher = EmailSearcher(debug_log=True, testing=testing_mode)
    
    # If in testing mode, we need to mock the semantic_search method
    if testing_mode:
        # Create test documents based on the query
        mock_docs = create_mock_documents_for_query(query)
        
        # Mock the semantic_search method
        def mock_semantic_search(self, query: str, config: SearchConfig = None) -> SearchResult:
            log_debug(f"Mock semantic search called with query: {query}")
            if config is None:
                config = SearchConfig()
                
            # Add similarity scores based on simple keyword matching
            for doc in mock_docs:
                # Simple scoring based on word overlap
                query_words = set(query.lower().split())
                content_words = set(doc.page_content.lower().split())
                overlap = len(query_words.intersection(content_words))
                score = overlap / max(len(query_words), 1)  # Avoid division by zero
                
                # Boost score if there's a direct match to important terms
                if "fasstag" in query.lower() and "fasstag" in doc.page_content.lower():
                    score += 0.3
                if "toll" in query.lower() and "toll" in doc.page_content.lower():
                    score += 0.2
                    
                doc.metadata["similarity_score"] = min(0.95, 0.7 + score)  # Cap at 0.95
                
            # Simulate token limiting
            docs_to_return = mock_docs[:config.num_docs]
            total_tokens = sum(self.estimate_tokens(doc.page_content) for doc in docs_to_return)
            
            return SearchResult(
                documents=docs_to_return,
                total_tokens=total_tokens,
                has_more=len(docs_to_return) < len(mock_docs),
                next_doc_index=len(docs_to_return) if len(docs_to_return) < len(mock_docs) else -1,
                total_docs=len(mock_docs)
            )
        searcher.semantic_search = types.MethodType(mock_semantic_search, searcher)
    
    # Create search config
    config = SearchConfig(
        num_docs=num_docs,
        rerank_multiplier=rerank_multiplier,
        rerank_method=rerank_method,
        max_total_tokens=max_total_tokens
    )
    
    # Perform search
    result = searcher.semantic_search(query, config)
    
    # Process results
    results = []
    for i, doc in enumerate(result.documents):
        results.append({
            "index": i,
            "content": doc.page_content,
            "word_count": len(doc.page_content.split()),
            "token_estimate": searcher.estimate_tokens(doc.page_content),
            "metadata": doc.metadata
        })
    
    # Create safe filename from query
    safe_query = re.sub(r'[^a-zA-Z0-9_\s]', '', query)  # Remove special characters
    safe_query = safe_query.replace(' ', '_')  # Replace spaces with underscores
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"similarity_search_results_{rerank_method.lower().replace(' ', '_')}_{safe_query[:30]}.json")
    
    # Add pagination info to output
    output_data = {
        "query": query,
        "num_docs": num_docs,
        "rerank_method": rerank_method,
        "total_tokens": result.total_tokens,
        "has_more": result.has_more,
        "next_doc_index": result.next_doc_index,
        "total_docs": result.total_docs,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary to console
    print("\nSummary of retrieved documents:")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Has more: {result.has_more}")
    print(f"Total docs: {result.total_docs}")
    
    # Print document details
    for i, doc in enumerate(result.documents):
        print(f"\nDocument {i+1}:")
        print(f"Word count: {len(doc.page_content.split())}")
        print(f"Token estimate: {searcher.estimate_tokens(doc.page_content)}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:200]}...")

def create_mock_documents_for_query(query: str) -> List[Document]:
    """Create mock documents based on the query for testing purposes."""
    # Default mock documents
    mock_docs = [
        Document(
            page_content="This is a test email about FASTag Toll payment on 2 Dec 2024",
            metadata={
                "thread_id": "test_thread_1",
                "full_content_id": "test_thread_1_0",
                "sender": "toll@example.com",
                "to": "recipient@example.com",
                "subject": "FASTag Payment Confirmation",
                "date": "2024-12-02 12:00:00+0000",
                "is_chunk": True
            }
        ),
        Document(
            page_content="Another test email about highway tolls and payments",
            metadata={
                "thread_id": "test_thread_2",
                "full_content_id": "test_thread_2_0",
                "sender": "highway@example.com",
                "to": "recipient@example.com",
                "subject": "Highway Toll Receipt",
                "date": "2024-12-01 10:00:00+0000",
                "is_chunk": True
            }
        )
    ]
    
    # Add query-specific mock documents if needed
    if "filter coffee" in query.lower():
        mock_docs.append(
            Document(
                page_content="Your order from Filter Coffee has been delivered",
                metadata={
                    "thread_id": "test_thread_3",
                    "full_content_id": "test_thread_3_0",
                    "sender": "orders@filtercoffee.com",
                    "to": "recipient@example.com",
                    "subject": "Order Delivered",
                    "date": "2024-11-15 14:30:00+0000",
                    "is_chunk": True
                }
            )
        )
    
    if "email" in query.lower():
        if "prabha" in query.lower() or "sundar" in query.lower():
            mock_docs.append(
                Document(
                    page_content="Email from Prabha Sundar (prabha.sundar@example.com)",
                    metadata={
                        "thread_id": "test_thread_4",
                        "full_content_id": "test_thread_4_0",
                        "sender": "prabha.sundar@example.com",
                        "to": "recipient@example.com",
                        "subject": "Meeting Schedule",
                        "date": "2024-11-10 09:15:00+0000",
                        "is_chunk": True
                    }
                )
            )
        elif "seshambal" in query.lower():
            mock_docs.append(
                Document(
                    page_content="Email from Seshambal (seshambal@example.com)",
                    metadata={
                        "thread_id": "test_thread_5",
                        "full_content_id": "test_thread_5_0",
                        "sender": "seshambal@example.com",
                        "to": "recipient@example.com",
                        "subject": "Family Updates",
                        "date": "2024-11-05 16:45:00+0000",
                        "is_chunk": True
                    }
                )
            )
        elif "sathyavasu" in query.lower():
            mock_docs.append(
                Document(
                    page_content="Email from Sathyavasu (sathyavasu@example.com)",
                    metadata={
                        "thread_id": "test_thread_6",
                        "full_content_id": "test_thread_6_0",
                        "sender": "sathyavasu@example.com",
                        "to": "recipient@example.com",
                        "subject": "Weekend Plans",
                        "date": "2024-11-01 11:30:00+0000",
                        "is_chunk": True
                    }
                )
            )
        elif "axis" in query.lower() or "bank" in query.lower():
            mock_docs.append(
                Document(
                    page_content="Email from Axis Bank (customer.service@axisbank.com)",
                    metadata={
                        "thread_id": "test_thread_7",
                        "full_content_id": "test_thread_7_0",
                        "sender": "customer.service@axisbank.com",
                        "to": "recipient@example.com",
                        "subject": "Account Statement",
                        "date": "2024-10-28 08:00:00+0000",
                        "is_chunk": True
                    }
                )
            )
    
    return mock_docs

if __name__ == "__main__":
    args = parse_args()
    setup_debug_logging(args.debugLog)
    
    # Example queries to test
    test_queries = [
        "Where did I pay FASTag Toll on 2 Dec 2024?",  
        #"Have I ordered from Filter Coffee?",  
        #"What is Prabha Sundar's email?", # Fails to get context
        #"What is Seshambal's email?",
        #"What is Sathyavasu's email",
        #"What is Axis Bank's email?"
    ]
    
    for query in test_queries:
        test_similarity_search(query, num_docs=50, rerank_multiplier=3, rerank_method="Cosine Similarity", 
                              max_total_tokens=3000, output_dir=args.outputDir, testing_mode=args.testing)
        print("\n" + "="*80 + "\n")  # Separator between queries
