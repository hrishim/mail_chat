#!/usr/bin/env python3
import json
import re
from typing import List, Dict
from langchain_core.documents import Document
from email_searcher import EmailSearcher, SearchConfig
from utils import log_debug

def test_similarity_search(query: str, num_docs: int = 10, rerank_multiplier: int = 3, rerank_method: str = "Cosine Similarity"):
    """Test similarity search with different parameters and log results."""
    # Initialize searcher
    searcher = EmailSearcher()
    
    # Create search config
    config = SearchConfig(
        num_docs=num_docs,
        rerank_multiplier=rerank_multiplier,
        rerank_method=rerank_method,
        return_full_threads=True
    )
    
    # Get documents
    docs = searcher.semantic_search(query, config)
    
    # Process results
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,
            "word_count": len(doc.page_content.split()),
            "metadata": doc.metadata
        })
    
    # Create filename from query
    safe_query = re.sub(r'[^a-zA-Z0-9_\s]', '', query)  # Remove special characters
    safe_query = safe_query.replace(' ', '_')  # Replace spaces with underscores
    output_file = f"similarity_search_results_{rerank_method.lower().replace(' ', '_')}_{safe_query[:30]}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "query": query,
            "num_docs": num_docs,
            "rerank_method": rerank_method,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary to console
    print("\nSummary of retrieved documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Word count: {len(doc.page_content.split())}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:200]}...")

if __name__ == "__main__":
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
        test_similarity_search(query, num_docs=50, rerank_multiplier=3, rerank_method="Cosine Similarity")
        print("\n" + "="*80 + "\n")  # Separator between queries
