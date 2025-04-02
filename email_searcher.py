#!/usr/bin/env python3
import os
import time
import numpy as np
from typing import List, Dict, Optional
from langchain_core.documents import Document
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from utils import log_debug

@dataclass
class SearchConfig:
    """Configuration for email search."""
    num_docs: int = 10
    rerank_multiplier: int = 3
    rerank_method: str = "Cosine Similarity"
    return_full_threads: bool = True

class EmailSearcher:
    def __init__(self, debug_log: bool = False):
        """Initialize EmailSearcher.
        
        Args:
            debug_log: Whether to enable debug logging
        """
        self.debug_log = debug_log
        
        # Load environment variables
        load_dotenv()
        
        # Initialize embeddings
        ngc_key = os.getenv("NGC_API_KEY")
        if not ngc_key:
            raise ValueError("NGC_API_KEY environment variable is required")
        
        self.embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key=ngc_key
        )
        
        # Load vector store
        vectordb_path = os.getenv("VECTOR_DB", "./mail_vectordb")
        self.vectorstore = FAISS.load_local(
            vectordb_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def cosine_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using cosine similarity."""
        if self.debug_log:
            log_debug("\nReranking with cosine similarity:")
            log_debug(f"  Query: {query}")
            log_debug(f"  Input documents: {len(docs)}")
            log_debug(f"  Target documents: {k}")
            start_time = time.perf_counter()
            
        # Get embeddings
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in docs]
        
        # Calculate similarities
        similarities = [
            np.dot(query_embedding, doc_embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            for doc_embedding in doc_embeddings
        ]
        
        # Sort by similarity
        reranked_docs = [doc for _, doc in sorted(
            zip(similarities, docs),
            key=lambda x: x[0],
            reverse=True
        )][:k]
        
        if self.debug_log:
            rerank_time = time.perf_counter() - start_time
            log_debug("\nReranking results:")
            log_debug(f"  Reranking time: {rerank_time:.3f} seconds")
            for i, (doc, sim) in enumerate(zip(reranked_docs, sorted(similarities, reverse=True)[:k])):
                log_debug(f"\nDocument {i+1}:")
                log_debug(f"  Similarity score: {sim:.4f}")
                log_debug(f"  Word count: {len(doc.page_content.split())}")
                log_debug(f"  Content preview: {doc.page_content[:200]}...")
        
        return reranked_docs

    def get_full_thread(self, thread_id: str) -> Optional[Document]:
        """Get the complete email thread for a given thread ID."""
        if self.debug_log:
            log_debug(f"\nGetting full thread for ID: {thread_id}")
            start_time = time.perf_counter()
            
        # Get all chunks belonging to this thread
        all_docs = []
        for stored_doc in self.vectorstore.docstore._dict.values():
            if stored_doc.metadata.get('thread_id') == thread_id:
                all_docs.append(stored_doc)
        
        if not all_docs:
            if self.debug_log:
                log_debug(f"  No documents found for thread ID: {thread_id}")
            return None
            
        # Sort chunks by their position in the thread
        all_docs.sort(key=lambda x: x.metadata.get('chunk_index', 0))
        
        # Combine all chunks to reconstruct the full thread
        full_thread_content = "\n".join(d.page_content for d in all_docs)
        
        # Create a new document with the full thread content
        thread_doc = Document(
            page_content=full_thread_content,
            metadata={
                'thread_id': thread_id,
                'total_chunks': len(all_docs),
                'is_full_thread': True
            }
        )
        
        if self.debug_log:
            thread_time = time.perf_counter() - start_time
            log_debug(f"  Retrieved thread with {len(all_docs)} chunks")
            log_debug(f"  Thread retrieval time: {thread_time:.3f} seconds")
            
        return thread_doc

    def semantic_search(self, query: str, config: SearchConfig = None) -> List[Document]:
        """
        Search for relevant email documents using semantic similarity via FAISS.
        
        Args:
            query: The search query
            config: Search configuration. If None, default config will be used.
            
        Returns:
            List of relevant documents, either as chunks or full threads based on config.
        """
        if config is None:
            config = SearchConfig()
            
        if self.debug_log:
            log_debug(f"\nSemantic searching for: {query}")
            log_debug(f"  Config: num_docs={config.num_docs}, rerank={config.rerank_method}, full_threads={config.return_full_threads}")
            start_time = time.perf_counter()
        
        # Get initial matching chunks
        initial_k = config.num_docs if config.rerank_method == "No Reranking" else config.num_docs * config.rerank_multiplier
        initial_docs = self.vectorstore.similarity_search(query, k=initial_k)
        
        if self.debug_log:
            retrieval_time = time.perf_counter() - start_time
            log_debug(f"  Retrieved {len(initial_docs)} initial documents")
            log_debug(f"  Initial retrieval time: {retrieval_time:.3f} seconds")
            if config.rerank_method == "No Reranking":
                log_debug("  Skipping reranking")
        
        # Rerank chunks if selected
        if config.rerank_method != "No Reranking":
            initial_docs = self.cosine_rerank(query, initial_docs, config.num_docs)
        
        # Return chunks if full threads not requested
        if not config.return_full_threads:
            return initial_docs
        
        # Get complete threads for matched chunks
        if self.debug_log:
            log_debug("\nGetting full threads for matched chunks:")
            thread_start_time = time.perf_counter()
            
        thread_docs = []
        seen_threads = set()
        
        for doc in initial_docs:
            thread_id = doc.metadata.get('thread_id')
            if not thread_id or thread_id in seen_threads:
                continue
                
            seen_threads.add(thread_id)
            thread_doc = self.get_full_thread(thread_id)
            if thread_doc:
                thread_docs.append(thread_doc)
        
        if self.debug_log:
            thread_time = time.perf_counter() - thread_start_time
            total_time = time.perf_counter() - start_time
            log_debug(f"  Retrieved {len(thread_docs)} full threads")
            log_debug(f"  Thread retrieval time: {thread_time:.3f} seconds")
            log_debug(f"  Total search time: {total_time:.3f} seconds")
        
        return thread_docs
