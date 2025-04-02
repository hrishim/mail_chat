#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from utils import log_debug, args
import numpy as np
from typing import List, Dict, Optional
from langchain_core.documents import Document
from dataclasses import dataclass

@dataclass
class SearchConfig:
    """Configuration for email search."""
    num_docs: int = 10
    rerank_multiplier: int = 3
    rerank_method: str = "Cosine Similarity"
    return_full_threads: bool = True

class EmailSearcher:
    def __init__(self):
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
        if args.debugLog:
            log_debug("\nReranking documents using cosine similarity...")
            
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
        
        if args.debugLog:
            log_debug(f"Reranked {len(docs)} documents to top {k}")
            
        return reranked_docs

    def get_full_thread(self, thread_id: str) -> Optional[Document]:
        """Get all chunks for a thread and combine them into a single document."""
        thread_chunks = []
        
        # Get all documents and filter by thread_id
        all_docs = []
        for stored_doc in self.vectorstore.docstore._dict.values():
            if stored_doc.metadata.get('thread_id') == thread_id:
                all_docs.append(stored_doc)
        
        if not all_docs:
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
            
        if args.debugLog:
            log_debug(f"\nSemantic searching for: {query}")
            log_debug(f"Config: num_docs={config.num_docs}, rerank={config.rerank_method}, full_threads={config.return_full_threads}")
        
        # Get initial matching chunks
        initial_k = config.num_docs if config.rerank_method == "No Reranking" else config.num_docs * config.rerank_multiplier
        initial_docs = self.vectorstore.similarity_search(query, k=initial_k)
        
        # Rerank if requested
        if config.rerank_method != "No Reranking":
            initial_docs = self.cosine_rerank(query, initial_docs, config.num_docs)
            
        if not config.return_full_threads:
            return initial_docs
            
        # Get complete threads for the ranked/reranked chunks
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
                
        return thread_docs
