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
    """A versatile email search engine that retrieves relevant email content from vector databases.
    
    The EmailSearcher class provides semantic search capabilities over email content using
    vector similarity search. Currently implemented with FAISS as the primary vector store,
    but designed to be extensible for other vector databases like Chroma in the future.
    
    Key Features:
        - Semantic search using NVIDIA embeddings
        - Cosine similarity based reranking
        - Support for both chunk-level and full-thread retrieval
        - Configurable search parameters via SearchConfig
        
    The search process involves:
    1. Converting query to embeddings using NVIDIA's embedding model
    2. Performing initial vector similarity search in FAISS
    3. Optional reranking of results using cosine similarity
    4. Optional retrieval of complete email threads
    
    Future extensions can implement alternative:
        - Vector stores (e.g. Chroma, Pinecone)
        - Embedding models
        - Reranking methods
        - Search strategies
    """
    def __init__(self, debug_log: bool = False):
        """Initialize EmailSearcher with NVIDIA embeddings and FAISS vector store.
        
        Args:
            debug_log: Whether to enable debug logging for detailed search process information
            
        Raises:
            ValueError: If NGC_API_KEY environment variable is not set
            
        Note:
            Currently uses NVIDIA's NV-Embed-QA model for embeddings and FAISS for vector storage.
            The vector store path can be configured via the VECTOR_DB environment variable.
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
        if not os.path.exists(vectordb_path):
            raise ValueError(f"Vector store path '{vectordb_path}' does not exist")
            
        self.vectorstore = FAISS.load_local(
            vectordb_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def cosine_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using cosine similarity between query and document embeddings.
        
        This method provides a second level of refinement after the initial vector search.
        It recomputes embeddings for both query and documents to calculate more precise
        similarity scores.
        
        Args:
            query: The search query string
            docs: List of documents from initial vector search
            k: Number of top documents to return after reranking
            
        Returns:
            List[Document]: Top k documents sorted by cosine similarity to query
            
        Note:
            Performance scales with number of input documents as it requires computing
            embeddings for each document. Consider the rerank_multiplier in SearchConfig
            to balance between quality and speed.
        """
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
        """Get all messages from a thread in chronological order.
        
        Args:
            thread_id: Unique identifier for the email thread
            
        Returns:
            Optional[Document]: A document containing the complete thread if found,
                          None if no messages found for the thread_id
        """
        if self.debug_log:
            log_debug(f"\nGetting full thread for ID: {thread_id}")
            start_time = time.perf_counter()
            
        # Get all messages in this thread
        thread_messages = []
        for stored_doc in self.vectorstore.docstore._dict.values():
            if stored_doc.metadata.get('thread_id') == thread_id:
                thread_messages.append(stored_doc)
        
        if not thread_messages:
            if self.debug_log:
                log_debug(f"  No messages found for thread ID: {thread_id}")
            return None
            
        # Sort messages by their position in thread
        thread_messages.sort(key=lambda x: x.metadata.get('thread_position', 0))
        
        # Combine messages with headers
        thread_content = []
        for msg in thread_messages:
            # Only add headers if they exist
            headers = []
            if msg.metadata.get('sender'):
                headers.append(f"From: {msg.metadata['sender']}")
            if msg.metadata.get('to'):
                headers.append(f"To: {msg.metadata['to']}")
            if msg.metadata.get('subject'):
                headers.append(f"Subject: {msg.metadata['subject']}")
            if msg.metadata.get('date'):
                headers.append(f"Date: {msg.metadata['date']}")
            if msg.metadata.get('x_gmail_labels'):
                headers.append(f"Labels: {', '.join(msg.metadata['x_gmail_labels'])}")
            
            if headers:  # Only add headers if we have any
                thread_content.append("\n".join(headers))
                thread_content.append("")  # Empty line after headers
            
            # Add message content if it exists and isn't just whitespace
            content = msg.page_content.strip()
            if content:
                thread_content.append(content)
                thread_content.append("\n---Next Message in Thread---\n")
        
        # Remove the last separator if we added any content
        if thread_content and thread_content[-1].strip() == "---Next Message in Thread---":
            thread_content.pop()
            if thread_content and not thread_content[-1].strip():  # Remove trailing empty line
                thread_content.pop()
        
        # Create a new document with the full thread
        thread_doc = Document(
            page_content="\n".join(thread_content),
            metadata={
                **thread_messages[0].metadata,  # Use first message's metadata as base
                'thread_id': thread_id,
                'num_messages': len(thread_messages),
                'is_full_thread': True
            }
        )
        
        if self.debug_log:
            thread_time = time.perf_counter() - start_time
            log_debug(f"  Retrieved thread with {len(thread_messages)} messages")
            log_debug(f"  Thread retrieval time: {thread_time:.3f} seconds")
            
        return thread_doc

    def semantic_search(self, query: str, config: SearchConfig = None) -> List[Document]:
        """Search for relevant email content using semantic similarity.
        
        This is the main search method that orchestrates the complete search process:
        1. Initial semantic search using FAISS vector similarity
        2. Optional reranking using cosine similarity
        3. Optional reconstruction of full email threads
        
        Args:
            query: The search query string
            config: Search configuration parameters. If None, uses default config
                   See SearchConfig class for available parameters
            
        Returns:
            List[Document]: Relevant documents, either as chunks or full threads based on config
            
        Note:
            The search process is highly configurable through SearchConfig:
            - num_docs: Number of results to return
            - rerank_multiplier: Controls the quality vs speed tradeoff for reranking
            - rerank_method: Whether to apply cosine similarity reranking
            - return_full_threads: Whether to return complete email threads
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
            
            # Log full emails after reranking
            log_debug("\nFull emails after reranking:")
            for i, doc in enumerate(thread_docs, 1):
                log_debug("--- Email %d start ---", i)
                log_debug("Thread ID: %s", doc.metadata.get('thread_id'))
                log_debug("Content:\n%s", doc.page_content)
                log_debug("--- Email %d end ---\n", i)
        
        return thread_docs
