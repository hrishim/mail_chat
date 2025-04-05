#!/usr/bin/env python3
import os
import time
import numpy as np
from typing import List, Dict, Optional, NamedTuple, Tuple
from langchain_core.documents import Document
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import log_debug

@dataclass
class SearchConfig:
    """Configuration for email search."""
    num_docs: int = 10
    rerank_multiplier: int = 3
    rerank_method: str = "Cosine Similarity"
    return_full_threads: bool = True
    max_total_tokens: int = 3000  # Default conservative limit for most LLMs
    start_index: int = 0  # For pagination
    chunk_filter: Optional[dict] = None  # For filtering by chunk metadata

class SearchResult(NamedTuple):
    """Container for search results with pagination info."""
    documents: List[Document]  # Documents that fit within token limit
    total_tokens: int          # Total tokens in returned documents
    has_more: bool             # Whether there are more documents available
    next_doc_index: int        # Index to start from for next page
    total_docs: int            # Total number of matching documents

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
    def __init__(self, debug_log: bool = False, testing: bool = False):
        """Initialize EmailSearcher with NVIDIA embeddings and FAISS vector store.
        
        Args:
            debug_log: Whether to enable debug logging for detailed search process information
            testing: If True, skip loading the vector store (for testing purposes)
            
        Raises:
            ValueError: If NGC_API_KEY environment variable is not set
            
        Note:
            Currently uses NVIDIA's NV-Embed-QA model for embeddings and FAISS for vector storage.
            The vector store path can be configured via the VECTOR_DB environment variable.
        """
        self.debug_log = debug_log
        
        # Load environment variables from the project root directory
        project_root = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.join(project_root, '.env')
        load_dotenv(dotenv_path)
        
        if self.debug_log:
            log_debug(f"Loading environment variables from: {dotenv_path}")
        
        # Initialize embeddings
        ngc_key = os.getenv("NGC_API_KEY")
        if not ngc_key:
            raise ValueError("NGC_API_KEY environment variable is required")
        
        self.embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key=ngc_key
        )
        
        # Skip vector store loading for testing
        if testing:
            if self.debug_log:
                log_debug("Testing mode: Skipping vector store loading")
            self.vectorstore = None
            return
            
        # Load vector store
        raw_vector_db = os.environ.get("VECTOR_DB")
        if self.debug_log:
            log_debug(f"Raw VECTOR_DB value: {raw_vector_db!r}")
            
        vectordb_path = os.getenv("VECTOR_DB", "./mail_vectordb")
        if self.debug_log:
            log_debug(f"Vector database path from .env: {vectordb_path!r}")
            
        # Try to fix path if it has quotes
        if vectordb_path and vectordb_path.startswith('"') and vectordb_path.endswith('"'):
            vectordb_path = vectordb_path[1:-1]
            if self.debug_log:
                log_debug(f"Removed quotes from path: {vectordb_path!r}")
            
        if not os.path.exists(vectordb_path):
            raise ValueError(f"Vector store path '{vectordb_path}' does not exist")
            
        self.vectorstore = FAISS.load_local(
            vectordb_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text using character count.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count (rough approximation)
            
        Note:
            This is a conservative estimate. Actual token count may be lower.
            Uses average of 4 characters per token as a rough approximation.
        """
        return len(text) // 4  # Conservative estimate of 4 chars per token

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
            log_debug(f"Reranking {len(docs)} documents using cosine similarity")
            start_time = time.perf_counter()
            
        if not docs:
            return []
            
        # Get embeddings for query and documents
        query_embedding = self.embeddings.embed_query(query)
        
        # Process documents in batches to avoid memory issues
        doc_embeddings = []
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            doc_embeddings.extend(batch_embeddings)
            
        # Calculate cosine similarity
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, similarity))
            
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k documents
        top_indices = [idx for idx, _ in similarities[:k]]
        top_docs = [docs[idx] for idx in top_indices]
        
        # Add similarity scores to metadata
        for i, doc in enumerate(top_docs):
            doc.metadata["similarity_score"] = float(similarities[i][1])
            
        if self.debug_log:
            rerank_time = time.perf_counter() - start_time
            log_debug(f"  Reranking time: {rerank_time:.3f} seconds")
            log_debug(f"  Top similarity scores: {[s for _, s in similarities[:3]]}")
            
        return top_docs

    def get_full_thread(self, thread_id: str, message_index: int) -> Optional[Document]:
        """Get full thread document by thread_id and message_index.
        
        Args:
            thread_id: Unique identifier for the email thread
            message_index: Index of the message in the thread
            
        Returns:
            Optional[Document]: A document containing the complete thread if found,
                          None if no document found for the thread_id and message_index
        """
        if self.debug_log:
            log_debug(f"Getting full thread for ID: {thread_id}, message index: {message_index}")
            start_time = time.perf_counter()
        
        # Construct the full document ID
        full_doc_id = f"full_{thread_id}_{message_index}"
        
        # Retrieve the document directly from the docstore
        thread_doc = self.vectorstore.docstore._dict.get(full_doc_id)
        
        if self.debug_log:
            thread_time = time.perf_counter() - start_time
            if thread_doc:
                log_debug(f"  Found thread document with {len(thread_doc.page_content)} characters")
            else:
                log_debug(f"  No thread document found for ID: {full_doc_id}")
            log_debug(f"  Thread retrieval time: {thread_time:.3f} seconds")
            
        return thread_doc

    def _process_docs_with_limit(
        self, 
        docs: List[Document], 
        max_tokens: int,
        start_index: int = 0
    ) -> SearchResult:
        """Process documents to fit within token limit with pagination support.
        
        Args:
            docs: List of documents to process
            max_tokens: Maximum total tokens allowed
            start_index: Starting index for pagination
            
        Returns:
            SearchResult containing documents that fit within token budget and pagination info
        """
        processed_docs = []
        total_tokens = 0
        current_index = start_index
        
        for i, doc in enumerate(docs[start_index:], start=start_index):
            current_index = i
            doc_tokens = self.estimate_tokens(doc.page_content)
            
            if doc_tokens > max_tokens:
                # If a single document is too large, split it
                splits = self._split_document(doc, max_tokens)
                for split_doc in splits:
                    split_tokens = self.estimate_tokens(split_doc.page_content)
                    if total_tokens + split_tokens <= max_tokens:
                        processed_docs.append(split_doc)
                        total_tokens += split_tokens
                    else:
                        break
            else:
                if total_tokens + doc_tokens <= max_tokens:
                    processed_docs.append(doc)
                    total_tokens += doc_tokens
                else:
                    break
        
        has_more = current_index < len(docs) - 1
        next_doc_index = current_index + 1 if has_more else -1
        
        return SearchResult(
            documents=processed_docs,
            total_tokens=total_tokens,
            has_more=has_more,
            next_doc_index=next_doc_index,
            total_docs=len(docs)
        )

    def _split_document(self, doc: Document, max_tokens: int) -> List[Document]:
        """Split a document into smaller parts that fit within token limit.
        
        Args:
            doc: Document to split
            max_tokens: Maximum tokens per split
            
        Returns:
            List of split documents with updated metadata
        """
        # Use the same text splitter settings as in data_prep
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 4,  # Convert tokens to approximate chars
            chunk_overlap=40,
            length_function=len,
            is_separator_regex=False
        )
        
        splits = text_splitter.split_text(doc.page_content)
        split_docs = []
        
        for i, split in enumerate(splits):
            # Create new metadata for the split
            split_metadata = doc.metadata.copy()
            split_metadata.update({
                'is_split': True,
                'split_index': i,
                'total_splits': len(splits),
                'original_content_id': doc.metadata.get('full_content_id', '')
            })
            
            split_docs.append(Document(
                page_content=split,
                metadata=split_metadata
            ))
        
        return split_docs

    def semantic_search(self, query: str, config: SearchConfig = None) -> SearchResult:
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
            SearchResult: Container with relevant documents and pagination info
            
        Note:
            The search process is highly configurable through SearchConfig:
            - num_docs: Number of results to return
            - rerank_multiplier: Controls the quality vs speed tradeoff for reranking
            - rerank_method: Whether to apply cosine similarity reranking
            - return_full_threads: Whether to return complete email threads
            - max_total_tokens: Maximum total tokens to return
            - start_index: Starting index for pagination
        """
        if config is None:
            config = SearchConfig()
            
        if self.debug_log:
            log_debug(f"\nSemantic searching for: {query}")
            log_debug(f"  Config: num_docs={config.num_docs}, rerank={config.rerank_method}, "
                     f"full_threads={config.return_full_threads}, "
                     f"max_tokens={config.max_total_tokens}, "
                     f"start_index={config.start_index}")
            start_time = time.perf_counter()
        
        # Get initial matching chunks
        initial_k = config.num_docs if config.rerank_method == "No Reranking" else config.num_docs * config.rerank_multiplier
        initial_docs = self.vectorstore.similarity_search(query, k=initial_k)
        
        # Filter for chunk documents only
        initial_docs = [doc for doc in initial_docs if doc.metadata.get('is_chunk', False)]
        
        if self.debug_log:
            retrieval_time = time.perf_counter() - start_time
            log_debug(f"  Retrieved {len(initial_docs)} initial chunk documents")
            log_debug(f"  Initial retrieval time: {retrieval_time:.3f} seconds")
            if config.rerank_method == "No Reranking":
                log_debug("  Skipping reranking")
        
        # Rerank chunks if selected
        if config.rerank_method != "No Reranking":
            initial_docs = self.cosine_rerank(query, initial_docs, config.num_docs)
        
        # Return chunks if full threads not requested
        if not config.return_full_threads:
            return self._process_docs_with_limit(initial_docs, config.max_total_tokens, config.start_index)
        
        # Get complete threads for matched chunks
        if self.debug_log:
            log_debug("\nGetting full threads for matched chunks:")
            thread_start_time = time.perf_counter()
            
        thread_docs = []
        seen_content_ids = set()
        
        for doc in initial_docs:
            full_content_id = doc.metadata.get('full_content_id')
            if not full_content_id or full_content_id in seen_content_ids:
                continue
                
            seen_content_ids.add(full_content_id)
            
            try:
                thread_id, message_index = full_content_id.split('_')
                thread_doc = self.get_full_thread(thread_id, int(message_index))
                if thread_doc:
                    thread_docs.append(thread_doc)
            except ValueError:
                if self.debug_log:
                    log_debug(f"  Invalid full_content_id format: {full_content_id}")
                continue
        
        # Process thread documents with token limit
        result = self._process_docs_with_limit(thread_docs, config.max_total_tokens, config.start_index)
        
        if self.debug_log:
            thread_time = time.perf_counter() - thread_start_time
            total_time = time.perf_counter() - start_time
            log_debug(f"  Retrieved {len(result.documents)} of {result.total_docs} full threads")
            log_debug(f"  Total tokens: {result.total_tokens}")
            log_debug(f"  Has more: {result.has_more}")
            log_debug(f"  Thread retrieval time: {thread_time:.3f} seconds")
            log_debug(f"  Total search time: {total_time:.3f} seconds")
            
            # Log full emails after reranking
            log_debug("\nFull emails after reranking:")
            for i, doc in enumerate(result.documents, 1):
                log_debug("--- Email %d start ---", i)
                log_debug("Thread ID: %s", doc.metadata.get('thread_id'))
                log_debug("Content:\n%s", doc.page_content)
                log_debug("--- Email %d end ---\n", i)
        
        return result
