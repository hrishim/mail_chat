#!/usr/bin/env python3
import os
import json
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import BaseMessage
from email_searcher import EmailSearcher, SearchConfig, Document
from utils import log_debug, args, log_error

class QARetriever:
    """Base class for email question-answering retrievers."""
    
    def __init__(self, user_name: str, user_email: str, debug_log: bool = False):
        """Initialize the QA retriever.
        
        Args:
            user_name: Name of the user for context
            user_email: Email of the user for context
            debug_log: Whether to enable debug logging
        """
        self.user_name = user_name
        self.user_email = user_email
        self.debug_log = debug_log
        
        # Initialize email searcher
        self.email_searcher = EmailSearcher(debug_log=debug_log)
        
        # Initialize LLM
        self.llm = ChatNVIDIA(
            model="meta/llama3-8b-instruct",
            api_key=os.getenv("NGC_API_KEY"),
            api_url="http://0.0.0.0:8000/v1/completions"
        )
        
        # Base system prompt
        self._base_system_prompt = f"""You are an AI assistant helping {user_name} ({user_email}) search through their emails.
        You should provide direct, concise answers based on the email content provided.
        If you cannot find the answer in the emails, say so clearly.
        
        Current conversation context will be provided in chat_history if available.
        """
    
    def get_context(self, question: str, rerank_method: str = "Cosine Similarity", 
                   num_docs: int = 4, rerank_multiplier: int = 3) -> str:
        """Get relevant email context for the question.
        
        Args:
            question: The user's question
            rerank_method: Method to use for reranking results
            num_docs: Number of documents to return
            rerank_multiplier: Multiplier for initial retrieval before reranking
            
        Returns:
            str: Combined relevant email content
        """
        config = SearchConfig(
            num_docs=num_docs,
            rerank_multiplier=rerank_multiplier,
            rerank_method=rerank_method,
            return_full_threads=True
        )
        
        # Get relevant documents
        docs = self.email_searcher.semantic_search(question, config)
        
        # Combine document contents, but limit to ~4000 tokens (rough estimate of 4 chars per token)
        # This leaves room for the prompt, chat history, and completion
        max_chars = 16000  # ~4000 tokens
        total_chars = 0
        selected_docs = []
        
        for doc in docs:
            content_chars = len(doc.page_content)
            
            if total_chars + content_chars > max_chars:
                # If this doc would exceed limit, stop here
                break
                
            selected_docs.append(doc.page_content)
            total_chars += content_chars
        
        if self.debug_log:
            log_debug(f"Selected {len(selected_docs)} documents with total {total_chars} chars")
            for i, doc in enumerate(docs[:num_docs], 1):
                log_debug(f"\nDocument {i}:")
                log_debug(f"  Similarity score: {doc.metadata.get('similarity_score', 0):.4f}")
                log_debug(f"  Word count: {len(doc.page_content.split())}")
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                log_debug(f"  Content preview: {preview}")
            
            # Show full content of selected documents
            for i, doc in enumerate(docs[:num_docs], 1):
                log_debug(f"\n--- Email {i} contents ---")
                log_debug(doc.page_content)
                log_debug(f"--- Email {i} end ---\n")
            
        return "\n\n".join(selected_docs)
    
    def simple_qa(self, question: str, chat_history: Optional[List[BaseMessage]] = None,
                 rerank_method: str = "Cosine Similarity",
                 num_docs: int = 4, rerank_multiplier: int = 3) -> str:
        """Simple question-answering method that uses direct RAG with chat history.
        
        Args:
            question: The question to answer
            chat_history: Optional list of chat messages for context
            rerank_method: Method to use for reranking results
            num_docs: Number of documents to return
            rerank_multiplier: Multiplier for initial retrieval before reranking
            
        Returns:
            str: The answer to the question
        """
        def print_llm_input(x):
            if args.debugLog:
                log_debug("LLM_INPUT: %s", x)
                log_debug("LLM_INPUT type: %s", type(x))
                # If it's a list of messages, print each one's role and content
                if isinstance(x, list):
                    log_debug("Detailed message structure:")
                    for msg in x:
                        log_debug("Role: %s", msg.type)
                        log_debug("Content: %s\n", msg.content)
                    # Show what actually gets sent to the API
                    messages = [{"role": msg.type, "content": msg.content} for msg in x]
                    log_debug("Actual API request format:")
                    log_debug(json.dumps({
                        "model": "meta/llama3-8b-instruct",
                        "messages": messages
                    }, indent=2))
            return x

        def print_llm_output(x):
            if args.debugLog:
                log_debug("LLM_RAW: %s", x)
                log_debug("LLM_RAW type: %s", type(x))
            return x
            
        # Create the chain
        qa_chain = (
            RunnablePassthrough.assign(
                context=lambda input_dict: self.get_context(
                    input_dict["question"], 
                    input_dict.get("rerank_method", "Cosine Similarity"),
                    input_dict.get("num_docs", 4),
                    input_dict.get("rerank_multiplier", 3)
                )
            )
            | ChatPromptTemplate.from_messages([
                ("system", self._base_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])
            | (lambda x: (print_llm_input(x), x)[1])  # Print input to LLM
            | self.llm
            | (lambda x: (print_llm_output(x), x)[1])  # Print output from LLM
            | StrOutputParser()
        )
        
        # Get response
        return qa_chain.invoke({
            "question": question,
            "chat_history": chat_history or [],
            "rerank_method": rerank_method,
            "num_docs": num_docs,
            "rerank_multiplier": rerank_multiplier
        })

    def multi_query(self, question: str, rerank_method: str = "Cosine Similarity", num_docs: int = 4, rerank_multiplier: int = 3) -> str:
        """Generate multiple variations of the input question and retrieve relevant context."""
        if self.debug_log:
            log_debug(f"\nGenerating query variations for: {question}")

        # Generate query variations using LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generates search query variations. Given an input question, generate 3 alternative phrasings that would help find relevant information. Each variation should be semantically similar but use different keywords and phrasings. Return ONLY the variations, one per line, with no additional text or explanations."),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        variations_text = chain.invoke({"question": question})
        
        # Split variations into list and add original question
        variations = [q.strip() for q in variations_text.strip().split('\n') if q.strip()]
        if question not in variations:
            variations.insert(0, question)
            
        if self.debug_log:
            log_debug("Generated queries:")
            for i, q in enumerate(variations):
                log_debug(f"{i}. {q}")
            
        # Get unique documents from all variations
        all_docs = []
        seen_thread_ids = set()
        
        for query in variations:
            if not query:  # Skip empty queries
                continue
                
            try:
                # Get documents for this variation
                config = SearchConfig(
                    num_docs=num_docs,
                    rerank_multiplier=rerank_multiplier,
                    rerank_method=rerank_method,
                    return_full_threads=True
                )
                docs = self.email_searcher.semantic_search(query, config)
                
                # Only add documents from threads we haven't seen
                for doc in docs:
                    thread_id = doc.metadata.get('thread_id')
                    if thread_id and thread_id not in seen_thread_ids:
                        seen_thread_ids.add(thread_id)
                        all_docs.append(doc)
                        
            except Exception as e:
                if self.debug_log:
                    log_error(f"Error processing variation '{query}': {str(e)}")
                continue
                
        if not all_docs:
            return "I couldn't find any relevant emails matching your question."
            
        # Combine documents while respecting token limit (~4000 tokens, ~4 chars per token)
        max_chars = 16000
        total_chars = 0
        selected_docs = []
        
        for doc in all_docs:
            # Skip empty documents
            if not doc.page_content or not doc.page_content.strip():
                continue
                
            content = doc.page_content.strip()
            content_chars = len(content)
            
            if total_chars + content_chars > max_chars:
                break
                
            if self.debug_log:
                log_debug(f"Adding document with thread_id {doc.metadata.get('thread_id')}")
                log_debug("Content:\n%s", content)
                
            selected_docs.append(content)
            total_chars += content_chars
            
        if self.debug_log:
            log_debug(f"Selected {len(selected_docs)} documents with total {total_chars} chars")
            
        if not selected_docs:
            return "I couldn't find any relevant emails matching your question."
            
        # Combine selected documents
        context = "\n\n".join(selected_docs)
        
        # Generate final response using combined context
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions about emails. Use the provided email content to answer the question. If you cannot find a relevant answer in the emails, say so."),
            ("human", "Email content:\n{context}\n\nQuestion: {question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        
        return response
