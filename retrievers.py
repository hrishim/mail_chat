#!/usr/bin/env python3
import os
import json
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import BaseMessage
from email_searcher import EmailSearcher, SearchConfig
from utils import log_debug, args

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
            content = doc.page_content
            content_chars = len(content)
            
            if total_chars + content_chars > max_chars:
                # If this doc would exceed limit, stop here
                break
                
            selected_docs.append(content)
            total_chars += content_chars
        
        if self.debug_log:
            log_debug("Selected %d documents with total %d chars", len(selected_docs), total_chars)
            
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
