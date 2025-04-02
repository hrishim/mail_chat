import os
import json
import gradio as gr
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
import subprocess
import logging
import traceback
import threading
from container_manager import ContainerManager
from utils import log_debug, log_error, args
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.memory import ConversationBufferMemory
import numpy as np
import requests
import time
import inspect

# Load environment variables
load_dotenv()

def is_valid_vectordb(path: Path) -> bool:
    """Check if the given path is a valid FAISS vector database directory."""
    return path.is_dir() and (path / "index.faiss").exists() and (path / "index.pkl").exists()

class EmailChatBot:
    def __init__(self, vectordb_path: Optional[str] = None, user_name: Optional[str] = None, user_email: Optional[str] = None,
                 num_docs: int = 4, rerank_multiplier: int = 3, 
                 llm_container: str = "meta-llama3-8b-instruct",
                 reranker_container: str = "nv-rerankqa-mistral-4b"):
        # Use VECTOR_DB from env if vectordb_path not provided
        if vectordb_path is None:
            vectordb_path = os.getenv("VECTOR_DB", "./mail_vectordb")
        
        # Get user details from env if not provided
        self.user_name = user_name or os.getenv("USER_FULLNAME", "YOUR_NAME")
        self.user_email = user_email or os.getenv("USER_EMAIL", "YOUR_EMAIL")
        
        # Container names that can be changed
        self._llm_container = llm_container
        self._reranker_container = reranker_container
        
        # Log initial parameters
        if args.debugLog:
            log_debug("Initializing EmailChatBot:")
            log_debug(f"  Database folder: {vectordb_path}")
            log_debug(f"  User name: {self.user_name}")
            log_debug(f"  User email: {self.user_email}")
            log_debug(f"  Number of documents: {num_docs}")
            log_debug(f"  Reranking multiplier: {rerank_multiplier}")
            log_debug(f"  LLM container: {self._llm_container}")
            log_debug(f"  Reranker container: {self._reranker_container}")
        
        # Document retrieval settings
        self.num_docs = num_docs
        self.rerank_multiplier = rerank_multiplier
        
        self.vectordb_path = Path(vectordb_path)
        if not self.vectordb_path.exists():
            raise ValueError(f"Vector database directory '{vectordb_path}' does not exist")
        if not is_valid_vectordb(self.vectordb_path):
            raise ValueError(f"'{vectordb_path}' is not a valid FAISS vector database directory")
        
        self.chat_history: List[Dict[str, str]] = []
        self.llm_url = "http://0.0.0.0:8000/v1/completions"
        # Initialize container statuses using class method
        self.container_status = self.get_container_status()
        self.reranker_status = self.get_reranker_status()
        self.last_retrieved_docs = None  # Store last retrieved documents
        
        # Initialize embeddings
        ngc_key = os.getenv("NGC_API_KEY")
        if not ngc_key:
            raise ValueError("NGC_API_KEY environment variable is required")
        self.embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key=ngc_key
        )
        
        self.setup_components()

    def get_container_status(self) -> str:
        """Get the current status of the LLM container."""
        return ContainerManager.check_container_status(self._llm_container)

    def get_reranker_status(self) -> str:
        """Get the current status of the reranker container."""
        return ContainerManager.check_container_status(self._reranker_container, health_port=8001)

    @property
    def container_name(self):
        return self._llm_container

    @container_name.setter
    def container_name(self, name: str):
        self._llm_container = name
        # Update status when container name changes
        self.container_status = self.get_container_status()

    @property
    def reranker_container_name(self):
        return self._reranker_container

    @reranker_container_name.setter
    def reranker_container_name(self, name: str):
        self._reranker_container = name
        # Update status when container name changes
        self.reranker_status = self.get_reranker_status()

    def start_container(self) -> str:
        """Start the LLM container."""
        ngc_key = os.getenv('NGC_API_KEY')
        return ContainerManager.start_container(
            container_name=self.container_name,
            image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
            port=8000,
            ngc_key=ngc_key
        )

    def stop_container(self) -> str:
        """Stop the LLM container."""
        return ContainerManager.stop_container(self.container_name)

    def start_reranker(self) -> str:
        """Start the reranker container."""
        try:
            if self.get_reranker_status() != "stopped":
                return "Reranker is already running"
            
            # Get NGC API key from environment
            ngc_key = os.getenv("NGC_API_KEY")
            if not ngc_key:
                return "NGC_API_KEY environment variable is required"
            
            # Create cache directory if it doesn't exist
            nim_cache = os.path.expanduser("~/.cache/nim")
            os.makedirs(nim_cache, exist_ok=True)
            
            # Start the container
            subprocess.run([
                "docker", "run", "-d",
                "--name", self.reranker_container_name,
                "--gpus", "all",
                "--shm-size=16GB",
                "-e", f"NGC_API_KEY={ngc_key}",
                "-v", f"{nim_cache}:/opt/nim/.cache",
                "-u", str(os.getuid()),
                "-p", "8001:8000",
                "nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:1.0.2"
            ], check=True)
            
            return "Starting reranker container..."
        except subprocess.CalledProcessError as e:
            if args.debugLog:
                log_debug(f"Error starting reranker: {str(e)}")
            return f"Error starting reranker: {str(e)}"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Unexpected error starting reranker: {str(e)}")
            return f"Unexpected error starting reranker: {str(e)}"

    def stop_reranker(self) -> str:
        """Stop the reranker container."""
        try:
            if self.get_reranker_status() == "stopped":
                return "Reranker is not running"
            
            subprocess.run(["docker", "stop", self.reranker_container_name], check=True)
            subprocess.run(["docker", "rm", self.reranker_container_name], check=True)
            return "Reranker stopped successfully"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error stopping reranker: {str(e)}")
            return f"Error stopping reranker: {str(e)}"

    def mistral_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using NV-RerankQA-Mistral-4B-v3."""
        if args.debugLog:
            log_debug(f"\nReranking with Mistral reranker:")
            log_debug(f"  Query: {query}")
            log_debug(f"  Input documents: {len(docs)}")
            log_debug(f"  Target documents: {k}")
            start_time = time.perf_counter()
        
        # Prepare documents for reranking
        passages = [doc.page_content for doc in docs]
        
        # Rerank using Mistral
        try:
            scores = self.reranker.predict(
                query=query,
                passages=passages
            )
            
            # Sort by score (moved outside debug block)
            reranked_docs = [doc for _, doc in sorted(
                zip(scores, docs),
                key=lambda x: x[0],
                reverse=True
            )][:k]
            
            if args.debugLog:
                rerank_time = time.perf_counter() - start_time
                # Log reranking results
                log_debug("\nReranking results:")
                log_debug(f"  Reranking time: {rerank_time:.3f} seconds")
                for i, (doc, score) in enumerate(zip(reranked_docs, sorted(scores, reverse=True)[:k])):
                    log_debug(f"\nDocument {i+1}:")
                    log_debug(f"  Score: {score:.4f}")
                    log_debug(f"  Word count: {len(doc.page_content.split())}")
                    log_debug(f"  Content: {doc.page_content}")
            
            return reranked_docs
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error in Mistral reranking: {str(e)}")
                log_debug(f"Falling back to cosine similarity reranking")
            return self.cosine_rerank(query, docs, k)

    def cosine_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using cosine similarity."""
        if args.debugLog:
            log_debug(f"\nReranking with cosine similarity:")
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
        
        # Sort by similarity (moved outside debug block)
        reranked_docs = [doc for _, doc in sorted(
            zip(similarities, docs),
            key=lambda x: x[0],
            reverse=True
        )][:k]
        
        if args.debugLog:
            rerank_time = time.perf_counter() - start_time
            # Log reranking results
            log_debug("\nReranking results:")
            log_debug(f"  Reranking time: {rerank_time:.3f} seconds")
            for i, (doc, sim) in enumerate(zip(reranked_docs, sorted(similarities, reverse=True)[:k])):
                log_debug(f"\nDocument {i+1}:")
                log_debug(f"  Similarity score: {sim:.4f}")
                log_debug(f"  Word count: {len(doc.page_content.split())}")
                log_debug(f"  Content: {doc.page_content}")
        
        return reranked_docs

    def get_relevant_context(self, query: str, rerank_method: str = "Cosine Similarity") -> str:
        """Retrieve relevant email context for the query."""
        if args.debugLog:
            log_debug(f"\nRetrieving context for query: {query}")
            log_debug(f"  Use reranking: {rerank_method != 'No Reranking'}")
            start_time = time.perf_counter()
        
        if rerank_method == "No Reranking":
            # Original method without reranking
            docs = self.vectorstore.similarity_search(query, k=self.num_docs)
            if args.debugLog:
                retrieval_time = time.perf_counter() - start_time
                log_debug(f"  Retrieved {len(docs)} documents without reranking")
                log_debug(f"  Retrieval time: {retrieval_time:.3f} seconds")
        else:
            # Get more candidates for reranking
            docs = self.vectorstore.similarity_search(query, k=self.num_docs * self.rerank_multiplier)
            if args.debugLog:
                retrieval_time = time.perf_counter() - start_time
                log_debug(f"  Retrieved {len(docs)} documents for reranking")
                log_debug(f"  Initial retrieval time: {retrieval_time:.3f} seconds")
            
            # Apply selected reranking method
            if rerank_method == "Mistral Reranker":
                docs = self.mistral_rerank(query, docs, self.num_docs)
            else:  # Cosine Similarity
                docs = self.cosine_rerank(query, docs, self.num_docs)
            
            if args.debugLog:
                log_debug(f"  Reranked to {len(docs)} documents")
        
        # Get surrounding chunks for each document
        enhanced_docs = []
        for doc in docs:
            # Get document index from metadata if available
            doc_idx = doc.metadata.get('doc_idx', -1)
            if doc_idx != -1:
                # Get 2 chunks before and after
                start_idx = max(0, doc_idx - 2)
                end_idx = doc_idx + 3  # +3 to include the current chunk
                
                # Get surrounding chunks using FAISS index
                surrounding_docs = self.vectorstore.docstore.get_range(start_idx, end_idx)
                if surrounding_docs:
                    # Sort by index to maintain order
                    surrounding_docs.sort(key=lambda x: x.metadata.get('doc_idx', -1))
                    # Combine text in order
                    doc.page_content = "\n".join(d.page_content for d in surrounding_docs)
            enhanced_docs.append(doc)
        
        # Combine document contents
        context = "\n\n".join(doc.page_content for doc in enhanced_docs)
        if args.debugLog:
            log_debug(f"  Final context word count: {len(context.split())}")
        return context

    def setup_components(self):
        """Initialize the RAG components."""
        # Load the vector store
        self.vectorstore = FAISS.load_local(
            self.vectordb_path.as_posix(),
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Initialize the conversational chain components
        class DebugCallbackHandler(ConsoleCallbackHandler):
            """Custom callback handler that uses log_debug when debug is enabled"""
            def __init__(self):
                super().__init__()
                self.debug_enabled = args.debugLog
                self.print_function = log_debug if self.debug_enabled else print

            def on_llm_start(self, *args, **kwargs):
                if self.debug_enabled:
                    super().on_llm_start(*args, **kwargs)

            def on_llm_end(self, *args, **kwargs):
                if self.debug_enabled:
                    super().on_llm_end(*args, **kwargs)

            def on_llm_error(self, *args, **kwargs):
                if self.debug_enabled:
                    super().on_llm_error(*args, **kwargs)

            def on_chain_start(self, *args, **kwargs):
                if self.debug_enabled:
                    super().on_chain_start(*args, **kwargs)

            def on_chain_end(self, *args, **kwargs):
                if self.debug_enabled:
                    super().on_chain_end(*args, **kwargs)

            def on_chain_error(self, *args, **kwargs):
                if self.debug_enabled:
                    super().on_chain_error(*args, **kwargs)

            def _print(self, content: str) -> None:
                self.print_function(content)

        self.llm = ChatNVIDIA(
            base_url="http://0.0.0.0:8000/v1",
            model="meta/llama3-8b-instruct",
            temperature=0.1,
            max_tokens=1000,
            top_p=1.0,
            verbose=args.debugLog,  # Only enable verbose logging in debug mode
            callbacks=[
                DebugCallbackHandler()
            ]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Base system prompt that's common to both prompts
        self._base_system_prompt = f"""You are a helpful AI assistant that answers questions about the user's email history.
            The user's name is {self.user_name} and their email address is {self.user_email}.
            When they ask questions using "I" or "me", it refers to {self.user_name} ({self.user_email})."""

        # Create a custom QA prompt with user identity
        self.qa_prompt = PromptTemplate(
            template=f"""{self._base_system_prompt}
            
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep answers short and direct.
            
            Context: {{context}}
            
            Question: {{question}}
            
            Answer:"""
        )
        
        # Create the retrieval chain
        self.retriever = self.vectorstore.as_retriever()
        
        # Define the RAG chain with chat history
        def print_llm_input(x):
            if args.debugLog:
                log_debug("\nLLM_INPUT:", x)
                log_debug("\nLLM_INPUT type:", type(x))
                # If it's a list of messages, print each one's role and content
                if isinstance(x, list):
                    log_debug("\nDetailed message structure:")
                    for msg in x:
                        log_debug(f"Role: {msg.type}")
                        log_debug(f"Content: {msg.content}\n")
                    # Show what actually gets sent to the API
                    messages = [{"role": msg.type, "content": msg.content} for msg in x]
                    log_debug("\nActual API request format:")
                    log_debug(json.dumps({
                        "model": "meta/llama3-8b-instruct",
                        "messages": messages
                    }, indent=2))
            return x

        def print_llm_output(x):
            if args.debugLog:
                log_debug("\nLLM_RAW:", x)
                log_debug("\nLLM_RAW type:", type(x))
            return x

        self.qa = (
            RunnablePassthrough.assign(
                context=lambda input_dict: self.get_relevant_context(input_dict["question"], input_dict.get("rerank_method", "Cosine Similarity"))
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

        # Create the custom RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self._base_system_prompt}
            
            Use the following email content to answer the user's question.
            
            Rules:
            1. Answer ONLY the question asked - no additional context or explanations
            2. If you don't know the answer, just say "I don't know"
            3. Keep answers short and direct
            4. Do not include system messages, UI prompts, or follow-up questions
            
            Context from emails: {{context}}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{{question}}")
        ])
        self.prompt2 = ChatPromptTemplate.from_messages([
            ("system", f"""{self._base_system_prompt}
            
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep answers short and direct.
            
            Context: {{context}}
            
            Question: {{question}}
            
            Answer:"""),
            MessagesPlaceholder(variable_name="chat_history")
        ])

    def update_parameters(self, vectordb_path: str, user_name: str, user_email: str, num_docs: int, rerank_multiplier: int) -> None:
        """Update bot parameters and reinitialize components if needed."""
        if args.debugLog:
            log_debug("Updating parameters:")
            log_debug(f"  Database folder: {vectordb_path}")
            log_debug(f"  User name: {user_name}")
            log_debug(f"  User email: {user_email}")
            log_debug(f"  Number of documents: {num_docs}")
            log_debug(f"  Reranking multiplier: {rerank_multiplier}")
        
        self.vectordb_path = Path(vectordb_path)
        self.user_name = user_name
        self.user_email = user_email
        self.num_docs = num_docs
        self.rerank_multiplier = rerank_multiplier
        
        # Reinitialize components with new parameters
        self.setup_components()

    def query_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the local LLM."""
        try:
            if args.debugLog:
                log_debug("\nPROMPT STARTS")
                log_debug(prompt)
                log_debug("PROMPT ENDS\n")
                log_debug(f"Sending request to {self.llm_url}")
            response = requests.post(
                self.llm_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "meta/llama3-8b-instruct",
                    "prompt": prompt,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            if args.debugLog:
                log_debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            json_response = response.json()
            if args.debugLog:
                log_debug(f"Response JSON: {json_response}")
            return json_response["choices"][0]["text"].strip()
        except requests.exceptions.Timeout:
            error_msg = "LLM request timed out"
            log_error(error_msg)
            return "I apologize, but the request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error to LLM"
            log_error(error_msg)
            return "I apologize, but I couldn't connect to the language model. Please ensure the LLM container is running."
        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}\nError type: {type(e)}\n"
            if args.debugLog:
                error_msg += f"Traceback: {traceback.format_exc()}"
            log_error(error_msg)
            print(f"Line {inspect.currentframe().f_lineno}: {error_msg}")
            return "I apologize, but I encountered an error while processing your request."

    def format_chat_history(self) -> List[Dict[str, str]]:
        """Format chat history for the prompt."""
        messages = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def _update_chat_history(self, message: str, response: str):
        """Update both chat history and memory with new messages."""
        # Update chat history
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response})
        # Update memory with just the new messages
        self.memory.chat_memory.add_user_message(message)
        self.memory.chat_memory.add_ai_message(response)

    def chat_chain(self, message: str, rerank_method: str = "Cosine Similarity") -> str:
        """Process a chat message using the retrieval chain."""
        try:
            if args.debugLog:
                log_debug("\nProcessing query with chat_chain:")
                log_debug(f"  Query: {message}")
                log_debug(f"  Reranking method: {rerank_method}")
                log_debug(f"  Number of documents: {self.num_docs}")
                log_debug(f"  Reranking multiplier: {self.rerank_multiplier}")
            
            # Format chat history for the chain
            formatted_history = []
            for msg in self.chat_history:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                else:
                    formatted_history.append(AIMessage(content=msg["content"]))
            
            # Invoke the RAG chain with context and chat history
            response = self.qa.invoke({
                "question": message,
                "rerank_method": rerank_method,
                "chat_history": formatted_history
            })
            
            # Update chat history
            self._update_chat_history(message, response)
            
            if args.debugLog:
                log_debug(f"\nGenerated response: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error in chat_chain: {str(e)}\n{traceback.format_exc()}"
            log_error(error_msg)
            return "I encountered an error processing your request. Please try again."

    def chat_custom(self, message: str, rerank_method: str = "Cosine Similarity") -> str:
        """Process a chat message using custom RAG."""
        try:
            if args.debugLog:
                log_debug("\nProcessing query with chat_custom:")
                log_debug(f"  Query: {message}")
                log_debug(f"  Retrieval method: Simple RAG")
                log_debug(f"  Reranking method: {rerank_method}")
                log_debug(f"  Number of documents: {self.num_docs}")
                log_debug(f"  Reranking multiplier: {self.rerank_multiplier}")
                context_start = time.perf_counter()
            
            # Get relevant context
            context = self.get_relevant_context(message, rerank_method)
            
            if args.debugLog:
                context_time = time.perf_counter() - context_start
                log_debug(f"\nRetrieved context:")
                log_debug(f"  Context retrieval time: {context_time:.3f} seconds")
                log_debug(f"  Word count: {len(context.split())}")
                log_debug(f"  Content: {context}")
            
            # Generate response
            prompt = self.prompt2.format(
                context=context,
                question=message,
                chat_history=self.format_chat_history()
            )
            
            if args.debugLog:
                response_start = time.perf_counter()
            
            #intermediate_response = self.query_llm(prompt)
            #Create a prompt with the intermediate response
            #prompt = self.prompt2.format(
                #context=intermediate_response,
                #question=message,
                #chat_history=self.format_chat_history()
            #)
            response = self.query_llm(prompt)
            
            if args.debugLog:
                response_time = time.perf_counter() - response_start
                log_debug(f"\nGenerated response:")
                log_debug(f"  Response time: {response_time:.3f} seconds")
                log_debug(f"  {response}")
            
            # Update chat history and memory
            self._update_chat_history(message, response)
            
            return response
        except Exception as e:
            error_msg = f"Error in chat_custom: {str(e)}\nError type: {type(e)}\n"
            if args.debugLog:
                error_msg += f"Traceback: {traceback.format_exc()}"
            log_error(error_msg)
            print(f"Line {inspect.currentframe().f_lineno}: {error_msg}")
            return "I apologize, but I encountered an error while processing your request."

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        self.memory.clear()
        self.last_retrieved_docs = None
        return "Chat history cleared."

def create_chat_interface():
    # Track current parameter values
    current_params = {
        'vectordb_path': os.getenv("VECTOR_DB", "./mail_vectordb"),
        'user_name': os.getenv("USER_FULLNAME", "YOUR_NAME"),
        'user_email': os.getenv("USER_EMAIL", "YOUR_EMAIL"),
        'num_docs': 4,
        'rerank_multiplier': 3
    }
    
    def has_unsaved_changes(vdb_path, uname, uemail, num_docs, rerank_multiplier):
        """Check if any parameters have been changed from their current values."""
        return (
            vdb_path != current_params['vectordb_path'] or
            uname != current_params['user_name'] or
            uemail != current_params['user_email'] or
            int(num_docs) != current_params['num_docs'] or
            int(rerank_multiplier) != current_params['rerank_multiplier']
        )
    
    def update_change_indicator(*params):
        """Update the change indicator and update button state."""
        has_changes = has_unsaved_changes(*params)
        indicator = "⚠️ You have unsaved changes" if has_changes else ""
        button_state = gr.update(interactive=has_changes)
        return indicator, button_state
    
    # Get default values from env
    default_vectordb = os.getenv("VECTOR_DB", "./mail_vectordb")
    default_user_name = os.getenv("USER_FULLNAME", "YOUR_NAME")
    default_user_email = os.getenv("USER_EMAIL", "YOUR_EMAIL")
    
    # Create initial bot instance just to check container status
    initial_bot = EmailChatBot(
        vectordb_path=default_vectordb,
        user_name=default_user_name,
        user_email=default_user_email
    )
    
    with gr.Blocks(title="Email Chat Assistant") as interface:
        gr.Markdown("# 📧 Email Chat Assistant")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    container_status = gr.Textbox(
                        label="Container Status",
                        value=initial_bot.get_container_status(),
                        interactive=False
                    )
                    reranker_status = gr.Textbox(
                        label="Reranker Status",
                        value=initial_bot.get_reranker_status(),
                        interactive=False
                    )
                with gr.Row():
                    start_btn = gr.Button("Start LLM", variant="primary", interactive=initial_bot.get_container_status() == "stopped")
                    stop_btn = gr.Button("Stop LLM", variant="secondary", interactive=initial_bot.get_container_status() == "ready")
                    refresh_status = gr.Button("Refresh Status")
                with gr.Row():
                    start_reranker_btn = gr.Button("Start Reranker", variant="primary")
                    stop_reranker_btn = gr.Button("Stop Reranker", variant="secondary")
                    refresh_reranker_status = gr.Button("Refresh Reranker Status")
                with gr.Group():
                    vectordb_path = gr.Textbox(
                        label="Vector Database Directory",
                        value=default_vectordb,
                        info="Path to the FAISS vector database directory"
                    )
                    
                    user_name = gr.Textbox(
                        label="User Name",
                        value=default_user_name,
                        info="Your full name for personalized responses"
                    )
                    
                    user_email = gr.Textbox(
                        label="User Email",
                        value=default_user_email,
                        info="Your email for personalized responses"
                    )
                    
                    num_docs = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        label="Number of Documents",
                        info="Number of relevant documents to retrieve"
                    )
                    
                    rerank_multiplier = gr.Slider(
                        minimum=2,
                        maximum=10,  # Increased from 5 to 10
                        value=3,
                        step=1,
                        label="Reranking Multiplier",
                        info="Multiplier for number of documents to consider for reranking"
                    )
                    
                    change_indicator = gr.Markdown("")
                    
                    with gr.Row():
                        update_params = gr.Button("Update Parameters", variant="primary", interactive=False)
                        reset_params = gr.Button("Reset to Default Values", variant="secondary")
            with gr.Column(scale=1):
                pass
        
        # Initialize bot - use initial bot if container is already running
        bot = initial_bot if initial_bot.get_container_status() in ["ready", "starting"] else None
        
        retrieval_method = gr.Radio(
            ["Simple RAG", "Conversational Chain"],
            label="Retrieval Method",
            value="Conversational Chain"
        )
        
        rerank_method = gr.Dropdown(
            choices=["No Reranking", "Cosine Similarity", "Mistral Reranker"],
            label="Reranking Method",
            value="Cosine Similarity",
            type="value",
            info="Select the method to rerank retrieved documents"
        )
        
        chatbot = gr.Chatbot(
            height=600,
            type="messages"  # Use the new message format
        )
        msg = gr.Textbox(
            label="Type your message here",
            placeholder="e.g., 'Did I travel to the US in 2023? What dates did I travel?'",
            lines=2
        )
        
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear History")
        
        def update_status():
            """Get container status and return status string and button states separately."""
            if bot is None:
                return "stopped", gr.update(interactive=True), gr.update(interactive=False)  # Enable start button when stopped
            status = bot.get_container_status()
            return status, gr.update(interactive=status == "stopped"), gr.update(interactive=status == "ready")  # Enable stop only when ready

        def get_status_only():
            """Get container status string only and update button states."""
            if bot is None:
                return "stopped", gr.update(interactive=True), gr.update(interactive=False)
            status = bot.get_container_status()
            return status, gr.update(interactive=status == "stopped"), gr.update(interactive=status == "ready")
        
        def update_reranker_status():
            """Get reranker status and return status string and button state separately."""
            if bot is None:
                return "stopped", gr.update(interactive=True)
            status = bot.get_reranker_status()
            return status, gr.update(interactive=status == "stopped")

        def get_reranker_status_only():
            """Get reranker status string only."""
            if bot is None:
                return "stopped"
            return bot.get_reranker_status()
        
        def reset_to_defaults():
            """Reset all parameters to their default values from .env"""
            # Re-read from env in case .env was modified
            fresh_defaults = [
                os.getenv("VECTOR_DB", "./mail_vectordb"),
                os.getenv("USER_FULLNAME", "YOUR_NAME"),
                os.getenv("USER_EMAIL", "YOUR_EMAIL"),
                4,
                3
            ]
            return fresh_defaults

        def update_parameters(vdb_path, uname, uemail, num_docs, rerank_multiplier):
            nonlocal bot, current_params
            if bot is not None:
                # Convert slider values to integers
                num_docs = int(num_docs)
                rerank_multiplier = int(rerank_multiplier)
                
                # Update parameters on existing bot instead of creating new one
                bot.update_parameters(vdb_path, uname, uemail, num_docs, rerank_multiplier)
                # Update current parameters
                current_params.update({
                    'vectordb_path': vdb_path,
                    'user_name': uname,
                    'user_email': uemail,
                    'num_docs': num_docs,
                    'rerank_multiplier': rerank_multiplier
                })
                # Clear change indicator and disable update button
                return ["Parameters updated successfully", "", gr.update(interactive=False)]
            return ["Bot not initialized", "", gr.update(interactive=False)]
        
        def start_llm(vdb_path, uname, uemail, num_docs, rerank_multiplier):
            nonlocal bot, current_params
            try:
                # Convert slider values to integers
                num_docs = int(num_docs)
                rerank_multiplier = int(rerank_multiplier)
                
                # Use provided values
                bot = EmailChatBot(
                    vectordb_path=vdb_path.strip(),
                    user_name=uname.strip(),
                    user_email=uemail.strip(),
                    num_docs=num_docs,
                    rerank_multiplier=rerank_multiplier
                )
                # Update current parameters
                current_params.update({
                    'vectordb_path': vdb_path,
                    'user_name': uname,
                    'user_email': uemail,
                    'num_docs': num_docs,
                    'rerank_multiplier': rerank_multiplier
                })
                result = bot.start_container()
                status = bot.get_container_status()
                return [result, "", gr.update(interactive=False), gr.update(interactive=status == "stopped"), gr.update(interactive=status == "ready")]
            except ValueError as e:
                return [str(e), "", gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)]
        
        def stop_llm():
            nonlocal bot
            if bot is not None:
                result = bot.stop_container()
                bot = None
                return ["stopped", result, gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)]  # Match order: container_status, change_indicator, start_btn, stop_btn
            return ["stopped", "Bot not initialized", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)]  # Match order
        
        def start_reranker():
            nonlocal bot
            if bot is not None:
                result = bot.start_reranker()
                status = bot.get_reranker_status()
                return [result, status, gr.update(interactive=status == "stopped")]
            return ["Bot not initialized", "stopped", gr.update(interactive=True)]
        
        def stop_reranker():
            nonlocal bot
            if bot is not None:
                result = bot.stop_reranker()
                return [result, "stopped", gr.update(interactive=True)]
            return ["Bot not initialized", "stopped", gr.update(interactive=True)]
        
        def respond(message, history, method, rerank_method):
            if bot is None:
                return "", history + [{"role": "assistant", "content": "Please start the LLM container first"}]
                
            status = bot.get_container_status()
            if status != "ready":
                return "", history + [{"role": "assistant", "content": f"Please wait for the LLM container to be ready before sending messages. Current status: {status}"}]
            
            # Check if mistral reranking is selected but not ready
            if rerank_method == "Mistral Reranker" and bot.get_reranker_status() != "ready":
                return "", history + [{"role": "assistant", "content": "Mistral reranker is not ready. Please start it first or choose a different reranking method."}]
            
            if method == "Simple RAG":
                bot_response = bot.chat_custom(message, rerank_method=rerank_method)
            else:
                bot_response = bot.chat_chain(message, rerank_method=rerank_method)
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            
            return "", history
        
        def clear_chat_history():
            if bot is not None:
                bot.clear_history()
            return None
        
        def update_rerank_ui(rerank_choice):
            """Update UI elements based on reranking choice."""
            return gr.update(interactive=rerank_choice != "No Reranking")
        
        # Set up event handlers
        start_btn.click(
            start_llm,
            [vectordb_path, user_name, user_email, num_docs, rerank_multiplier],
            [container_status, change_indicator, update_params, start_btn, stop_btn]
        )
        stop_btn.click(
            stop_llm,
            None,
            [container_status, change_indicator, update_params, start_btn, stop_btn]
        )
        refresh_status.click(
            get_status_only,  
            None,
            [container_status, start_btn, stop_btn]
        )
        refresh_reranker_status.click(
            get_reranker_status_only,
            None,
            reranker_status
        )
        update_params.click(
            update_parameters,
            [vectordb_path, user_name, user_email, num_docs, rerank_multiplier],
            [container_status, change_indicator, update_params]
        )
        reset_params.click(
            reset_to_defaults,
            None,
            [vectordb_path, user_name, user_email, num_docs, rerank_multiplier]
        )
        
        start_reranker_btn.click(
            start_reranker,
            None,
            [reranker_status, start_reranker_btn]
        )
        stop_reranker_btn.click(
            stop_reranker,
            None,
            [reranker_status, start_reranker_btn]
        )
        
        # Connect reranking method to UI updates
        rerank_method.change(
            update_rerank_ui,
            inputs=[rerank_method],
            outputs=[rerank_multiplier]
        )
        
        submit.click(respond, [msg, chatbot, retrieval_method, rerank_method], [msg, chatbot])
        clear.click(clear_chat_history, None, chatbot)
        msg.submit(respond, [msg, chatbot, retrieval_method, rerank_method], [msg, chatbot])
        
        # Connect parameter change events to update indicator
        for param in [vectordb_path, user_name, user_email, num_docs, rerank_multiplier]:
            param.change(
                update_change_indicator,
                inputs=[vectordb_path, user_name, user_email, num_docs, rerank_multiplier],
                outputs=[change_indicator, update_params]
            )
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    create_chat_interface()