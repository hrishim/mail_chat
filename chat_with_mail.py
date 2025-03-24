import os
import sys
import json
import time
import gradio as gr
import subprocess
import requests
import argparse
import inspect
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import numpy as np
import logging
import traceback

# Parse command line arguments
parser = argparse.ArgumentParser(description='Email Chatbot with RAG')
parser.add_argument('--debugLog', type=str, help='Path to the debug log file. If not specified, debug logging will be disabled.')
args = parser.parse_args()

# Configure logging only if debug log file is specified
logger = logging.getLogger('rag_debug')
if args.debugLog:
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(args.debugLog)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
else:
    # Disable logging if no debug file specified
    logger.setLevel(logging.ERROR)
    logger.addHandler(logging.NullHandler())

# Load environment variables
load_dotenv()

def log_debug(message: str) -> None:
    """Helper function to log debug messages only if debug logging is enabled."""
    if args.debugLog:
        logger.debug(message)

def log_error(message: str) -> None:
    """Helper function to log error messages regardless of debug setting."""
    logger.error(message)

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

    @classmethod
    def check_container_status(cls, container_name: str, health_port: int = 8000) -> str:
        """Get the current status of a container.
        
        Args:
            container_name: Name of the Docker container to check
            health_port: Port number for the health endpoint (default: 8000 for LLM, use 8001 for reranker)
            
        Returns:
            str: Status of the container ('stopped', 'starting', or 'ready')
        """
        try:
            # First check if container is running
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                return "stopped"
            
            status = result.stdout.strip().lower()
            if "up" not in status:
                return "stopped"
            
            # Container is up, now check if model is ready using health endpoint
            try:
                response = requests.get(f"http://0.0.0.0:{health_port}/v1/health/ready", timeout=2)
                if args.debugLog:
                    log_debug(f"Health endpoint status code: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    if args.debugLog:
                        log_debug(f"Health endpoint response: {data}")
                    if data.get("message") == "Service is ready.":
                        return "ready"
                return "starting"
            except (requests.exceptions.RequestException, ValueError) as e:
                # If health check fails or invalid JSON, model is still starting
                if args.debugLog:
                    log_debug(f"Health endpoint error: {str(e)}")
                return "starting"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Container status check error: {str(e)}")
            return "stopped"

    def get_container_status(self) -> str:
        """Get the current status of the LLM container."""
        return self.check_container_status(self._llm_container)

    def get_reranker_status(self) -> str:
        """Get the current status of the reranker container."""
        return self.check_container_status(self._reranker_container, health_port=8001)

    @property
    def container_name(self):
        return self._llm_container

    @container_name.setter
    def container_name(self, name: str):
        self._llm_container = name
        # Update status when container name changes
        self.container_status = self.check_container_status(name)

    @property
    def reranker_container_name(self):
        return self._reranker_container

    @reranker_container_name.setter
    def reranker_container_name(self, name: str):
        self._reranker_container = name
        # Update status when container name changes
        self.reranker_status = self.check_container_status(name, health_port=8001)

    def start_container(self) -> str:
        """Start the LLM container."""
        try:
            if self.get_container_status() != "stopped":
                return "Container is already running"

            # Create NIM cache directory if it doesn't exist
            nim_cache = os.path.expanduser("~/.cache/nim")
            os.makedirs(nim_cache, exist_ok=True)
            os.chmod(nim_cache, 0o777)

            # First try to remove any stopped container with the same name
            subprocess.run(["docker", "rm", "-f", self.container_name], 
                         capture_output=True, text=True)
            
            # Login to NGC
            ngc_key = os.getenv('NGC_API_KEY')
            if not ngc_key:
                return "NGC_API_KEY environment variable not set"
            
            subprocess.run(["docker", "login", "nvcr.io", 
                          "--username", "$oauthtoken", 
                          "--password", ngc_key], check=True)
            
            # Start the container
            subprocess.run([
                "docker", "run", "-d",
                "--name", self.container_name,
                "--gpus", "all",
                "-e", f"NGC_API_KEY={ngc_key}",
                "-v", f"{nim_cache}:/opt/nim/.cache",
                "-u", str(os.getuid()),
                "-p", "8000:8000",
                "--shm-size=2g",
                "--ulimit", "memlock=-1",
                "--ipc=host",
                "nvcr.io/nim/meta/llama3-8b-instruct:1.0.0"
            ], check=True)
            
            return "Container starting..."
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error starting container: {str(e)}")
            return f"Error starting container: {str(e)}"

    def stop_container(self) -> str:
        """Stop the LLM container."""
        try:
            subprocess.run(["docker", "stop", self.container_name], check=True)
            subprocess.run(["docker", "rm", self.container_name], check=True)
            return "Container stopped"
        except Exception as e:
            if args.debugLog:
                log_debug(f"Error stopping container: {str(e)}")
            return f"Error stopping container: {str(e)}"

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
        
        # Combine document contents
        context = "\n\n".join(doc.page_content for doc in docs)
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
        self.llm = ChatNVIDIA(
            base_url="http://0.0.0.0:8000/v1",
            model="meta/llama3-8b-instruct",
            temperature=0.1,
            max_tokens=1000,
            top_p=1.0
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create a custom QA prompt with user identity
        custom_qa_prompt = PromptTemplate(
            template=f"""You are a helpful AI assistant that answers questions about the user's email history.
            The user's name is {self.user_name} and their email address is {self.user_email}.
            When they ask questions using "I" or "me", it refers to {self.user_name} ({self.user_email}).
            
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep answers short and direct.
            
            Context: {{context}}
            
            Question: {{question}}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type="stuff",
            memory=self.memory,
            combine_docs_chain_kwargs={'prompt': custom_qa_prompt},
        )

        # Create the simple RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a helpful AI assistant that answers questions about the user's email history. 
            The user's name is {self.user_name} and their email address is {self.user_email}.
            When they ask questions using "I" or "me", it refers to {self.user_name} ({self.user_email}).
            
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

    def chat_chain(self, message: str, rerank_method: str = "Cosine Similarity") -> str:
        """Process a chat message using the ConversationalRetrievalChain."""
        try:
            if args.debugLog:
                log_debug("\nProcessing query with chat_chain:")
                log_debug(f"  Query: {message}")
                log_debug(f"  Retrieval method: Conversational Chain")
                log_debug(f"  Reranking method: {rerank_method}")
                log_debug(f"  Number of documents: {self.num_docs}")
                log_debug(f"  Reranking multiplier: {self.rerank_multiplier}")
            
            if rerank_method != "No Reranking":
                # Create a custom retriever that uses reranking
                def get_relevant_documents(query: str, *, run_manager=None) -> List[Document]:
                    if args.debugLog:
                        start_time = time.perf_counter()
                    
                    # Get initial candidates
                    docs = self.vectorstore.similarity_search(
                        query,
                        k=self.num_docs * self.rerank_multiplier
                    )
                    
                    if args.debugLog:
                        retrieval_time = time.perf_counter() - start_time
                        # Log initial retrieval
                        total_words = sum(len(doc.page_content.split()) for doc in docs)
                        log_debug(f"\nInitial retrieval:")
                        log_debug(f"  Number of documents: {len(docs)}")
                        log_debug(f"  Total words: {total_words}")
                        for i, doc in enumerate(docs):
                            log_debug(f"\nDocument {i+1}:")
                            log_debug(f"  Word count: {len(doc.page_content.split())}")
                            log_debug(f"  Content: {doc.page_content}")
                    
                    # Apply selected reranking method
                    if rerank_method == "Mistral Reranker":
                        reranked_docs = self.mistral_rerank(query, docs, self.num_docs)
                    else:  # Cosine Similarity
                        reranked_docs = self.cosine_rerank(query, docs, self.num_docs)
                    
                    if args.debugLog:
                        # Log reranked results
                        total_words = sum(len(doc.page_content.split()) for doc in reranked_docs)
                        log_debug(f"\nAfter reranking:")
                        log_debug(f"  Number of documents: {len(reranked_docs)}")
                        log_debug(f"  Total words: {total_words}")
                        for i, doc in enumerate(reranked_docs):
                            log_debug(f"\nDocument {i+1}:")
                            log_debug(f"  Word count: {len(doc.page_content.split())}")
                            log_debug(f"  Content: {doc.page_content}")
                    
                    # Return the reranked documents
                    return reranked_docs
                
                retriever = self.vectorstore.as_retriever()
                retriever._get_relevant_documents = get_relevant_documents
                self.qa.retriever = retriever
            else:
                # Log that we're using standard retrieval
                if args.debugLog:
                    log_debug("\nUsing standard retrieval without reranking")
            
            # Generate response
            if args.debugLog:
                response_start = time.perf_counter()
            
            result = self.qa.invoke({"question": message})
            response = result.get("answer", "I apologize, but I couldn't generate a response.")
            
            if args.debugLog:
                response_time = time.perf_counter() - response_start
                log_debug(f"\nGenerated response:")
                log_debug(f"  Response time: {response_time:.3f} seconds")
                log_debug(f"  {response}")
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            error_msg = f"Error in chat_chain: {str(e)}\nError type: {type(e)}\n"
            if args.debugLog:
                error_msg += f"Traceback: {traceback.format_exc()}"
            log_error(error_msg)
            print(f"Line {inspect.currentframe().f_lineno}: {error_msg}")
            return "I apologize, but I encountered an error while processing your request."

    def chat_simple(self, message: str, rerank_method: str = "Cosine Similarity") -> str:
        """Process a chat message using simple RAG."""
        try:
            if args.debugLog:
                log_debug("\nProcessing query with chat_simple:")
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
            prompt = self.prompt.format(
                context=context,
                question=message,
                user_name=self.user_name,
                user_email=self.user_email
            )
            
            if args.debugLog:
                response_start = time.perf_counter()
            
            response = self.query_llm(prompt)
            
            if args.debugLog:
                response_time = time.perf_counter() - response_start
                log_debug(f"\nGenerated response:")
                log_debug(f"  Response time: {response_time:.3f} seconds")
                log_debug(f"  {response}")
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            error_msg = f"Error in chat_simple: {str(e)}\nError type: {type(e)}\n"
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
                bot_response = bot.chat_simple(message, rerank_method=rerank_method)
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