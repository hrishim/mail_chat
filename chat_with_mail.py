import os
import sys
import json
import time
import gradio as gr
import subprocess
import requests
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

# Load environment variables
load_dotenv()

def is_valid_vectordb(path: Path) -> bool:
    """Check if the given path is a valid FAISS vector database directory."""
    return path.is_dir() and (path / "index.faiss").exists() and (path / "index.pkl").exists()

class EmailChatBot:
    def __init__(self, vectordb_path: Optional[str] = None, user_name: Optional[str] = None, user_email: Optional[str] = None,
                 num_docs: int = 4, rerank_multiplier: int = 3):
        # Use VECTOR_DB from env if vectordb_path not provided
        if vectordb_path is None:
            vectordb_path = os.getenv("VECTOR_DB", "./mail_vectordb")
        
        # Get user details from env if not provided
        self.user_name = user_name or os.getenv("USER_FULLNAME", "YOUR_NAME")
        self.user_email = user_email or os.getenv("USER_EMAIL", "YOUR_EMAIL")
        
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
        self.container_status = "stopped"
        self.reranker_status = "stopped"
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

    @property
    def container_name(self):
        return "meta-llama3-8b-instruct"

    @property
    def reranker_container_name(self):
        return "nv-rerankqa-mistral-4b"

    def get_container_status(self) -> str:
        """Get the current status of the LLM container."""
        try:
            # First check if container is running
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
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
                response = requests.get("http://0.0.0.0:8000/v1/health/ready", timeout=2)
                print(f"Health endpoint status code: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"Health endpoint response: {data}")
                    if data.get("message") == "Service is ready.":
                        return "ready"
                return "starting"
            except (requests.exceptions.RequestException, ValueError) as e:
                # If health check fails or invalid JSON, model is still starting
                print(f"Health endpoint error: {str(e)}")
                return "starting"
                
        except Exception as e:
            print(f"Error checking container status: {str(e)}")
            return "unknown"

    def get_reranker_status(self) -> str:
        """Get the current status of the reranker container."""
        try:
            # First check if container is running
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.reranker_container_name}", "--format", "{{.Status}}"],
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
                response = requests.get("http://0.0.0.0:8001/v1/health/ready", timeout=2)
                if response.status_code == 200:
                    return "ready"
                return "starting"
            except requests.exceptions.RequestException:
                return "starting"
                
        except Exception as e:
            print(f"Error checking reranker status: {str(e)}")
            return "unknown"

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
            return f"Error starting container: {str(e)}"

    def stop_container(self) -> str:
        """Stop the LLM container."""
        try:
            subprocess.run(["docker", "stop", self.container_name], check=True)
            subprocess.run(["docker", "rm", self.container_name], check=True)
            return "Container stopped"
        except Exception as e:
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
            return f"Error starting reranker: {str(e)}"
        except Exception as e:
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
            return f"Error stopping reranker: {str(e)}"

    def mistral_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using NV-RerankQA-Mistral-4B-v3."""
        try:
            # Check if reranker is ready
            if self.get_reranker_status() != "ready":
                print("Reranker is not ready, falling back to cosine similarity")
                return self.cosine_rerank(query, docs, k)

            # Prepare the request
            headers = {"Content-Type": "application/json"}
            data = {
                "query": query,
                "passages": [doc.page_content for doc in docs],
                "model": "nv-rerankqa-mistral-4b-v3"
            }

            # Make request to local reranker
            response = requests.post(
                "http://0.0.0.0:8001/v1/reranking/rerank",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            results = response.json()

            # Sort by scores and return top k
            scored_docs = list(zip(results["scores"], docs))
            sorted_docs = sorted(scored_docs, key=lambda x: float(x[0]), reverse=True)
            return [doc for _, doc in sorted_docs[:k]]

        except Exception as e:
            print(f"Reranking error: {str(e)}, falling back to cosine similarity")
            return self.cosine_rerank(query, docs, k)

    def cosine_rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using cosine similarity."""
        # Compute query embedding once
        query_embedding = self.embeddings.embed_query(query)
        
        # Score documents using cosine similarity
        scores = []
        for doc in docs:
            # Embed the document content
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append((similarity, doc))
        
        # Sort by similarity score and take top k
        sorted_docs = sorted(scores, key=lambda x: x[0], reverse=True)
        return [doc for _, doc in sorted_docs[:k]]

    def get_relevant_context(self, query: str, rerank_method: str = "Cosine Similarity") -> str:
        """Retrieve relevant email context for the query."""
        if rerank_method == "No Reranking":
            # Original method without reranking
            docs = self.vectorstore.similarity_search(query, k=self.num_docs)
        else:
            # Get more candidates for reranking
            docs = self.vectorstore.similarity_search(query, k=self.num_docs*self.rerank_multiplier)
            
            # Apply selected reranking method
            if rerank_method == "Mistral Reranker":
                docs = self.mistral_rerank(query, docs, self.num_docs)
            else:  # Cosine Similarity
                docs = self.cosine_rerank(query, docs, self.num_docs)
        
        # Store the retrieved documents
        self.last_retrieved_docs = docs
        return "\n\n".join(doc.page_content for doc in docs)

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
            print(f"Sending request to {self.llm_url}")
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
            print(f"Response status code: {response.status_code}")
            response.raise_for_status()
            json_response = response.json()
            print(f"Response JSON: {json_response}")
            return json_response["choices"][0]["text"].strip()
        except requests.exceptions.Timeout:
            print("LLM request timed out")
            return "I apologize, but the request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            print("Connection error to LLM")
            return "I apologize, but I couldn't connect to the language model. Please ensure the LLM container is running."
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
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

    def chat_simple(self, message: str, rerank_method: str = "Cosine Similarity") -> str:
        """Process a chat message using the simple RAG approach."""
        try:
            # Get relevant context
            print("Getting relevant context...")
            context = self.get_relevant_context(message, rerank_method=rerank_method)
            print(f"Found {len(context.split())} words of context")
            
            # Format the prompt with context and chat history
            print("Formatting prompt...")
            formatted_prompt = self.prompt.format(
                context=context,
                chat_history=self.format_chat_history(),
                question=message
            )
            print("Prompt formatted successfully")

            # Get response from LLM
            print("Querying LLM...")
            response = self.query_llm(formatted_prompt)
            print("Got response from LLM")
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return "I apologize, but I encountered an error while processing your request."

    def chat_chain(self, message: str, rerank_method: str = "Cosine Similarity") -> str:
        """Process a chat message using the ConversationalRetrievalChain."""
        try:
            if rerank_method != "No Reranking":
                # Create a custom retriever that uses reranking
                retriever = self.vectorstore.as_retriever()
                original_get_relevant_docs = retriever._get_relevant_documents
                
                def reranked_get_relevant_docs(*args, **kwargs):
                    # Get more candidates
                    k = kwargs.get('k', self.num_docs)  
                    docs = original_get_relevant_docs(*args, **{**kwargs, 'k': k * self.rerank_multiplier})
                    
                    # Apply selected reranking method
                    if rerank_method == "Mistral Reranker":
                        return self.mistral_rerank(message, docs, k)
                    else:  # Cosine Similarity
                        return self.cosine_rerank(message, docs, k)
                
                retriever._get_relevant_documents = reranked_get_relevant_docs
                self.qa.retriever = retriever
            
            result = self.qa.invoke({"question": message})
            response = result.get("answer", "I apologize, but I couldn't generate a response.")
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            print(f"Error in chat_chain: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return "I apologize, but I encountered an error while processing your request."

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        self.memory.clear()
        self.last_retrieved_docs = None
        return "Chat history cleared."

def create_chat_interface():
    """Create and launch the Gradio interface."""
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
                    start_btn = gr.Button("Start LLM", variant="primary")
                    stop_btn = gr.Button("Stop LLM", variant="secondary")
                    refresh_status = gr.Button("Refresh Status")
                with gr.Row():
                    start_reranker_btn = gr.Button("Start Reranker", variant="primary")
                    stop_reranker_btn = gr.Button("Stop Reranker", variant="secondary")
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
                        maximum=5,
                        value=3,
                        step=1,
                        label="Reranking Multiplier",
                        info="Multiplier for number of documents to consider for reranking"
                    )
                    
                    with gr.Row():
                        update_params = gr.Button("Update Parameters", variant="primary")
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
            if bot is None:
                return "stopped"
            return bot.get_container_status()
        
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
            nonlocal bot
            if bot is not None:
                # Update parameters on existing bot instead of creating new one
                bot.update_parameters(vdb_path, uname, uemail, num_docs, rerank_multiplier)
                return "Parameters updated successfully"
            return "Bot not initialized"
        
        def start_llm(vdb_path, uname, uemail, num_docs, rerank_multiplier):
            nonlocal bot
            try:
                # Use provided values
                bot = EmailChatBot(
                    vectordb_path=vdb_path.strip(),
                    user_name=uname.strip(),
                    user_email=uemail.strip(),
                    num_docs=num_docs,
                    rerank_multiplier=rerank_multiplier
                )
                result = bot.start_container()
                return [result, update_status()]
            except ValueError as e:
                return [str(e), "stopped"]
        
        def stop_llm():
            nonlocal bot
            if bot is not None:
                result = bot.stop_container()
                bot = None
                return [result, "stopped"]
            return ["Bot not initialized", "stopped"]
        
        def start_reranker():
            nonlocal bot
            if bot is not None:
                result = bot.start_reranker()
                return [result, bot.get_reranker_status()]
            return ["Bot not initialized", "stopped"]
        
        def stop_reranker():
            nonlocal bot
            if bot is not None:
                result = bot.stop_reranker()
                return [result, "stopped"]
            return ["Bot not initialized", "stopped"]
        
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
            [container_status, container_status]
        )
        stop_btn.click(
            stop_llm,
            None,
            [container_status, container_status]
        )
        update_params.click(
            update_parameters,
            [vectordb_path, user_name, user_email, num_docs, rerank_multiplier],
            container_status
        )
        reset_params.click(
            reset_to_defaults,
            None,
            [vectordb_path, user_name, user_email, num_docs, rerank_multiplier]
        )
        
        start_reranker_btn.click(
            start_reranker,
            None,
            [reranker_status, reranker_status]
        )
        stop_reranker_btn.click(
            stop_reranker,
            None,
            [reranker_status, reranker_status]
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
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    create_chat_interface()