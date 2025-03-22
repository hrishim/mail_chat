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
        self.last_retrieved_docs = None  # Store last retrieved documents
        
        # Initialize embeddings
        self.embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
        
        self.setup_components()

    @property
    def container_name(self):
        return "meta-llama3-8b-instruct"

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

    def get_relevant_context(self, query: str, use_rerank: bool = False) -> str:
        """Retrieve relevant email context for the query."""
        if not use_rerank:
            # Original method
            docs = self.vectorstore.similarity_search(query, k=self.num_docs)
        else:
            # Get more candidates for reranking
            docs = self.vectorstore.similarity_search(query, k=self.num_docs*self.rerank_multiplier)
            
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
            docs = [doc for _, doc in sorted_docs[:self.num_docs]]
        
        # Store the retrieved documents
        self.last_retrieved_docs = docs
        return "\n\n".join(doc.page_content for doc in docs)

    def chat_simple(self, message: str, use_rerank: bool = False) -> str:
        """Process a chat message using the simple RAG approach."""
        try:
            # Get relevant context
            print("Getting relevant context...")
            context = self.get_relevant_context(message, use_rerank=use_rerank)
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

    def chat_chain(self, message: str, use_rerank: bool = False, rerank_multiplier: int = 5) -> str:
        """Process a chat message using the ConversationalRetrievalChain."""
        try:
            if use_rerank:
                # Create a custom retriever that uses reranking
                retriever = self.vectorstore.as_retriever()
                original_get_relevant_docs = retriever._get_relevant_documents
                
                def reranked_get_relevant_docs(*args, **kwargs):
                    # Get more candidates
                    k = kwargs.get('k', 4)
                    docs = original_get_relevant_docs(*args, **{**kwargs, 'k': k * rerank_multiplier})
                    
                    # Compute query embedding once
                    query_embedding = self.embeddings.embed_query(message)
                    
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
                    
                    # Sort by similarity score (first element of tuple)
                    sorted_results = sorted(scores, key=lambda x: x[0], reverse=True)
                    # Return top k reranked docs
                    return [doc for _, doc in sorted_results[:k]]
                
                retriever._get_relevant_documents = reranked_get_relevant_docs
                self.qa.retriever = retriever
            
            result = self.qa({"question": message})
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
    initial_status = initial_bot.get_container_status()
    
    with gr.Blocks(title="Email Chat Assistant") as interface:
        gr.Markdown("# Email Chat Assistant\nChat with your email history using AI")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    vectordb_path = gr.Textbox(
                        label="Vector Database Directory",
                        placeholder="Enter vector database directory path",
                        value=default_vectordb,
                    )
                    gr.Markdown("*Value loaded from .env file*" if os.getenv("VECTOR_DB") else "*Using default path*")
                    
                    user_name = gr.Textbox(
                        label="User Name",
                        placeholder="Enter your name",
                        value=default_user_name,
                    )
                    gr.Markdown("*Value loaded from .env file*" if os.getenv("USER_FULLNAME") else "*Using default value*")
                    
                    user_email = gr.Textbox(
                        label="User Email",
                        placeholder="Enter your email",
                        value=default_user_email,
                    )
                    gr.Markdown("*Value loaded from .env file*" if os.getenv("USER_EMAIL") else "*Using default value*")
                    
                    num_docs = gr.Slider(
                        label="Number of Documents to Retrieve",
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1
                    )
                    rerank_multiplier = gr.Slider(
                        label="Reranking Multiplier",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1
                    )
                    
                    with gr.Row():
                        update_params = gr.Button("Update Parameters", variant="primary")
                        reset_params = gr.Button("Reset to Default Values", variant="secondary")
                
                container_status = gr.Textbox(
                    label="LLM Container Status",
                    value=initial_status,
                    interactive=False
                )
            with gr.Column(scale=1):
                start_btn = gr.Button("Start LLM", variant="primary")
                stop_btn = gr.Button("Stop LLM", variant="secondary")
                refresh_status = gr.Button("Refresh Status", variant="secondary")
        
        # Initialize bot - use initial bot if container is already running
        bot = initial_bot if initial_status in ["ready", "starting"] else None
        
        retrieval_method = gr.Radio(
            choices=["Simple RAG", "Conversational Chain"],
            value="Conversational Chain",
            label="Retrieval Method"
        )
        
        use_rerank = gr.Checkbox(label="Use Reranking", value=True)
        
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
            """Update bot with new parameters if it exists"""
            nonlocal bot
            if bot is not None:
                try:
                    # Create new bot with updated parameters
                    new_bot = EmailChatBot(
                        vectordb_path=vdb_path.strip(),
                        user_name=uname.strip(),
                        user_email=uemail.strip(),
                        num_docs=num_docs,
                        rerank_multiplier=rerank_multiplier
                    )
                    # If creation successful, update the bot
                    bot = new_bot
                    return "Parameters updated successfully"
                except ValueError as e:
                    return str(e)
            return "No active bot session. Start LLM to apply parameters."

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
        
        def respond(message, history, method, use_rerank):
            if bot is None:
                return "", history + [{"role": "assistant", "content": "Please start the LLM container first"}]
                
            status = bot.get_container_status()
            if status != "ready":
                return "", history + [{"role": "assistant", "content": f"Please wait for the LLM container to be ready before sending messages. Current status: {status}"}]
            
            if method == "Simple RAG":
                bot_response = bot.chat_simple(message, use_rerank=use_rerank)
            else:
                bot_response = bot.chat_chain(message, use_rerank=use_rerank)
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            
            return "", history
        
        def clear_chat_history():
            if bot is not None:
                bot.clear_history()
            return None
        
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
        submit.click(respond, [msg, chatbot, retrieval_method, use_rerank], [msg, chatbot])
        clear.click(clear_chat_history, None, chatbot)
        msg.submit(respond, [msg, chatbot, retrieval_method, use_rerank], [msg, chatbot])
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    create_chat_interface()