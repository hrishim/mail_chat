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

class EmailChatBot:
    def __init__(self, vectordb_path: str = "./mail_vectordb"):
        self.vectordb_path = Path(vectordb_path)
        self.chat_history: List[Dict[str, str]] = []
        self.llm_url = "http://0.0.0.0:8000/v1/completions"
        self.container_status = "stopped"
        self.setup_components()

    @property
    def container_name(self):
        return "meta-llama3-8b-instruct"

    def get_container_status(self) -> str:
        """Get the current status of the LLM container."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                return "stopped"
            status = result.stdout.strip().lower()
            if "up" in status:
                # Check if it's still initializing
                logs = subprocess.run(
                    ["docker", "logs", self.container_name],
                    capture_output=True,
                    text=True
                ).stdout
                if "Uvicorn running on http://0.0.0.0:8000" in logs:
                    return "ready"
                return "starting"
            return "stopped"
        except:
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
        if not self.vectordb_path.exists():
            raise ValueError(f"Vector database not found at {self.vectordb_path}")

        # Initialize embeddings
        self.embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            truncate="END",
            api_key=os.getenv('NGC_API_KEY')
        )

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
        
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            chain_type="stuff",
            memory=self.memory,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT},
        )

        # Create the simple RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions about the user's email history. 
            Use the following pieces of email content to answer the user's question. 
            If you don't know the answer, just say that you don't know.
            Always maintain a friendly and professional tone.
            
            Context from emails: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
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

    def get_relevant_context(self, query: str, k: int = 4, use_rerank: bool = False) -> str:
        """Retrieve relevant email context for the query."""
        if not use_rerank:
            # Original method
            docs = self.vectorstore.similarity_search(query, k=k)
        else:
            # Get more candidates for reranking
            docs = self.vectorstore.similarity_search(query, k=k*3)
            
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
            
            # Sort by scores and take top k
            docs = [doc for _, doc in sorted(scores, reverse=True)[:k]]
        
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

    def chat_chain(self, message: str, use_rerank: bool = False) -> str:
        """Process a chat message using the ConversationalRetrievalChain."""
        try:
            if use_rerank:
                # Create a custom retriever that uses reranking
                retriever = self.vectorstore.as_retriever()
                original_get_relevant_docs = retriever._get_relevant_documents
                
                def reranked_get_relevant_docs(*args, **kwargs):
                    # Get more candidates
                    docs = original_get_relevant_docs(*args, **kwargs)
                    k = len(docs) // 3  # Get final k from original count
                    
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
                    
                    # Return top k reranked docs
                    return [doc for _, doc in sorted(scores, reverse=True)[:k]]
                
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
        return "Chat history cleared."

def create_chat_interface():
    """Create and launch the Gradio interface."""
    # Initialize the chat bot
    bot = EmailChatBot()
    
    # Define the interface
    with gr.Blocks(title="Email Chat Assistant") as interface:
        gr.Markdown("# Email Chat Assistant\nChat with your email history using AI")
        
        with gr.Row():
            with gr.Column(scale=2):
                container_status = gr.Textbox(
                    label="LLM Container Status",
                    value=bot.get_container_status(),
                    interactive=False
                )
            with gr.Column(scale=1):
                start_btn = gr.Button("Start LLM", variant="primary")
                stop_btn = gr.Button("Stop LLM", variant="secondary")
                refresh_status = gr.Button("Refresh Status", variant="secondary")
        
        retrieval_method = gr.Radio(
            choices=["Simple RAG", "Conversational Chain"],
            value="Simple RAG",
            label="Retrieval Method"
        )
        
        use_rerank = gr.Checkbox(label="Use Reranking")
        
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
            return bot.get_container_status()
        
        def start_llm():
            result = bot.start_container()
            return result, update_status()
        
        def stop_llm():
            result = bot.stop_container()
            return result, update_status()
        
        def respond(message, history, method, use_rerank):
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
            bot.clear_history()
            return None
        
        # Set up event handlers
        start_btn.click(start_llm, outputs=[container_status, container_status])
        stop_btn.click(stop_llm, outputs=[container_status, container_status])
        refresh_status.click(update_status, outputs=container_status)
        submit.click(respond, [msg, chatbot, retrieval_method, use_rerank], [msg, chatbot])
        clear.click(clear_chat_history, None, chatbot)
        msg.submit(respond, [msg, chatbot, retrieval_method, use_rerank], [msg, chatbot])
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    create_chat_interface()