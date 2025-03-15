import os
import sys
import json
import time
import gradio as gr
import subprocess
import requests
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Load environment variables
load_dotenv()

class EmailChatBot:
    def __init__(self, vectordb_path: str = "./mail_vectordb"):
        self.vectordb_path = Path(vectordb_path)
        self.chat_history: List[Dict[str, str]] = []
        self.setup_environment()
        self.setup_components()

    def setup_environment(self):
        """Set up the NIM environment and start the LLM container if needed."""
        # Create NIM cache directory if it doesn't exist
        nim_cache = os.path.expanduser("~/.cache/nim")
        os.makedirs(nim_cache, exist_ok=True)
        os.chmod(nim_cache, 0o777)

        # Check if container is running
        result = subprocess.run(["docker", "ps", "-q", "-f", "name=meta-llama3-8b-instruct"], 
                              capture_output=True, text=True)
        
        if not result.stdout.strip():
            print("Starting LLM container...")
            # First try to remove any stopped container with the same name
            subprocess.run(["docker", "rm", "-f", "meta-llama3-8b-instruct"], 
                         capture_output=True, text=True)
            
            # Login to NGC
            ngc_key = os.getenv('NGC_API_KEY')
            if not ngc_key:
                raise ValueError("NGC_API_KEY environment variable not set")
            
            subprocess.run(["docker", "login", "nvcr.io", 
                          "--username", "$oauthtoken", 
                          "--password", ngc_key], check=True)
            
            # Start the container
            subprocess.run([
                "docker", "run", "-d",
                "--name", "meta-llama3-8b-instruct",
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
            
            print("Waiting for LLM container to initialize...")
            self.wait_for_llm_ready()
            print("LLM container is ready!")

    def wait_for_llm_ready(self, timeout=600, check_interval=10):
        """Wait for the LLM container to be fully initialized."""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("LLM container failed to initialize within timeout period")

            # Check container logs for the ready message
            logs = subprocess.run(
                ["docker", "logs", "meta-llama3-8b-instruct"],
                capture_output=True,
                text=True
            ).stdout

            if "Uvicorn running on http://0.0.0.0:8000" in logs:
                # Give it a few more seconds to fully initialize
                time.sleep(5)
                return True

            # Test if the API is responsive
            try:
                response = requests.post(
                    "http://0.0.0.0:8000/v1/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "meta/llama3-8b-instruct",
                        "prompt": "test",
                        "max_tokens": 1
                    },
                    timeout=5
                )
                if response.status_code == 200:
                    return True
            except:
                pass

            print("Waiting for LLM to initialize...")
            time.sleep(check_interval)

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
            allow_dangerous_deserialization=True  # We trust our local vector store
        )

        # Initialize the LLM
        self.llm_url = "http://0.0.0.0:8000/v1/completions"

        # Create the RAG prompt
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
                timeout=30  # Add timeout
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

    def get_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieve relevant email context for the query."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)

    def chat(self, message: str) -> str:
        """Process a chat message and return the response."""
        try:
            # Get relevant context
            print("Getting relevant context...")
            context = self.get_relevant_context(message)
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

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        return "Chat history cleared."

def create_chat_interface():
    """Create and launch the Gradio interface."""
    # Initialize the chat bot
    bot = EmailChatBot()
    
    # Define the interface
    with gr.Blocks(title="Email Chat Assistant") as interface:
        gr.Markdown("# Email Chat Assistant\nChat with your email history using AI")
        
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(
            label="Type your message here",
            placeholder="e.g., 'Did I travel to the US in 2023? What dates did I travel?'",
            lines=2
        )
        
        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear History")
        
        def respond(message, history):
            bot_response = bot.chat(message)
            history.append((message, bot_response))
            return "", history
        
        def clear_chat_history():
            bot.clear_history()
            return None
        
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat_history, None, chatbot)
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # Launch the interface
    interface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    create_chat_interface()