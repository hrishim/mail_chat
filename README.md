# mail_chat
Chat with your Email - A Privacy-Focused Local Email Assistant

## Introduction
`mail_chat` is a privacy-focused tool that enables you to have interactive conversations with your email archive while keeping your data completely local. It uses NVIDIA's Inference Microservices (NIM) to run both the Llama 3 8B Instruct model and NV-Embed-QA model locally for inference and embeddings generation. Your email data never leaves your machine at any point during processing or querying.

**NOTE:** This project is still in *Beta* and has limitations and bugs. In my testing the Simple RAG is practically useless (it is my scaffolding to get better). Conversational Chain RAG is the one that works, but it is not as smooth as I would like it to be. Partly because of model size and partly because I am still learning how to build RAG pipelines.

### Privacy & Security
- **Completely Local Processing**: All components (LLM, embeddings, vector store) run locally
- **No External Services**: The RAG pipeline is entirely self-contained on your machine
- **Local Email Processing**: All email content is processed and stored locally
- **Local LLM Inference**: Chat interactions use local Llama 3 model inference
- **Local Embeddings**: Uses NVIDIA's NV-Embed-QA model running in a local container for generating embeddings
- **Secure Storage**: Embeddings are stored locally in a FAISS vector database
- **API Security**: Uses environment variables and `.env` files for secure credential management

### Key Features
- **Privacy First**: All email processing and conversations happen locally on your machine
- **Local LLM Integration**: Uses NVIDIA NIM to run Llama 3 8B and NV-Embed-QA models locally
- **Efficient Processing**: Processes email archives (mbox format) into optimized vector embeddings
- **Secure Credential Management**: Uses environment variables for secure API key handling
- **Modern Architecture**: Uses LangChain's modular design with components from `langchain_core` and `langchain_community`

### Technical Architecture
This project leverages LangChain's modern modular architecture:
- **Core Components** (`langchain_core`):
  - Document handling and metadata management via `langchain_core.documents` (required for vector store metadata)
  - Core abstractions and interfaces for RAG pipeline
- **Community Components** (`langchain_community`):
  - Vector store implementation using FAISS from `langchain_community.vectorstores`
  - Efficient storage and retrieval of email embeddings
  - Memory-efficient vector similarity search
- **NVIDIA AI Integration**:
  - Local model inference using NVIDIA NIM containers
  - Integration with NVIDIA AI Endpoints (version 0.2.0+ required)
- **Local Infrastructure**:
  - NVIDIA NIM containers for model inference
  - FAISS vector database for local storage
  - Environment-based configuration management

**Note on Package Versions**: This project specifically uses:
- `langchain_core.documents` for Document class and metadata handling
- `langchain_community.vectorstores` for FAISS integration
These are part of LangChain's modularization effort and require specific versions as listed in `requirements.txt`

## Installation

### Prerequisites
1. **NVIDIA GPU**: Required specifications:
   - **Minimum**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
   - **Recommended**: NVIDIA RTX 4090 (24GB VRAM) or better
   - **Note**: Lower VRAM GPUs (e.g., RTX 3080, 3070) may not be able to run the NIM containers due to memory requirements of the Llama 3 8B model
2. **System Memory**:
   - **Minimum**: 32GB RAM
   - **Recommended**: 64GB RAM
   - **Note**: The FAISS vector store from `langchain_community.vectorstores` is loaded into memory, so RAM requirements increase with the size of your email archive
3. **Docker with NVIDIA Container Runtime**: Required for running the NIM containers
4. **Python 3.10+**: Required for running the application
5. **NVIDIA Driver**: Latest NVIDIA driver compatible with CUDA
6. **Storage**: At least 20GB free disk space for the models and containers
7. **Internet Connection**: Required only for initial container downloads and NGC authentication

### Dependencies
This project uses several key dependencies:
- **LangChain Core**: Core abstractions and interfaces for RAG components
- **LangChain Community**: Community-maintained integrations including FAISS vector store
- **NVIDIA AI Endpoints**: For running NV-Embed-QA model locally through NIM (version 0.2.0 or higher required)
- **FAISS-CPU**: For efficient vector storage and retrieval
- **Python-Magic**: For file type detection
- **Python-Dotenv**: For secure environment variable management

All dependencies are specified in `requirements.txt` with their correct versions.

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hrishim/mail_chat.git
   cd mail_chat
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create a new virtual environment
   python -m venv ~/.venv/mymail_rag
   
   # Activate it (choose one based on your shell):
   # For bash/zsh:
   source ~/.venv/mymail_rag/bin/activate
   # For fish:
   source ~/.venv/mymail_rag/bin/activate.fish
   # For csh/tcsh:
   source ~/.venv/mymail_rag/bin/activate.csh
   
   # Verify you're using the correct Python
   which python
   # Should show: ~/.venv/mymail_rag/bin/python
   ```

3. **Install Dependencies**
   ```bash
   # Make sure you're in the project directory and your virtual environment is active
   pip install -r requirements.txt
   ```

4. **NVIDIA NGC Setup**
   - Create an account on [NVIDIA NGC](https://catalog.ngc.nvidia.com/)
   - Generate an API key:
     1. Log in to your NGC account
     2. Go to your "Account Settings"
     3. Select "Generate API Key"
     4. Save your API key securely
   - For detailed instructions, visit the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/gpu-cloud/ngc-overview/)

5. **NGC Authentication**
   
   Choose one of these two methods to authenticate:

   **Method 1: Using Environment Variable**
   ```bash
   # Make sure your virtual environment is active
   source ~/.venv/mymail_rag/bin/activate
   
   # Set NGC API key in your environment
   export NGC_API_KEY='your-api-key-here'
   
   # Log in to NGC container registry
   docker login nvcr.io --username '$oauthtoken' --password "${NGC_API_KEY}"
   
   # Pull the Llama 3 model
   docker pull nvcr.io/nim/meta/llama3-8b-instruct:1.0.0
   ```

   **Method 2: Using .env File**
   ```bash
   # Make sure your virtual environment is active
   source ~/.venv/mymail_rag/bin/activate
   
   # Create .env file
   echo "NGC_API_KEY=your-api-key-here" > .env
   
   # Load the API key and log in to NGC
   export NGC_API_KEY=$(grep NGC_API_KEY .env | cut -d '=' -f2)
   docker login nvcr.io --username '$oauthtoken' --password "${NGC_API_KEY}"
   
   # Pull the Llama 3 model
   docker pull nvcr.io/nim/meta/llama3-8b-instruct:1.0.0
   ```

   The `.env` file will be used by both the Docker authentication and our Python scripts.

## Data Preparation

### Memory Usage Considerations
Before processing your email archive, consider these memory requirements:
- The FAISS vector store from `langchain_community.vectorstores` loads all embeddings into RAM
- Each email chunk requires approximately 384 bytes for metadata and 1024 floating point numbers (4KB) for embeddings
- As a rough estimate, each email thread might generate 2-3 chunks on average
- Example calculation:
  - For 10,000 email threads:
    - ~25,000 chunks (assuming 2.5 chunks per thread)
    - ~110MB for embeddings (25,000 * 4KB)
    - ~10MB for metadata
    - Additional memory overhead for FAISS indexing
  - Recommend having at least 2-3x this amount in free RAM

### Export Your Gmail Data
1. Go to [Google Takeout](https://takeout.google.com/)
2. **Deselect All** products first by clicking "Deselect all" at the top
3. Scroll down and find "Mail" in the list
4. Click the checkbox next to "Mail" to select only your email data
5. Click "Multiple formats" next to Mail
6. Change the export format to "MBOX" format
7. Click "Next step"
8. Configure your export:
   - Choose "Export once"
   - Choose your preferred file size (we recommend "2GB" for easier handling)
   - Choose your preferred delivery method (we recommend "Download link via email")
9. Click "Create export"
10. Wait for the email from Google (usually takes a few minutes to hours depending on size)
11. Download your archive when ready
12. Extract the downloaded archive
13. Look for the file named `All mail Including Spam and Trash.mbox`
    - This contains all your emails, including those in Spam and Trash
    - If you don't want these, you can export specific labels instead in step 5
14. Move or copy this file to your project directory

### Process Your Email Archive
Make sure your virtual environment is active, then run the data preparation script:
```bash
# Activate virtual environment if not already active
source ~/.venv/mymail_rag/bin/activate

# Run the script
python data_prep.py path/to/your/email.mbox
```

Optional arguments:
- `--chunk-size`: Number of email threads to process in each batch (default: 50)
- `--text-chunk-size`: Maximum size of each text chunk in characters (default: 400)
- `--chunk-overlap`: Number of characters to overlap between chunks (default: 100)
- `--vectordb-dir`: Directory to store the vector database (default: ./mail_vectordb)

The script will:
1. Load and parse your email archive
2. Process emails into meaningful chunks
3. Create embeddings using NVIDIA's NV-Embed-QA model running locally through NIM
4. Store the embeddings in a local FAISS vector database

All processing happens locally on your machine. No email content or embeddings are ever sent to external services.

## Next Steps
After processing your email archive, you'll be ready to start chatting with your emails using the local Llama 3 model. Stay tuned for instructions on setting up the RAG pipeline and interactive chat interface. The entire RAG pipeline runs locally, ensuring your email conversations remain private.
