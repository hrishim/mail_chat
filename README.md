# mail_chat
Chat with your Email - A Privacy-Focused Local Email Assistant

## Introduction
`mail_chat` is a privacy-focused tool that enables you to have interactive conversations with your email archive while keeping your data completely local. It leverages NVIDIA's NIM (NVIDIA Inference Microservices) implementation of Meta's Llama 3 8B Instruct model for local inference, ensuring your email content never leaves your machine.

### Key Features
- **Privacy First**: All email processing and conversations happen locally on your machine
- **Local LLM Integration**: Uses NVIDIA's containerized Llama 3 8B model for local inference
- **Efficient Processing**: Processes email archives (mbox format) into optimized vector embeddings
- **Secure Credential Management**: Uses environment variables for secure API key handling

## Installation

### Prerequisites
1. **NVIDIA GPU**: Your machine must have an NVIDIA GPU capable of running the Llama 3 8B model
2. **NVIDIA Container Runtime**: Required for running the NIM container
3. **Python 3.10+**: Required for running the application

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/hrishim/mail_chat.git
   cd mail_chat
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv ~/.venv/nvidia_rag
   source ~/.venv/nvidia_rag/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **NVIDIA NGC Setup**
   - Create an account on [NVIDIA NGC](https://ngc.nvidia.com/)
   - Generate an API key:
     1. Log in to your NGC account
     2. Go to your "Account Settings"
     3. Select "Generate API Key"
     4. Save your API key securely
   - For detailed instructions, visit [NVIDIA's NGC documentation](https://docs.nvidia.com/ngc/gpu-cloud/ngc-overview/)

5. **Environment Configuration**
   - Create a `.env` file in the project root:
     ```bash
     touch .env
     ```
   - Add your NGC API key to the `.env` file:
     ```
     NGC_API_KEY=your-api-key-here
     ```

## Data Preparation

### Export Your Email
1. Export your email archive in mbox format
   - For Gmail:
     1. Go to [Google Takeout](https://takeout.google.com/)
     2. Select "Mail"
     3. Choose "mbox format"
     4. Download your archive
   - Save the mbox file in your project directory

### Process Your Email Archive
Run the data preparation script:
```bash
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
3. Create embeddings using NVIDIA's NV-Embed-QA model
4. Store the embeddings in a local FAISS vector database

## Next Steps
After processing your email archive, you'll be ready to start chatting with your emails using the local Llama 3 model. Stay tuned for instructions on setting up the RAG pipeline and interactive chat interface.
