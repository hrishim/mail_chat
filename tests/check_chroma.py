#!/usr/bin/env python3
import os
import sys
import chromadb
from dotenv import load_dotenv

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path, override=True)

# Get the Chroma DB path from environment
chroma_path = os.getenv("LANGCHAIN_CHROMA")
if not chroma_path:
    print("LANGCHAIN_CHROMA environment variable not set")
    sys.exit(1)

print(f"Checking Chroma database at: {chroma_path}")

# Remove quotes if present
if chroma_path.startswith('"') and chroma_path.endswith('"'):
    chroma_path = chroma_path[1:-1]

# Check if the directory exists
if not os.path.exists(chroma_path):
    print(f"Chroma directory does not exist: {chroma_path}")
    sys.exit(1)

print(f"Chroma directory exists and is {os.path.getsize(chroma_path) / (1024*1024):.2f} MB")

# Try to connect to the Chroma database
try:
    client = chromadb.PersistentClient(path=os.path.dirname(chroma_path))
    print(f"Successfully connected to Chroma client")
    
    # List all collections
    collections = client.list_collections()
    print(f"Available collections: {collections}")
    
    # Try to get the email_chunks collection
    try:
        collection = client.get_collection("email_chunks")
        print(f"Found 'email_chunks' collection with {collection.count()} documents")
        
        # Get a sample of documents
        if collection.count() > 0:
            results = collection.peek(limit=5)
            print(f"Sample document IDs: {results['ids']}")
            print(f"Sample document metadatas: {results['metadatas']}")
    except Exception as e:
        print(f"Error accessing 'email_chunks' collection: {str(e)}")
        
except Exception as e:
    print(f"Error connecting to Chroma database: {str(e)}")

print("Check complete")
