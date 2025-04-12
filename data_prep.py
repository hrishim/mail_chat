from multiprocessing import Pool, cpu_count
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
from mailbox import mboxMessage
from typing import Optional, Generator, Union, Dict, Any
import json
import mailbox
from email import message_from_bytes
from email.header import decode_header, make_header
from bs4 import BeautifulSoup
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from db_utils import EmailStore
from date_utils import TZ_MAP, DATE_FORMATS, parse_date_for_sorting

# Load environment variables from .env file
load_dotenv()

# Initialize text splitter with conservative settings for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # ~100 tokens
    chunk_overlap=40,  # 10% overlap
    length_function=len,
    is_separator_regex=False
)

class Message:
    def __init__(self, to: str, sender:str, subject: str, date: str, 
                 content: str, x_gmail_labels: list[str]=[],
                 x_gm_thrid: Optional[str]=None, inReplyTo:Optional[str]=None, 
                 references:list[str] = [], message_id: str = ""):
        self.to: str = to
        self.sender: str = sender
        self.subject: str = subject
        self.date: str = date
        self.content: str = content
        self.x_gmail_labels: list[str] = x_gmail_labels
        self.x_gm_thrid: Optional[str] = x_gm_thrid
        self.inReplyTo: Optional[str] = inReplyTo
        self.references: list[str] = references
        self.message_id: str = message_id

    def __str__(self):
        """Return just the content for string representation.
        This is used for display purposes only."""
        return self.content

    def get_metadata(self) -> dict:
        """Get all metadata as a dictionary for storage in Document."""
        return {
            'to': self.to,
            'sender': self.sender,
            'subject': self.subject,
            'date': self.date,
            'x_gmail_labels': self.x_gmail_labels,
            'thread_id': self.x_gm_thrid,
            'inReplyTo': self.inReplyTo,
            'references': self.references,
            'message_id': self.message_id
        }

    def to_document(self) -> Document:
        """Convert Message to a LangChain Document with proper metadata separation."""
        return Document(
            page_content=self.content,
            metadata=self.get_metadata()
        )

    def get_header_str(self) -> str:
        """Get formatted header string for display purposes."""
        return (
            f"From: {self.sender}\n"
            f"To: {self.to}\n"
            f"Subject: {self.subject}\n"
            f"Date: {self.date}\n"
            f"Labels: {', '.join(self.x_gmail_labels)}"
        )

    @classmethod
    def from_email_message(cls, message: mboxMessage) -> Optional['Message']:
        """Create Message object from mboxMessage."""
        try:
            content = extract_content(message)
            if not content:
                return None
            
            content = clean_content(content)
            
            # Get Gmail-specific headers
            gmail_labels = str(message.get('X-Gmail-Labels', '')).split(',')
            gmail_labels = [label.strip() for label in gmail_labels if label.strip()]
            
            thread_id = str(message.get('X-GM-THRID', ''))
            in_reply_to = str(message.get('In-Reply-To', ''))
            references = str(message.get('References', '')).split()
            message_id = str(message.get('Message-ID', ''))
            
            # Parse and standardize the date
            date_str = str(message.get('Date', ''))
            if not date_str:
                print("Warning: Message has no date, using default date")
                date = "1970-01-01 00:00:00+0000"
            else:
                try:
                    date = parse_date_for_sorting(date_str)
                except ValueError:
                    print("Warning: Could not parse date, using default date")
                    date = "1970-01-01 00:00:00+0000"
            
            # Properly decode headers
            def decode_header_str(header):
                if not header:
                    return ''
                try:
                    return str(make_header(decode_header(str(header))))
                except:
                    return str(header)
            
            return cls(
                to=decode_header_str(message.get('To')),
                sender=decode_header_str(message.get('From')),
                subject=decode_header_str(message.get('Subject')),
                date=date,
                content=content,
                x_gmail_labels=gmail_labels,
                x_gm_thrid=thread_id,
                inReplyTo=in_reply_to,
                references=references,
                message_id=message_id
            )
        except Exception as e:
            print(f"Error creating Message object: {e}")
            return None

def extract_content(message: mboxMessage) -> Optional[str]:
    """Extracts content from an email message."""
    if message.is_multipart():
        content = ''
        for part in message.walk():
            if part.get_content_type() == 'text/plain':
                # Explicitly cast get_payload(decode=True) to bytes
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    content += payload.decode('utf-8', errors='ignore')
            elif part.get_content_type() == 'text/html':
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    html_content = payload.decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(html_content, 'html.parser')
                    content += soup.get_text()
        return content if content else None  # Return None if content is empty
    else:
        if message.get_content_type() == 'text/plain':
            payload = message.get_payload(decode=True)
            if isinstance(payload, bytes):
                return payload.decode('utf-8', errors='ignore')
        elif message.get_content_type() == 'text/html':
            payload = message.get_payload(decode=True)
            if isinstance(payload, bytes):
                html_content = payload.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text()
        return None  # Return None for unsupported content types

def clean_content(content: Optional[str]) -> str:
    """Removes any remaining HTML tags and extra whitespace."""
    if content:
        content = re.sub(r'<.*?>', '', content)  # Remove any remaining HTML tags
        content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
        return content.strip()
    return ""

def log_date_error(date_str: str, error_type: str = "Failed to parse") -> None:
    """Log date parsing errors to file with absolute path."""
    try:
        # Create a log file in the current directory
        log_file = Path('date_fmt_error.txt').absolute()
        
        with open(log_file, 'a') as f:
            # Write the error and the date string
            f.write(f"{error_type}: '{date_str}'\n")
            
            # Try to parse with each format and log the error
            for fmt in DATE_FORMATS:
                try:
                    datetime.strptime(date_str, fmt)
                    f.write(f"  Format {fmt} works!\n")
                    break
                except ValueError as e:
                    f.write(f"  Format {fmt}: {str(e)}\n")
            
            f.write("\n")  # Add a blank line between entries
    except Exception as e:
        print(f"Error logging date error: {e}")

def prepare_chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare metadata for Chroma by converting lists to strings."""
    converted = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            converted[key] = ', '.join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            converted[key] = value
    return converted

def get_thread_batches(mbox_path: str, max_emails: Optional[int] = None) -> Generator[list[list[Message]], None, None]:
    """Load and organize messages from mbox file into batches of threads."""
    current_batch = {}  # thread_id -> list[Message]
    threads_per_batch = 100  # Process 100 threads at a time
    email_count = 0
    
    # Open and process mbox file
    mbox = mailbox.mbox(mbox_path)
    
    for idx, message in enumerate(mbox):
        if max_emails and email_count >= max_emails:
            print(f"Reached max emails limit ({max_emails})")
            break
        
        try:
            # Extract thread ID
            thread_id = str(message.get('X-GM-THRID', ''))
            if not thread_id:
                continue  # Skip messages without thread ID
            
            # Create Message object
            msg = Message.from_email_message(message)
            if msg is None:
                continue  # Skip if message creation failed
            
            # Add to current batch
            if thread_id not in current_batch:
                current_batch[thread_id] = []
            current_batch[thread_id].append(msg)
            
            email_count += 1
            
            # If we have enough threads, sort and yield the batch
            if len(current_batch) >= threads_per_batch:
                thread_batch = []
                for thread_messages in current_batch.values():
                    # Sort messages by date
                    thread_messages.sort(key=lambda msg: parse_date_for_sorting(msg.date) or datetime.min.replace(tzinfo=timezone.utc))
                    thread_batch.append(thread_messages)
                yield thread_batch
                current_batch = {}
                
        except Exception as e:
            print(f"Error processing message {idx}: {e}")
            with open('date_fmt_error.txt', 'a') as f:
                f.write(f"Error in message {idx}: {e}\n")
            continue
    
    # Yield remaining messages
    if current_batch:
        thread_batch = []
        for thread_messages in current_batch.values():
            # Sort messages by date
            thread_messages.sort(key=lambda msg: parse_date_for_sorting(msg.date) or datetime.min.replace(tzinfo=timezone.utc))
            thread_batch.append(thread_messages)
        yield thread_batch

def process_thread_batch(thread_batch: list[list[Message]]) -> list[Document]:
    """Process a single batch of threads into documents."""
    documents = []
    
    for thread_messages in thread_batch:
        # Sort messages by date
        thread_messages.sort(key=lambda x: parse_date_for_sorting(x.date))
        
        # Get thread context
        thread_id = thread_messages[0].x_gm_thrid
        thread_start_date = thread_messages[0].date
        
        # Process each message in thread
        for i, message in enumerate(thread_messages):
            # Create metadata for the document
            metadata = message.get_metadata()
            metadata.update({
                'is_chunk': False,
                'thread_id': thread_id,
                'message_index': i,
                'thread_start_date': thread_start_date,
                'thread_position': f"{i+1} of {len(thread_messages)}",
                'full_content_id': f"{thread_id}_{i}"
            })
            
            # Create full document
            full_doc = Document(
                page_content=message.content,
                metadata=metadata
            )
            documents.append(full_doc)
            
            # Create chunks for embedding
            chunks = text_splitter.split_text(message.content)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'is_chunk': True,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'full_content_id': f"{thread_id}_{i}"  # Link back to full content
                })
                
                # For Chroma, convert lists to strings in metadata
                if chunk_metadata:
                    chunk_metadata = prepare_chroma_metadata(chunk_metadata)
                
                chunk_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(chunk_doc)
    
    return documents

def parse_args():
    parser = argparse.ArgumentParser(description='Process email threads from mbox file.')
    parser.add_argument('mbox_file', help='Path to mbox file')
    parser.add_argument('--db-type', choices=['faiss', 'chroma'], default='faiss',
                      help='Type of vector database to use')
    parser.add_argument('--vectordb-dir', default='./mail_vectordb',
                      help='Directory to store vector database')
    parser.add_argument('--max-emails', type=int, default=None,
                      help='Maximum number of emails to process')
    parser.add_argument('--sqldb-path', default='./email_store.db',
                      help='Path to SQLite database for storing full documents')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    vectorstore = None
    all_documents = []  # Keep track of all documents
    total_threads = 0
    
    print("Processing mbox file...")
    try:
        # Process messages sequentially
        for thread_batch in get_thread_batches(args.mbox_file, max_emails=args.max_emails):
            # Process each batch
            documents = process_thread_batch(thread_batch)
            all_documents.extend(documents)  # Accumulate all documents
            total_threads += len(thread_batch)
            print(f"Processed batch: {len(documents)} documents from {len(thread_batch)} threads")
            print(f"Running total: {len(all_documents)} documents from {total_threads} threads")
            
        print(f"\nTotal documents processed: {len(all_documents)}")
        
        # Separate chunks and full documents
        chunks = [doc for doc in all_documents if doc.metadata.get('is_chunk', False)]
        full_docs = {doc.metadata['full_content_id']: doc 
                    for doc in all_documents 
                    if not doc.metadata.get('is_chunk', False)}
        
        print(f"Creating vectorstore with {len(chunks)} chunks...")
        embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            api_key=os.getenv("NGC_API_KEY")
        )
        
        # Process chunks in batches to avoid memory issues
        batch_size = 1000
        vectorstore = None
        
        if args.db_type == 'faiss':
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
                
                if vectorstore is None:
                    # First batch - create new vectorstore
                    vectorstore = FAISS.from_documents(
                        batch,
                        embeddings,
                    )
                else:
                    # Subsequent batches - add to existing vectorstore
                    vectorstore.add_documents(batch)
                    
                # Save progress after each batch
                print(f"Saving progress to {args.vectordb_dir}...")
                os.makedirs(args.vectordb_dir, exist_ok=True)
                vectorstore.save_local(args.vectordb_dir)
            
            # Add full documents to docstore without embedding them
            print(f"Adding {len(full_docs)} full documents to docstore...")
            for full_doc_id, full_doc in full_docs.items():
                vectorstore.docstore._dict[f"full_{full_doc_id}"] = full_doc
            
            print(f"Saving final vectorstore to {args.vectordb_dir}...")
            vectorstore.save_local(args.vectordb_dir)
        else:  # chroma
            client = chromadb.PersistentClient(path=args.vectordb_dir)
            
            # Create vectorstore for chunks
            vectorstore = Chroma(
                client=client,
                embedding_function=embeddings,
                collection_name="email_chunks"
            )
            
            # Add chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
                vectorstore.add_documents(batch)
            
            # Initialize SQL store for full documents
            email_store = EmailStore(args.sqldb_path)
            
            # Add full documents to SQL database
            print(f"Adding {len(full_docs)} full documents to SQL database...")
            for full_doc_id, full_doc in full_docs.items():
                thread_id, message_index = full_doc_id.split('_')
                email_store.add_document(
                    doc_id=f"full_{full_doc_id}",
                    thread_id=thread_id,
                    message_index=int(message_index),
                    content=full_doc.page_content,
                    metadata=full_doc.metadata
                )
        
        print("Done!")
    except Exception as e:
        print(f"Error processing mbox file: {e}")
        raise

if __name__ == '__main__':
    main()