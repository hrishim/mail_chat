import mailbox
#from email import message_from_bytes
#from email.header import decode_header
from bs4 import BeautifulSoup
import re
import os, sys, shutil
import argparse
from os.path import exists
from pathlib import Path
from typing import Union, Optional, Generator
from mailbox import mboxMessage
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document

class Message:
    def __init__(self, to: str, sender:str, subject: str, date: str, 
                 content: str, x_gmail_labels: list[str]=[],
                 x_gm_thrid: Optional[str]=None, inReplyTo:Optional[str]=None, 
                 references:list[str] = []):
        self.to: str = to
        self.sender: str = sender
        self.subject: str = subject
        self.date: str = date
        self.content: str = content
        self.x_gmail_labels: list[str] = x_gmail_labels
        self.x_gm_thrid: Optional[str] = x_gm_thrid
        self.inReplyTo: Optional[str] = inReplyTo
        self.references: list[str] = references

    def __str__(self):
        return (
            f"Sender: {self.sender}\n"
            f"To: {self.to}\n"
            f"Subject: {self.subject}\n"
            f"Date: {self.date}\n"
            f"x_gmail_labels: {self.x_gmail_labels}\n"
            #f"x_gm_thrid: {self.x_gm_thrid}\n"
            f"inReplyTo: {self.inReplyTo}\n"
            f"References: {self.references}\n"
            f"Content: {self.content}"
        )

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

def parse_email_date(date_str: str) -> str:
    """Parse email date string into a consistent format.
    
    Handles various email date formats and converts to our standard format.
    
    Args:
        date_str: Date string from email header
        
    Returns:
        Date string in format 'YYYY-MM-DD HH:MM:SS±HHMM'
    """
    # Clean up the date string
    date_str = date_str.split('(')[0].strip()  # Remove (UTC) or similar parenthetical
    
    # Try various email date formats
    for fmt in [
        '%a, %d %b %Y %H:%M:%S %z',  # Standard email format: 'Sun, 09 Feb 2025 09:37:31 -0800'
        '%d %b %Y %H:%M:%S %z',      # Without weekday
        '%a, %d %b %Y %H:%M:%S %Z',  # With timezone name
        '%a, %d %b %Y %H:%M:%S',     # Without timezone
    ]:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                # If no timezone, assume UTC
                return dt.strftime('%Y-%m-%d %H:%M:%S+0000')
            return dt.strftime('%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date string: {date_str}")

def parse_date_for_sorting(date_str: str) -> datetime:
    """Parse a date string into a datetime object for sorting.
    
    Handles both our standard format and email format dates.
    
    Args:
        date_str: Date string in either format
        
    Returns:
        datetime object
    """
    # Clean up the date string
    date_str = date_str.split('(')[0].strip()  # Remove (UTC) or similar parenthetical
    
    # Try our standard format first
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
    except ValueError:
        pass
    
    # If that fails, try email formats
    for fmt in [
        '%a, %d %b %Y %H:%M:%S %z',
        '%d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%a, %d %b %Y %H:%M:%S',
    ]:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                # If no timezone, assume UTC
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date string: {date_str}")

def load_mbox(file_path: Union[str, os.PathLike]) -> list[Message]:
    """Loads an mbox file and extracts messages."""
    mbox = mailbox.mbox(file_path)
    messages = []
    
    for msg in mbox:
        try:
            content = extract_content(msg)
            if content:
                content = clean_content(content)
                
                # Get Gmail-specific headers
                gmail_labels = msg.get('X-Gmail-Labels', '').split(',')
                gmail_labels = [label.strip() for label in gmail_labels if label.strip()]
                
                thread_id = msg.get('X-GM-THRID')
                in_reply_to = msg.get('In-Reply-To')
                references = msg.get('References', '').split()
                
                # Parse and standardize the date
                date_str = msg.get('Date')
                if date_str:
                    try:
                        date = parse_email_date(date_str)
                    except ValueError:
                        print(f"Warning: Could not parse date '{date_str}', skipping message")
                        continue
                else:
                    print("Warning: Message has no date, skipping")
                    continue
                
                message = Message(
                    to=msg.get('To', ''),
                    sender=msg.get('From', ''),
                    subject=msg.get('Subject', ''),
                    date=date,
                    content=content,
                    x_gmail_labels=gmail_labels,
                    x_gm_thrid=thread_id,
                    inReplyTo=in_reply_to,
                    references=references
                )
                messages.append(message)
        except Exception as e:
            print(f"Warning: Error processing message: {e}")
            continue
    
    return messages

def load_mbox_in_chunks(file_path: Union[str, os.PathLike], chunk_size: int = 100) -> Generator[list[Message], None, None]:
    """Loads an mbox file and yields messages in chunks."""
    mbox = mailbox.mbox(file_path)
    messages = []
    
    for msg in mbox:
        try:
            content = extract_content(msg)
            if content:
                content = clean_content(content)
                
                # Get Gmail-specific headers
                gmail_labels = msg.get('X-Gmail-Labels', '').split(',')
                gmail_labels = [label.strip() for label in gmail_labels if label.strip()]
                
                thread_id = msg.get('X-GM-THRID')
                in_reply_to = msg.get('In-Reply-To')
                references = msg.get('References', '').split()
                
                # Parse and standardize the date
                date_str = msg.get('Date')
                if date_str:
                    try:
                        date = parse_email_date(date_str)
                    except ValueError:
                        print(f"Warning: Could not parse date '{date_str}', skipping message")
                        continue
                else:
                    print("Warning: Message has no date, skipping")
                    continue
                
                message = Message(
                    to=msg.get('To', ''),
                    sender=msg.get('From', ''),
                    subject=msg.get('Subject', ''),
                    date=date,
                    content=content,
                    x_gmail_labels=gmail_labels,
                    x_gm_thrid=thread_id,
                    inReplyTo=in_reply_to,
                    references=references
                )
                messages.append(message)
                
                if len(messages) >= chunk_size:
                    yield messages
                    messages = []
        except Exception as e:
            print(f"Warning: Error processing message: {e}")
            continue
    
    if messages:
        yield messages

def chunk_thread_messages(thread_messages: list[Message], chunk_size: int = 400, chunk_overlap: int = 100) -> list[Document]:
    """Chunks messages from a single thread while maintaining conversation context.
    
    This function is optimized for RAG by keeping related messages together in chunks.
    Messages in the same thread are concatenated with clear separators before chunking.
    
    Args:
        thread_messages: List of messages in a thread, assumed to be sorted by date
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of Document objects containing chunks with their metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\nFrom:", "\nFrom:", "\n\n", "\n", " ", ""]
    )
    
    # Join messages with clear separators
    thread_text = ""
    for i, msg in enumerate(thread_messages):
        thread_text += str(msg)
        if i < len(thread_messages) - 1:
            thread_text += "\n\n---Next Message in Thread---\n\n"
    
    # Create metadata for the thread
    metadata = {
        'thread_start_date': thread_messages[0].date,
        'thread_end_date': thread_messages[-1].date,
        'num_messages': len(thread_messages),
        'participants': list(set([msg.sender for msg in thread_messages] + [msg.to for msg in thread_messages]))
    }
    
    # Create documents with metadata
    return text_splitter.create_documents([thread_text], [metadata])

def print_msg_keys(file_path: os.PathLike) -> None:
    mbox = mailbox.mbox(file_path)
    first_message = next(iter(mbox.values()))
    print(first_message.keys())
   
    #print(mbox[0])
    #for i, msg in enumerate(mbox):
        #print(f"---------- Message {i} ---------- ")
        #print(msg)
    

def print_messages(file_path: Union[str, os.PathLike]) -> None:
    messages = load_mbox(file_path)

    for i, msg in enumerate(messages):
        print(msg)
    return
    for i, msg in enumerate(messages):
        print(f"Message {i+1}:")
        print(f"From: {msg.sender}")
        print(f"Subject: {msg.subject}")
        print(f"Date: {msg.date}")
        print(f"Content: {msg.content}")
        print(f"X-Gmail-Labels: {msg.x_gmail_labels}")
        print(f"X-GM-THRID: {msg.x_gm_thrid}")
        print(f"In-Reply-To: {msg.inReplyTo}")
        print(f"References: {msg.references}")

        print("**************************************")


def organize_messages(messages: list[Message]) -> dict[str, list[Message]]:
    """Organize messages by thread ID."""
    org_msgs: dict[str, list[Message]] = {}

    for msg in messages:
        if not msg.x_gm_thrid:
            if 'orphan' not in org_msgs:
                org_msgs['orphan'] = []
            org_msgs['orphan'].append(msg)
        else:
            org_msgs[msg.x_gm_thrid] = org_msgs.get(msg.x_gm_thrid, []) + [msg]

    # Sorting each list in the dictionary by the 'date' field
    for key, thread_messages in org_msgs.items():
        org_msgs[key] = sorted(thread_messages, key=lambda msg: parse_date_for_sorting(msg.date))
    
    return org_msgs

#file_path = "AllMailIncludingSpamAndTrash.mbox"
#mbox = mailbox.mbox(file_path)
#print(type(mbox))
#first_message = next(iter(mbox.values()))
#print(type(first_message))
#exit(0)

# Example usage
#file_path = "AllMailIncludingSpamAndTrash.mbox"
#print_messages(file_path=file_path)


# Example usage with partial loading
#file_path = "AllMailIncludingSpamAndTrash.mbox"
#chunk_size = 100  # Adjust the chunk size as needed
#
#for message_chunk in load_mbox_in_chunks(file_path, chunk_size):
    #chunks = chunk_messages(message_chunk, chunk_size=1000, chunk_overlap=200)
    #for chunk in chunks:
        #print(chunk)
        #print("--------")
    #print("**************************************")


def load_and_organize_in_chunks(file_path: Union[str, os.PathLike], chunk_size: int = 100) -> Generator[dict[str, list[Message]], None, None]:
    """Loads messages from mbox in chunks and organizes them by thread.
    Uses a buffer to accumulate messages from the same thread before yielding.
    
    Args:
        file_path: Path to the mbox file
        chunk_size: Number of threads to accumulate before yielding
        
    Yields:
        Dictionary mapping thread IDs to lists of messages in that thread
    """
    mbox = mailbox.mbox(file_path)
    thread_buffer: dict[str, list[Message]] = {}
    orphan_messages: list[Message] = []
    threads_processed = 0

    def safe_strip(s: str) -> str:
        return s.strip("\r") if s is not None else ""

    for message in mbox:
        # Extract message data (reusing code from load_mbox)
        sender = safe_strip(message['From'])
        to = safe_strip(message['To'])
        subject = safe_strip(message['Subject'])
        date = message['Date']
        xgmThrid = safe_strip(message['X-GM-THRID'])
        igt = message.get('In-Reply-To')
        inReplyTo = safe_strip(igt) if igt else ""
        refs = message.get("References")
        references = refs.split() if refs else []
        labels = message.get("X-Gmail-Labels")
        x_gmail_labels: list[str] = labels.split(',') if labels else [] 

        # Extract content
        content = extract_content(message)
        content = clean_content(content)
        
        # Create Message object
        msg_obj = Message(to=to, sender=sender, subject=subject, date=date, 
                         content=content, x_gmail_labels=x_gmail_labels, 
                         x_gm_thrid=xgmThrid, inReplyTo=inReplyTo, 
                         references=references)

        # Add message to appropriate thread in buffer
        if xgmThrid:
            if xgmThrid not in thread_buffer:
                thread_buffer[xgmThrid] = []
                threads_processed += 1
            thread_buffer[xgmThrid].append(msg_obj)
        else:
            orphan_messages.append(msg_obj)

        # When we've accumulated enough threads, sort them by date and yield
        if threads_processed >= chunk_size:
            # Sort messages in each thread by date
            for thread_messages in thread_buffer.values():
                thread_messages.sort(key=lambda msg: parse_date_for_sorting(msg.date))
            
            # If we have orphan messages, add them as a special thread
            if orphan_messages:
                thread_buffer['orphan'] = sorted(orphan_messages, 
                                               key=lambda msg: parse_date_for_sorting(msg.date))
            
            yield thread_buffer
            
            # Reset buffers
            thread_buffer = {}
            orphan_messages = []
            threads_processed = 0

    # Yield remaining messages
    if thread_buffer or orphan_messages:
        # Sort remaining threads
        for thread_messages in thread_buffer.values():
            thread_messages.sort(key=lambda msg: parse_date_for_sorting(msg.date))
        
        if orphan_messages:
            thread_buffer['orphan'] = sorted(orphan_messages, 
                                           key=lambda msg: parse_date_for_sorting(msg.date))
        
        yield thread_buffer

def main():
    """Process email threads from mbox file in memory-efficient chunks."""
    parser = argparse.ArgumentParser(description='Process mbox file into threaded chunks for RAG.')
    parser.add_argument('mbox_file', type=str, help='Path to the mbox file to process')
    parser.add_argument('--chunk-size', type=int, default=50,
                       help='Number of threads to process in each batch (default: 50)')
    parser.add_argument('--text-chunk-size', type=int, default=400,
                       help='Maximum size of each text chunk in characters (default: 400, recommended for NV-Embed-QA)')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='Number of characters to overlap between chunks (default: 100)')
    parser.add_argument('--vectordb-dir', type=str, default='./mail_vectordb',
                       help='Directory to store the vector database (default: ./mail_vectordb)')
    parser.add_argument('--nvidia-api-key', type=str, default=None,
                       help='NVIDIA AI Endpoints API key. Can also be set via NVIDIA_API_KEY environment variable.')
    
    args = parser.parse_args()
    
    # Get NVIDIA API key from args or environment
    api_key = args.nvidia_api_key or os.environ.get('NVIDIA_API_KEY')
    if not api_key:
        print("Error: NVIDIA AI Endpoints API key is required.")
        print("You can provide it in one of two ways:")
        print("1. Set the NVIDIA_API_KEY environment variable:")
        print("   export NVIDIA_API_KEY='your-api-key'")
        print("2. Pass it as a command line argument:")
        print("   python data_prep.py --nvidia-api-key 'your-api-key' ./AllMailIncludingSpamAndTrash.mbox")
        sys.exit(1)
    
    # Validate that the mbox file exists
    if not exists(args.mbox_file):
        print(f"Error: Mbox file '{args.mbox_file}' does not exist.")
        sys.exit(1)
    
    # Create vector DB directory if it doesn't exist
    vectordb_dir = Path(args.vectordb_dir)
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize embeddings and vector store
    try:
        embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            truncate="END",
            nvidia_api_key=api_key
        )
    except Exception as e:
        print(f"Error initializing NVIDIA AI Endpoints: {e}")
        print("Please check your API key and try again.")
        sys.exit(1)
        
    vectorstore = None
    
    try:
        # Process emails in batches of threads
        for thread_batch in load_and_organize_in_chunks(args.mbox_file, chunk_size=args.chunk_size):
            print(f"\nProcessing new batch of up to {args.chunk_size} threads...")
            
            batch_documents = []
            
            for thread_id, messages in thread_batch.items():
                # Create chunks that maintain thread context
                documents = chunk_thread_messages(
                    messages,
                    chunk_size=args.text_chunk_size,
                    chunk_overlap=args.chunk_overlap
                )
                
                # Add thread_id to each document's metadata
                for i, doc in enumerate(documents):
                    doc.metadata['thread_id'] = thread_id
                    doc.metadata['chunk_index'] = i
                    doc.metadata['total_chunks'] = len(documents)
                    batch_documents.append(doc)
            
            # Create or update vector store
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    documents=batch_documents,
                    embedding=embeddings
                )
            else:
                vectorstore.add_documents(batch_documents)
            
            # Save after each batch to prevent data loss
            vectorstore.save_local(str(vectordb_dir))
            print(f"Saved vector store with {len(batch_documents)} new chunks")
                
    except Exception as e:
        print(f"Error processing mbox file: {e}")
        if vectorstore is not None:
            # Save what we have in case of error
            vectorstore.save_local(str(vectordb_dir))
        sys.exit(1)
    
    print(f"\nProcessing complete. Vector store saved to {vectordb_dir}")

if __name__ == '__main__':
    main()