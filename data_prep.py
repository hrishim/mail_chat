import mailbox
from email import message_from_bytes
from email.header import decode_header, make_header
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
from dotenv import load_dotenv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json

# Load environment variables from .env file
load_dotenv()

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
        return (
            f"Sender: {self.sender}\n"
            f"To: {self.to}\n"
            f"Subject: {self.subject}\n"
            f"Date: {self.date}\n"
            f"x_gmail_labels: {self.x_gmail_labels}\n"
            #f"x_gm_thrid: {self.x_gm_thrid}\n"
            f"inReplyTo: {self.inReplyTo}\n"
            f"References: {self.references}\n"
            f"Message-ID: {self.message_id}\n"
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

@lru_cache(maxsize=10000)
def parse_email_date(date_str: str) -> str:
    """Parse email date string into a consistent format.
    
    Handles various email date formats and converts to our standard format.
    
    Args:
        date_str: Date string from email header
        
    Returns:
        Date string in format 'YYYY-MM-DD HH:MM:SS±HHMM'
    """
    if not date_str:
        raise ValueError("Empty date string")

    # Clean up parenthetical timezone names
    date_str = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)

    # Map common timezone names to their UTC offsets
    tz_map = {
        'EDT': '-0400',  # Eastern Daylight Time
        'EST': '-0500',  # Eastern Standard Time
        'CDT': '-0500',  # Central Daylight Time
        'CST': '-0600',  # Central Standard Time
        'MDT': '-0600',  # Mountain Daylight Time
        'MST': '-0700',  # Mountain Standard Time
        'PDT': '-0700',  # Pacific Daylight Time
        'PST': '-0800',  # Pacific Standard Time
        'IST': '+0530',  # Indian Standard Time
    }
    
    # Replace timezone names with their offsets
    for tz_name, offset in tz_map.items():
        if f" {tz_name}" in date_str:
            date_str = date_str.replace(f" {tz_name}", f" {offset}")
            break

    # Handle non-standard timezone offsets
    if (match := re.search(r'([+-])(\d{2}):?(\d{2})$', date_str)):
        sign, hours, mins = match.groups()
        if int(mins) > 30:
            hours = str(int(hours) + 1).zfill(2)
            mins = "00"
        else:
            mins = "30"
        date_str = re.sub(r'[+-]\d{2}:?\d{2}$', f'{sign}{hours}{mins}', date_str)
    
    # Try various email date formats
    for fmt in [
        '%a, %d %b %Y %H:%M:%S %z',  # Standard email format: 'Sun, 09 Feb 2025 09:37:31 -0800'
        '%d %b %Y %H:%M:%S %z',      # Without weekday
        '%a, %d %b %Y %H:%M:%S %Z',  # With timezone name
        '%a, %d %b %Y %H:%M:%S',     # Without timezone
        '%a %b %d %H:%M:%S %Y %z',   # Unix style with timezone: 'Fri Apr 17 00:16:47 2015 +0530'
        '%a %b %d %H:%M:%S %Y',      # Unix style: 'Thu Mar 19 14:16:11 2025'
        '%a, %d %b %Y %H:%M %z',     # Without seconds
        '%d %b %Y %H:%M %z',         # Without seconds and weekday
        '%d %b %y %H:%M:%S',         # Short year format: '06 Apr 15 21:57:41'
        '%d %b %Y %H:%M:%S',         # Same but with full year
        '%d %b %y %H:%M %z',         # Short year with timezone: '30 Jun 13 07:26 -0800'
    ]:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                # If no timezone, assume UTC
                return dt.strftime('%Y-%m-%d %H:%M:%S+0000')
            return dt.strftime('%Y-%m-%d %H:%M:%S%z')
        except ValueError:
            continue
    
    print(f"WARNING: Failed to parse date string: '{date_str}'")
    print(f"Date string length: {len(date_str)}")
    print(f"Date string bytes: {date_str.encode()}")
    raise ValueError(f"Could not parse date string: '{date_str}'")

@lru_cache(maxsize=10000)
def parse_date_for_sorting(date_str: str) -> datetime:
    """Parse date string into datetime object for sorting."""
    if not date_str:
        print("WARNING: Empty date string in parse_date_for_sorting")
        raise ValueError("Empty date string")

    # Clean up parenthetical timezone names
    date_str = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)

    # Map common timezone names to their UTC offsets
    tz_map = {
        'EDT': '-0400',  # Eastern Daylight Time
        'EST': '-0500',  # Eastern Standard Time
        'CDT': '-0500',  # Central Daylight Time
        'CST': '-0600',  # Central Standard Time
        'MDT': '-0600',  # Mountain Daylight Time
        'MST': '-0700',  # Mountain Standard Time
        'PDT': '-0700',  # Pacific Daylight Time
        'PST': '-0800',  # Pacific Standard Time
        'IST': '+0530',  # Indian Standard Time
    }
    
    # Replace timezone names with their offsets
    for tz_name, offset in tz_map.items():
        if f" {tz_name}" in date_str:
            date_str = date_str.replace(f" {tz_name}", f" {offset}")
            break

    # Handle non-standard timezone offsets
    if (match := re.search(r'([+-])(\d{2}):?(\d{2})$', date_str)):
        sign, hours, mins = match.groups()
        if int(mins) > 30:
            hours = str(int(hours) + 1).zfill(2)
            mins = "00"
        else:
            mins = "30"
        date_str = re.sub(r'[+-]\d{2}:?\d{2}$', f'{sign}{hours}{mins}', date_str)

    # Clean up whitespace
    date_str = re.sub(r'\s+', ' ', date_str)

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
        '%a %b %d %H:%M:%S %Y %z',   # Unix style with timezone: 'Fri Apr 17 00:16:47 2015 +0530'
        '%a %b %d %H:%M:%S %Y',      # Unix style: 'Thu Mar 19 14:16:11 2025'
        '%a, %d %b %Y %H:%M %z',     # Without seconds
        '%d %b %Y %H:%M %z',         # Without seconds and weekday
        '%d %b %y %H:%M:%S',         # Short year format: '06 Apr 15 21:57:41'
        '%d %b %Y %H:%M:%S',         # Same but with full year
        '%d %b %y %H:%M %z',         # Short year with timezone: '30 Jun 13 07:26 -0800'
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
    
    # Try special handling for GMT+offset format
    gmt_match = re.match(r'(\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2}) GMT([+-]\d{2}):?(\d{2}) (\d{4})', date_str)
    if gmt_match:
        try:
            base_str, tz_hour, tz_min, year = gmt_match.groups()
            # Convert to standard format with offset
            reformatted = f"{base_str} {year} {tz_hour}{tz_min}"
            dt = datetime.strptime(reformatted, '%a %b %d %H:%M:%S %Y %z')
            return dt
        except ValueError:
            pass
    
    print(f"WARNING: Failed to parse date string in parse_date_for_sorting: '{date_str}'")
    print(f"Date string length: {len(date_str)}")
    print(f"Date string bytes: {date_str.encode()}")
    raise ValueError(f"Could not parse date string: '{date_str}'")

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
                gmail_labels = str(msg.get('X-Gmail-Labels', '')).split(',')
                gmail_labels = [label.strip() for label in gmail_labels if label.strip()]
                
                thread_id = str(msg.get('X-GM-THRID', ''))
                in_reply_to = str(msg.get('In-Reply-To', ''))
                references = str(msg.get('References', '')).split()
                message_id = str(msg.get('Message-ID', ''))
                
                # Parse and standardize the date
                date_str = str(msg.get('Date', ''))
                if date_str:
                    try:
                        date = parse_email_date(date_str)
                    except ValueError:
                        print(f"Warning: Could not parse date '{date_str}', skipping message")
                        continue
                else:
                    print("Warning: Message has no date, skipping")
                    continue

                # Properly decode headers
                def decode_header_str(header):
                    if not header:
                        return ''
                    try:
                        return str(make_header(decode_header(str(header))))
                    except:
                        return str(header)
                
                message = Message(
                    to=decode_header_str(msg.get('To')),
                    sender=decode_header_str(msg.get('From')),
                    subject=decode_header_str(msg.get('Subject')),
                    date=date,
                    content=content,
                    x_gmail_labels=gmail_labels,
                    x_gm_thrid=thread_id,
                    inReplyTo=in_reply_to,
                    references=references,
                    message_id=message_id
                )
                messages.append(message)
        except Exception as e:
            print(f"Warning: Error processing message: {e}")
            continue
    
    return messages

def load_mbox_in_chunks(file_path: Union[str, os.PathLike], batch_size: int = 100) -> Generator[list[Message], None, None]:
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
                message_id = msg.get('Message-ID', '')
                
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
                    references=references,
                    message_id=message_id
                )
                messages.append(message)
                
                if len(messages) >= batch_size:
                    yield messages
                    messages = []
        except Exception as e:
            print(f"Warning: Error processing message: {e}")
            continue
    
    if messages:
        yield messages

def chunk_thread_messages(thread_messages: list[Message], text_chunk_size: int = 400, chunk_overlap: int = 100) -> list[Document]:
    """Chunks messages from a single thread while maintaining conversation context.
    
    This function is optimized for RAG by keeping related messages together in chunks.
    Messages in the same thread are concatenated with clear separators before chunking.
    
    Args:
        thread_messages: List of messages in a thread, assumed to be sorted by date
        text_chunk_size: Maximum size of each text chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of Document objects containing chunks with their metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_chunk_size,
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
        print(f"Message-ID: {msg.message_id}")

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

def load_and_organize_in_chunks(file_path: Union[str, os.PathLike], threads_per_batch: int = 500, start_idx: int = 0) -> Generator[dict[str, list[Message]], None, None]:
    """Loads messages from mbox in chunks and organizes them by thread.
    Uses a buffer to accumulate messages from the same thread before yielding.
    
    Args:
        file_path: Path to the mbox file
        threads_per_batch: Number of threads to accumulate before yielding (default: 500)
        start_idx: Index of first message to process (for resuming)
        
    Yields:
        Dictionary mapping thread IDs to lists of messages in that thread
    """
    mbox = mailbox.mbox(file_path)
    thread_buffer: dict[str, list[Message]] = {}
    orphan_messages: list[Message] = []
    threads_processed = 0

    def safe_strip(s) -> str:
        """Safely decode and strip email headers."""
        if s is None:
            return ""
        try:
            # Convert Header objects to string and decode
            return str(make_header(decode_header(str(s)))).strip("\r")
        except:
            # Fallback to simple string conversion
            return str(s).strip("\r")

    for idx, message in enumerate(mbox):
        if idx < start_idx:
            continue

        # Extract message data (reusing code from load_mbox)
        sender = safe_strip(message['From'])
        to = safe_strip(message['To'])
        subject = safe_strip(message['Subject'])
        date = safe_strip(message['Date'])
        if not date:
            print(f"WARNING: Skipping message with empty date. Subject: '{subject}'")
            continue

        xgmThrid = safe_strip(message['X-GM-THRID'])
        igt = message.get('In-Reply-To')
        inReplyTo = safe_strip(igt) if igt else ""
        refs = message.get("References")
        references = refs.split() if refs else [] 

        # Extract content
        content = extract_content(message)
        content = clean_content(content)
        
        # Get Message-ID from header
        message_id = message.get('Message-ID', '')
        
        # Create Message object
        msg_obj = Message(to=to, sender=sender, subject=subject, date=date, 
                         content=content, x_gmail_labels=[], 
                         x_gm_thrid=xgmThrid, inReplyTo=inReplyTo, 
                         references=references, message_id=message_id)

        # Add message to appropriate thread in buffer
        if xgmThrid:
            if xgmThrid not in thread_buffer:
                thread_buffer[xgmThrid] = []
                threads_processed += 1
            thread_buffer[xgmThrid].append(msg_obj)
        else:
            orphan_messages.append(msg_obj)

        # When we've accumulated enough threads, sort them by date and yield
        if threads_processed >= threads_per_batch:
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

def process_thread_batch(thread_batch: dict[str, list[Message]], text_chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Process a single batch of threads into documents.
    This function runs in a separate process."""
    batch_documents = []
    
    for thread_id, messages in thread_batch.items():
        # Create chunks that maintain thread context
        documents = chunk_thread_messages(
            messages,
            text_chunk_size=text_chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Add thread_id to each document's metadata
        for i, doc in enumerate(documents):
            doc.metadata['thread_id'] = thread_id
            doc.metadata['chunk_index'] = i
            doc.metadata['total_chunks'] = len(documents)
            batch_documents.append(doc)
    
    return batch_documents

def main():
    """Process email threads from mbox file in memory-efficient chunks."""
    parser = argparse.ArgumentParser(description='Process mbox file into threaded chunks for RAG.')
    parser.add_argument('mbox_file', type=str, help='Path to the mbox file to process')
    parser.add_argument('--vectordb-dir', required=True,
                       help='Directory to save vector store in')
    parser.add_argument('--threads-per-batch', type=int, default=500,
                       help='Number of email threads to process in each batch (default: 500)')
    parser.add_argument('--text-chunk-size', type=int, default=400,
                       help='Maximum size of each text chunk in characters (default: 400, recommended for NV-Embed-QA)')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                       help='Number of characters to overlap between chunks (default: 100)')
    parser.add_argument('--save-frequency', type=int, default=5,
                       help='Save vector store every N batches (default: 5)')
    parser.add_argument('--num-processes', type=int, default=11,
                       help='Number of processes to use (default: 11)')
    parser.add_argument('--checkpoint-file', type=str, default='processing_checkpoint.json',
                       help='File to store processing checkpoint (default: processing_checkpoint.json)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing from last checkpoint')
    
    args = parser.parse_args()
    
    # Check for NVIDIA AI Endpoints API key
    api_key = os.getenv('NGC_API_KEY')
    if not api_key:
        print("Error: NGC_API_KEY environment variable not set")
        print("Please set your NVIDIA AI Endpoints API key first")
        sys.exit(1)
    
    # Create vector DB directory if it doesn't exist
    vectordb_dir = Path(args.vectordb_dir)
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    
    # Set number of processes
    if args.num_processes is None:
        args.num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Initialize embeddings
    try:
        embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",
            truncate="END",
            api_key=api_key,
            api_url="https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/e1c06c8d-f614-4af5-9e76-5f5d6d574e23"
        )
    except Exception as e:
        print(f"Error initializing NVIDIA AI Endpoints: {e}")
        print("Please check your API key and try again.")
        sys.exit(1)
    
    # Load existing vector store if it exists
    vectorstore = None
    if vectordb_dir.exists() and any(vectordb_dir.iterdir()):
        print(f"Loading existing vector store from {vectordb_dir}...")
        try:
            vectorstore = FAISS.load_local(
                str(vectordb_dir),
                embeddings,
                allow_dangerous_deserialization=True  # Safe since we created these files
            )
            print("Existing vector store loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load existing vector store: {e}")
            print("Starting fresh...")
    
    # Load or initialize checkpoint
    checkpoint_path = Path(args.checkpoint_file)
    checkpoint = {}
    if args.resume and checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                print(f"Resuming from checkpoint: {checkpoint['emails_processed']} emails processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Starting from beginning...")
            checkpoint = {'emails_processed': 0, 'last_message_id': None}
    else:
        checkpoint = {'emails_processed': 0, 'last_message_id': None}
    
    total_emails_processed = checkpoint['emails_processed']
    total_chunks_processed = 0
    batches_since_save = 0
    
    try:
        # Create a process pool
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            thread_batches = []
            futures = []
            
            # Process emails in batches of threads
            mbox = mailbox.mbox(args.mbox_file)
            
            # Skip to last processed message if resuming
            start_idx = 0
            if args.resume and checkpoint['last_message_id'] is not None:
                for idx, msg in enumerate(mbox):
                    if msg.get('Message-ID') == checkpoint['last_message_id']:
                        start_idx = idx + 1
                        print(f"Resuming from message {start_idx}")
                        break
            
            for thread_batch in load_and_organize_in_chunks(args.mbox_file, threads_per_batch=args.threads_per_batch, start_idx=start_idx):
                batch_email_count = sum(len(messages) for messages in thread_batch.values())
                total_emails_processed += batch_email_count
                
                # Update checkpoint
                last_message = None
                for messages in thread_batch.values():
                    if messages:
                        last_message = messages[-1]
                if last_message:
                    checkpoint['emails_processed'] = total_emails_processed
                    checkpoint['last_message_id'] = last_message.message_id
                    with open(checkpoint_path, 'w') as f:
                        json.dump(checkpoint, f)
                
                num_threads = len(thread_batch)
                print(f"\nCollected new batch:")
                print(f"├── Threads: {num_threads}")
                print(f"└── Emails: {batch_email_count} (avg {batch_email_count/num_threads:.1f} per thread)")
                print(f"\nProgress:")
                print(f"├── Total threads processed: {total_emails_processed//args.threads_per_batch * args.threads_per_batch}")
                print(f"└── Total emails processed: {total_emails_processed}")
                
                # Submit batch for parallel processing
                print(f"\nProcessing with {args.num_processes} workers...")
                
                future = executor.submit(
                    process_thread_batch,
                    thread_batch,
                    args.text_chunk_size,
                    args.chunk_overlap
                )
                futures.append(future)
                thread_batches.append(thread_batch)
                
                # Process results when we have enough batches or have completed futures
                if len(futures) >= args.num_processes:
                    # Process any completed futures
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            batch_documents = future.result()
                            
                            # Create or update vector store
                            if vectorstore is None:
                                vectorstore = FAISS.from_documents(
                                    documents=batch_documents,
                                    embedding=embeddings
                                )
                            else:
                                vectorstore.add_documents(batch_documents)
                            
                            total_chunks_processed += len(batch_documents)
                            print(f"Processed {len(batch_documents)} chunks")
                            print(f"Total chunks processed so far: {total_chunks_processed}")
                            
                            batches_since_save += 1
                    
                    # Remove processed futures
                    for future in done_futures:
                        futures.remove(future)
                        
                    # Save periodically to prevent data loss
                    if batches_since_save >= args.save_frequency:
                        print("Saving vector store...")
                        vectorstore.save_local(str(vectordb_dir))
                        batches_since_save = 0
            
            # Process any remaining futures
            for future in futures:
                batch_documents = future.result()
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(
                        documents=batch_documents,
                        embedding=embeddings
                    )
                else:
                    vectorstore.add_documents(batch_documents)
                
                total_chunks_processed += len(batch_documents)
                print(f"Processed {len(batch_documents)} chunks")
                print(f"Total chunks processed so far: {total_chunks_processed}")
        
        # Final save
        if vectorstore is not None:
            print("Saving final vector store state...")
            vectorstore.save_local(str(vectordb_dir))
        
        # Clear checkpoint on successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Processing completed successfully, checkpoint cleared")
                
    except Exception as e:
        print(f"Error processing mbox file: {e}")
        # Save progress on error
        if vectorstore is not None:
            print("Saving vector store state before exit...")
            vectorstore.save_local(str(vectordb_dir))
        raise

if __name__ == '__main__':
    main()