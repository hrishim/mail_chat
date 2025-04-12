from multiprocessing import Pool, cpu_count
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
from mailbox import mboxMessage
from typing import Optional, Generator, Union, Literal, Any, Dict
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
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from date_utils import TZ_MAP, DATE_FORMATS

# Load environment variables from .env file
load_dotenv()

# Initialize text splitter with conservative settings for embedding
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Reduced to stay under 512 token limit
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
                    date = parse_email_date(date_str)
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
    
    # Try to handle common variations
    date_str = date_str.strip()
    
    # Clean up parenthetical timezone names
    date_str = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)
    
    # Handle Unix-style timestamps with timezone but no space
    # e.g., "Thu Apr 16 20:59:04 2015+0530" -> "Thu Apr 16 20:59:04 2015 +0530"
    unix_tz_match = re.search(r'(\d{4})[+-]\d{4}$', date_str)
    if unix_tz_match:
        year_pos = date_str.find(unix_tz_match.group(1))
        if year_pos != -1:
            year_end = year_pos + 4
            date_str = date_str[:year_end] + ' ' + date_str[year_end:]
    
    # Handle UT timezone
    if ' UT' in date_str:
        date_str = date_str.replace(' UT', ' +0000')
    
    # Handle GMT+HHMM format
    if 'GMT+' in date_str:
        parts = date_str.split('GMT+')
        if len(parts) == 2:
            date_str = parts[0] + '+' + parts[1]
    
    # Map common timezone names to their UTC offsets
    tz_map = TZ_MAP
    
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
    
    # List of date formats to try
    formats = DATE_FORMATS
    
    # Try each format
    for fmt in formats:
        try:
            # Parse with current format
            dt = datetime.strptime(date_str, fmt)
            
            # If timezone is naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Convert to our standard format
            return dt.strftime('%Y-%m-%d %H:%M:%S%z')
            
        except ValueError:
            continue
            
    # If we get here, none of our formats worked
    raise ValueError(f"Failed to parse date string: '{date_str}'")

def log_date_error(date_str: str, error_type: str = "Failed to parse") -> None:
    """Log date parsing errors to file with absolute path."""
    try:
        error_log_path = Path(__file__).parent / 'date_fmt_error.txt'
        with open(error_log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {error_type}: '{date_str}'\n")
            f.write(f"  Tried formats:\n")
            for fmt in [
                '%Y-%m-%d %H:%M:%S%z',  # Our standard format
                '%a, %d %b %Y %H:%M:%S %z',
                '%d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S',
                '%a %b %d %H:%M:%S %Y %z',
                '%a %b %d %H:%M:%S %Y',
                '%a, %d %b %Y %H:%M %z',
                '%d %b %Y %H:%M %z',
                '%d %b %y %H:%M:%S',
                '%d %b %Y %H:%M:%S',
                '%d %b %y %H:%M %z',
            ]:
                f.write(f"    - {fmt}\n")
            f.write("\n")
            f.flush()  # Force write to disk
    except Exception as e:
        print(f"Error writing to date_fmt_error.txt: {e}")

@lru_cache(maxsize=10000)
def parse_date_for_sorting(date_str: str) -> Optional[datetime]:
    """Parse date string into datetime object for sorting. Returns None if parsing fails."""
    if not date_str:
        print("WARNING: Empty date string in parse_date_for_sorting")
        log_date_error(date_str, "Empty date string")
        return None

    # Clean up parenthetical timezone names and extra spaces
    date_str = re.sub(r'\s*\([^)]+\)\s*$', '', date_str)  # Remove (IST) etc.
    date_str = re.sub(r'\s+', ' ', date_str)  # Normalize spaces
    date_str = date_str.strip()

    # Replace timezone names with their offsets
    for tz_name, offset in TZ_MAP.items():
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
    
    # Try each format
    for fmt in DATE_FORMATS:
        try:
            # Parse with current format
            dt = datetime.strptime(date_str, fmt)
            
            # If timezone is naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt
            
        except ValueError:
            continue
            
    # Log the error and return None
    print(f"WARNING: Failed to parse date string in parse_date_for_sorting: '{date_str}'")
    log_date_error(date_str)
    return None

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
                if not date_str:
                    print("Warning: Message has no date, skipping")
                    continue
                
                try:
                    date = parse_email_date(date_str)
                except ValueError:
                    print(f"Warning: Could not parse date '{date_str}', skipping message")
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

def get_thread_batches(mbox_file: str, max_emails: Optional[int] = None) -> Generator[list[list[Message]], None, None]:
    """Get batches of threads from mbox file."""
    current_batch = {}  # thread_id -> list[Message]
    email_count = 0
    
    try:
        mbox = mailbox.mbox(mbox_file)
        for idx, message in enumerate(mbox):
            if max_emails and email_count >= max_emails:
                print(f"Reached max emails limit ({max_emails})")
                break
                
            try:
                msg = Message.from_email_message(message)
                if msg is None:
                    continue
                
                email_count += 1
                thread_id = msg.x_gm_thrid
                if thread_id not in current_batch:
                    current_batch[thread_id] = []
                current_batch[thread_id].append(msg)
                
                # Yield batch when it gets large enough
                if len(current_batch) >= 100:  # Process 100 threads at a time
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

    except Exception as e:
        print(f"Error processing mbox file: {e}")
        raise

def load_and_organize_in_chunks(file_path: Union[str, os.PathLike], threads_per_batch: int = 500, start_idx: int = 0) -> Generator[list[list[Message]], None, None]:
    """Load messages from mbox file and organize them by thread, yielding batches."""
    mbox = mailbox.mbox(file_path)
    current_batch: dict[str, list[Message]] = {}
    idx = -1
    
    for message in mbox:
        idx += 1
        if idx < start_idx:
            continue
            
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
        thread_end_date = thread_messages[-1].date
        participants = list(set([msg.sender for msg in thread_messages] + [msg.to for msg in thread_messages]))
        
        for i, message in enumerate(thread_messages):
            # Get base metadata from message
            metadata = message.get_metadata()
            
            # Add thread context
            metadata.update({
                'thread_id': thread_id,
                'thread_start_date': thread_start_date,
                'thread_end_date': thread_end_date,
                'thread_position': i,
                'thread_length': len(thread_messages),
                'participants': participants,
                'is_chunk': False,  # Flag to indicate this is a full message
                'full_content_id': f"{thread_id}_{i}"  # ID to link chunks back to full content
            })
            
            # Create document with full content
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
                
                chunk_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(chunk_doc)
    
    return documents

def prepare_chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare metadata for Chroma by converting lists to strings and filtering complex types.
    
    Args:
        metadata: Original metadata dictionary
        
    Returns:
        Filtered metadata dictionary with only simple types
    """
    # Convert lists to strings and filter out complex types
    converted = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            converted[key] = ', '.join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            converted[key] = value
            
    return converted

def create_chroma_document(doc: Document) -> Document:
    """Create a new document with Chroma-compatible metadata.
    
    Args:
        doc: Original document
        
    Returns:
        New document with filtered metadata
    """
    return Document(
        page_content=doc.page_content,
        metadata=prepare_chroma_metadata(doc.metadata)
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Process email threads from mbox file.')
    parser.add_argument('mbox_file', help='Path to mbox file')
    parser.add_argument('--vectordb-dir', default='./mail_vectordb',
                      help='Directory to store vector database')
    parser.add_argument('--db-type', choices=['faiss', 'chroma'], default='faiss',
                      help='Type of vector database to use (faiss or chroma)')
    parser.add_argument('--max-emails', type=int, default=None,
                      help='Maximum number of emails to process (default: all)')
    return parser.parse_args()

def create_vectorstore(db_type: Literal['faiss', 'chroma'], 
                      documents: list[Document], 
                      embeddings,
                      persist_dir: str) -> tuple[VectorStore, Optional[VectorStore]]:
    """Create vector stores based on the specified type.
    
    Args:
        db_type: Type of vector store to create ('faiss' or 'chroma')
        documents: List of documents to add to the store
        embeddings: Embedding function to use
        persist_dir: Directory to persist the vector stores
        
    Returns:
        Tuple of (chunks_store, full_docs_store). For FAISS, full_docs_store is None
        as full documents are stored in the chunks_store's docstore.
    """
    if db_type == 'faiss':
        store = FAISS.from_documents(documents, embeddings)
        return store, None
    else:  # chroma
        # Create Chroma collection for chunks (similar to FAISS approach)
        print("Creating email collection...")
        chunks_store = Chroma.from_documents(
            [create_chroma_document(doc) for doc in documents],
            embeddings,
            persist_directory=persist_dir,
            collection_name="email_collection"
        )
        return chunks_store, None

def add_documents_to_vectorstore(chunks_store: VectorStore,
                               full_docs_store: Optional[VectorStore],
                               documents: list[Document],
                               db_type: Literal['faiss', 'chroma'],
                               persist_dir: str):
    """Add documents to existing vector stores.
    
    Args:
        chunks_store: Store for email chunks
        full_docs_store: Store for full emails (None for FAISS)
        documents: List of documents to add
        db_type: Type of vector store ('faiss' or 'chroma')
        persist_dir: Directory to persist the vector stores
    """
    if db_type == 'faiss':
        chunks_store.add_documents(documents)
        chunks_store.save_local(persist_dir)
    else:  # chroma
        chunks_store.add_documents([create_chroma_document(doc) for doc in documents])
        chunks_store.persist()

def main():
    """Main entry point."""
    args = parse_args()
    vectorstore = None
    all_documents = []  # Keep track of all documents
    total_threads = 0
    
    print(f"Processing mbox file using {args.db_type} database...")
    try:
        # Process messages sequentially
        for thread_batch in get_thread_batches(args.mbox_file, args.max_emails):
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
        
        print(f"Creating {args.db_type} vectorstore with {len(chunks)} chunks...")
        embeddings = NVIDIAEmbeddings(
            model="NV-Embed-QA",  # Suitable for both retrieval and summarization
            api_key=os.getenv("NGC_API_KEY")
        )
        
        # Process chunks in batches to avoid memory issues
        batch_size = 1000
        
        if args.db_type == 'faiss':
            # Original FAISS implementation
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
                
                if vectorstore is None:
                    # First batch - create new vectorstore
                    vectorstore = FAISS.from_documents(batch, embeddings)
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
            # Create Chroma collection for chunks (similar to FAISS approach)
            print("Creating email collection...")
            chunks_store = Chroma.from_documents(
                [create_chroma_document(doc) for doc in chunks[:batch_size]],  # First batch
                embeddings,
                persist_directory=args.vectordb_dir,
                collection_name="email_collection"
            )
            
            # Add remaining chunks in batches
            for i in range(batch_size, len(chunks), batch_size):
                batch = [create_chroma_document(doc) for doc in chunks[i:i+batch_size]]
                print(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
                chunks_store.add_documents(batch)
            
            # Add full documents with special IDs (without embedding)
            print(f"Adding {len(full_docs)} full documents (without embeddings)...")
            
            # Process full documents in batches
            full_docs_list = list(full_docs.items())
            for i in range(0, len(full_docs_list), batch_size):
                batch_items = full_docs_list[i:i+batch_size]
                print(f"Processing full docs batch {i//batch_size + 1} of {(len(full_docs_list)-1)//batch_size + 1}...")
                
                # Add full documents directly to collection without embedding
                ids = []
                metadatas = []
                documents = []
                for full_doc_id, full_doc in batch_items:
                    doc = create_chroma_document(full_doc)
                    ids.append(f"full_{full_doc_id}")
                    metadatas.append(doc.metadata)
                    documents.append(doc.page_content)
                
                # Add to collection without attempting to embed
                chunks_store._collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=[[0.0] * 384] * len(documents)  # Use zero embeddings for full docs
                )
            
            # Save final state
            print(f"Saving final Chroma collection to {args.vectordb_dir}...")
            
            # For error handling
            vectorstore = chunks_store
            
        print("Done!")
        
    except Exception as e:
        print(f"Error processing mbox file: {e}")
        # Save progress on error
        if vectorstore is not None:
            print("Saving vector store state before exit...")
            if args.db_type == 'faiss':
                vectorstore.save_local(args.vectordb_dir)
            else:  # chroma
                vectorstore.persist()
        raise

if __name__ == '__main__':
    main()