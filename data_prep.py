from multiprocessing import Pool, cpu_count
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
from mailbox import mboxMessage
from typing import Optional, Generator, Union
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
from langchain_core.documents import Document

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
    
    # List of date formats to try
    formats = [
        # RFC 2822 and common variants
        '%a, %d %b %Y %H:%M:%S %z',  # Standard email format
        '%d %b %Y %H:%M:%S %z',      # Without weekday
        '%a, %d %b %Y %H:%M:%S %Z',  # With timezone name
        '%a, %d %b %Y %H:%M:%S',     # Without timezone
        '%a, %d %b %Y %H:%M %z',     # Without seconds
        '%d %b %Y %H:%M %z',         # Without seconds and weekday
        
        # Unix style formats
        '%a %b %d %H:%M:%S %Y %z',   # With timezone
        '%a %b %d %H:%M:%S %Y',      # Without timezone
        '%a %b %d %H:%M:%S %z %Y',   # Timezone before year
        
        # Short year formats
        '%d %b %y %H:%M:%S',         # Basic short year
        '%d %b %y %H:%M %z',         # Short year with timezone
        '%a, %d %b %y %H:%M:%S %z',  # Full format with short year
        '%a, %d %b %y %H:%M:%S',     # Without timezone
        
        # ISO formats
        '%Y-%m-%d %H:%M:%S%z',       # With timezone
        '%Y-%m-%d %H:%M:%S',         # Without timezone
        
        # Extra space variations
        '%a,  %d %b %Y %H:%M:%S %z',  # Double space after comma
        '%a,  %d %b %y %H:%M:%S %z',  # Double space with short year
        '%a,  %d %b %Y %H:%M:%S',     # Double space without timezone
        '%a,  %d %b %y %H:%M:%S',     # Double space, short year, no timezone
        
        # Special formats
        '%a %b %d %H:%M:%S GMT%z %Y',  # GMT with offset
        '%a %b %d %H:%M:%S %z %Y',     # Offset before year
    ]
    
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

    # Handle Unix-style timestamps with timezone but no space
    # e.g., "Thu Apr 16 20:59:04 2015+0530" -> "Thu Apr 16 20:59:04 2015 +0530"
    unix_tz_match = re.search(r'(\d{4})[+-]\d{4}$', date_str)
    if unix_tz_match:
        year_pos = date_str.find(unix_tz_match.group(1))
        if year_pos != -1:
            year_end = year_pos + 4
            date_str = date_str[:year_end] + ' ' + date_str[year_end:]

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
        'GMT': '+0000',  # Greenwich Mean Time
        'UT': '+0000',   # Universal Time
    }
    
    # Replace timezone names with their offsets
    for tz_name, offset in tz_map.items():
        if f" {tz_name}" in date_str:
            date_str = date_str.replace(f" {tz_name}", f" {offset}")
            break

    # Handle special case: "Fri May 31 15:00:09 IST 2013" -> "Fri May 31 15:00:09 2013 +0530"
    tz_year_match = re.search(r'(\d{2}:\d{2}:\d{2})\s+([A-Z]{2,4})\s+(\d{4})', date_str)
    if tz_year_match:
        time_str, tz_name, year = tz_year_match.groups()
        if tz_name in tz_map:
            date_str = date_str.replace(f"{time_str} {tz_name} {year}", f"{time_str} {year} {tz_map[tz_name]}")

    # Handle special case: "Thu Nov 07 18:30:38 GMT+05:30 2013" -> "Thu Nov 07 18:30:38 2013 +0530"
    gmt_offset_match = re.search(r'GMT([+-]\d{2}):(\d{2})\s+(\d{4})', date_str)
    if gmt_offset_match:
        hour, minute, year = gmt_offset_match.groups()
        date_str = re.sub(r'GMT([+-]\d{2}):(\d{2})\s+(\d{4})', rf'\3 \1\2', date_str)

    # Handle special case: "Sat Jun 08 12:44:28 GMT+05:30 2013" -> "Sat Jun 08 12:44:28 2013 +0530"
    gmt_offset_no_space = re.search(r'(\d{2}:\d{2}:\d{2})\s*GMT([+-]\d{2}):(\d{2})\s+(\d{4})', date_str)
    if gmt_offset_no_space:
        time_str, hour, minute, year = gmt_offset_no_space.groups()
        date_str = re.sub(r'(\d{2}:\d{2}:\d{2})\s*GMT([+-]\d{2}):(\d{2})\s+(\d{4})', rf'\1 \4 \2\3', date_str)

    # Handle special case: "Fri May 31 15:00:09 +0530 2013" -> "Fri May 31 15:00:09 2013 +0530"
    tz_before_year = re.search(r'(\d{2}:\d{2}:\d{2})\s+([+-]\d{4})\s+(\d{4})', date_str)
    if tz_before_year:
        time_str, tz, year = tz_before_year.groups()
        date_str = date_str.replace(f"{time_str} {tz} {year}", f"{time_str} {year} {tz}")

    # Handle weekday with comma and extra spaces: "Monday,  2 Aug 2004" -> "Mon, 2 Aug 2004"
    weekdays = {
        'Monday': 'Mon',
        'Tuesday': 'Tue',
        'Wednesday': 'Wed',
        'Thursday': 'Thu',
        'Friday': 'Fri',
        'Saturday': 'Sat',
        'Sunday': 'Sun'
    }
    for full, abbr in weekdays.items():
        if date_str.startswith(full):
            date_str = date_str.replace(full, abbr, 1)
            break

    # Try our standard format first
    try:
        return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
    except ValueError:
        pass

    # If that fails, try email formats
    formats = [
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
        '%a, %d %b %Y %H:%M:%S',     # Without timezone
        '%a, %d %b %y %H:%M:%S %z',  # Short year with timezone
        '%a, %d %b %y %H:%M:%S',     # Short year without timezone
        '%a, %d %b %Y %H:%M',        # Without seconds or timezone
        '%a %b %d %H:%M:%S %z %Y',   # Unix style with timezone before year
        '%d %b %y %H:%M:%S %z',      # Short year with timezone: '26 Nov 09 20:29:04 -0830'
        '%a, %-d %b %Y %H:%M:%S',    # Single digit day: 'Sat, 4 Oct 2008 09:52:43'
        '%d %b %Y',                  # Just date: '13 Apr 2006'
        '%A, %d %b %Y %H:%M:%S %z',  # Full weekday with timezone
        '%A, %-d %b %Y %H:%M:%S %z', # Full weekday, single digit day with timezone
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                # If no timezone, assume UTC
                from datetime import timezone
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

def get_thread_batches(mbox_file: str) -> Generator[list[list[Message]], None, None]:
    """Get batches of threads from mbox file."""
    current_batch = {}  # thread_id -> list[Message]
    
    try:
        mbox = mailbox.mbox(mbox_file)
        for idx, message in enumerate(mbox):
            try:
                msg = Message.from_email_message(message)
                if msg is None:
                    continue
                
                thread_id = msg.x_gm_thrid
                if thread_id not in current_batch:
                    current_batch[thread_id] = []
                current_batch[thread_id].append(msg)
                
                # Yield when we have accumulated enough threads
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
        # Save progress on error
        if vectorstore is not None:
            print("Saving vector store state before exit...")
            vectorstore.save_local(args.vectordb_dir)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Process email threads from mbox file.')
    parser.add_argument('mbox_file', help='Path to mbox file')
    parser.add_argument('--vectordb-dir', default='./mail_vectordb',
                      help='Directory to store vector database')
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
        for thread_batch in get_thread_batches(args.mbox_file):
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
        print("Done!")
    except Exception as e:
        print(f"Error processing mbox file: {e}")
        # Save progress on error
        if vectorstore is not None:
            print("Saving vector store state before exit...")
            vectorstore.save_local(args.vectordb_dir)
        raise

if __name__ == '__main__':
    main()