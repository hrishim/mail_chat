import mailbox
from email import message_from_bytes
from email.header import decode_header
from bs4 import BeautifulSoup
import re
import os, sys, shutil
import argparse
from os.path import exists
from pathlib import Path
from typing import Union, Optional
from mailbox import mboxMessage
from datetime import datetime

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
            f"x_gm_thrid: {self.x_gm_thrid}\n"
            f"inReplyTo: {self.inReplyTo}\n"
            f"References: {self.references}"
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

def load_mbox(file_path):
    """Loads an mbox file and extracts messages."""
    mbox = mailbox.mbox(file_path)
    messages = []

    def safe_strip(s: str) -> str:
        return s.strip("\r") if s is not None else ""
    
    for message in mbox:
        sender = safe_strip(message['From'])
        to = safe_strip(message['To'])
        subject = safe_strip(message['Subject'])
        date = message['Date']
        xgmThrid = safe_strip(message['X-GM-THRID'])
        igt = message.get('In-Reply-To')
        inReplyTo = safe_strip(igt) if igt else ""
        refs = message.get("References")
        if refs != None:
            references = refs.split()
        else:
            references = []
        labels = message.get("X-Gmail-Labels")
        x_gmail_labels: list[str] = labels.split(',') if labels else [] 

        # Extract content
        content = extract_content(message)
        content = clean_content(content)
        
        # Create Message object
        msg_obj = Message(to=to, sender=sender, subject=subject, date=date, 
                          content=content, x_gmail_labels=x_gmail_labels, x_gm_thrid=xgmThrid,
                          inReplyTo=inReplyTo, references=references)
        messages.append(msg_obj)
    
    return messages

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
    org_msgs: dict[str, list[Message]] = {}

    for msg in messages:
        if msg.x_gm_thrid is None:
            org_msgs['orphan'].append(msg)
        else:
            org_msgs[msg.x_gm_thrid]
    # Sorting each list in the dictionary by the 'date' field
    for key, messages in org_msgs.items():
        org_msgs[key] = sorted(messages, key=lambda msg: datetime.strptime(msg.date, "%Y-%m-%d %H:%M:%S%z"))
    return org_msgs

file_path = "AllMailIncludingSpamAndTrash.mbox"
mbox = mailbox.mbox(file_path)
print(type(mbox))
first_message = next(iter(mbox.values()))
print(type(first_message))
#exit(0)

# Example usage
file_path = "AllMailIncludingSpamAndTrash.mbox"
print_messages(file_path=file_path)


