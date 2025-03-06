import mailbox
from email import message_from_bytes
from email.header import decode_header
from bs4 import BeautifulSoup
import re
import os, sys, shutil
import argparse
from os.path import exists
from pathlib import Path
from typing import Union, List, Optional, Dict

class Message:
    def __init__(self, to, sender, subject, date, content, x_gmail_labels=[],
                 x_gm_thrid=None, inReplyTo=None, references=[]):
        self.to = to
        self.sender = sender
        self.subject = subject
        self.date = date
        self.content = content
        self.x_gmail_labels = x_gmail_labels
        self.x_gm_thrid = x_gm_thrid
        self.inReplyTo = inReplyTo
        self.references = references

def extract_content(message: mailbox.mboxMessage):
    """Extracts content from an email message."""
    if message.is_multipart():
        content = ''
        for part in message.walk():
            if part.get_content_type() == 'text/plain':
                content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif part.get_content_type() == 'text/html':
                html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                content += soup.get_text()
        return content
    else:
        if message.get_content_type() == 'text/plain':
            return message.get_payload(decode=True).decode('utf-8', errors='ignore')
        elif message.get_content_type() == 'text/html':
            html_content = message.get_payload(decode=True).decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()

def clean_content(content: str):
    """Removes any remaining HTML tags and extra whitespace."""
    content = re.sub(r'<.*?>', '', content)  # Remove any remaining HTML tags
    content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
    return content.strip()

def load_mbox(file_path):
    """Loads an mbox file and extracts messages."""
    mbox = mailbox.mbox(file_path)
    messages = []

    def safe_strip(s):
        return s.strip("\r") if s is not None else None
    
    for message in mbox:
        sender = safe_strip(message['From'])
        to = safe_strip(message['To'])
        subject = safe_strip(message['Subject'])
        date = message['Date']
        xgmThrid = safe_strip(message['X-GM-THRID'])
        inReplyTo = safe_strip(message.get('In-Reply-To'))
        refs = message.get("References")
        if refs != None:
            references = refs.split()
        else:
            references = None
        labels = message.get("X-Gmail-Labels")
        x_gmail_labels = labels.split(',')
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


def organize_messages(messages):
    return 0

file_path = "AllMailIncludingSpamAndTrash.mbox"
mbox = mailbox.mbox(file_path)
print(type(mbox))
first_message = next(iter(mbox.values()))
print(type(first_message))
#exit(0)

# Example usage
file_path = "AllMailIncludingSpamAndTrash.mbox"
print_messages(file_path=file_path)


