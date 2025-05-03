"""Database utilities for storing and retrieving email documents."""
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
import json

class EmailStore:
    def __init__(self, db_path: str):
        """Initialize the email store with the given database path."""
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    id TEXT PRIMARY KEY,  -- Format: thread_id_msg_index
                    thread_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,  -- JSON encoded metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Index for faster lookups by thread_id
            conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON emails(thread_id)")
    
    def add_document(self, doc_id: str, thread_id: str, message_index: int, content: str, metadata: Dict[str, Any]):
        """Add a document to the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO emails (id, thread_id, message_index, content, metadata) VALUES (?, ?, ?, ?, ?)",
                (doc_id, thread_id, message_index, content, json.dumps(metadata))
            )
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM emails WHERE id = ?",
                (doc_id,)
            ).fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'thread_id': row['thread_id'],
                    'message_index': row['message_index'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']),
                    'created_at': row['created_at']
                }
            return None
    
    def get_thread_messages(self, thread_id: str) -> list[Dict[str, Any]]:
        """Get all messages in a thread, ordered by message_index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM emails WHERE thread_id = ? ORDER BY message_index",
                (thread_id,)
            ).fetchall()
            
            return [{
                'id': row['id'],
                'thread_id': row['thread_id'],
                'message_index': row['message_index'],
                'content': row['content'],
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at']
            } for row in rows]
