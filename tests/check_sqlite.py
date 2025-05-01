#!/usr/bin/env python3
import os
import sys
import sqlite3
from dotenv import load_dotenv

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path, override=True)

# Get the SQLite DB path from environment
sqlite_db_path = os.getenv("SQLITE_DB")
if not sqlite_db_path:
    print("SQLITE_DB environment variable not set")
    sys.exit(1)

print(f"Checking SQLite database at: {sqlite_db_path}")

# Remove quotes if present
if sqlite_db_path.startswith('"') and sqlite_db_path.endswith('"'):
    sqlite_db_path = sqlite_db_path[1:-1]

# Check if the file exists
if not os.path.exists(sqlite_db_path):
    print(f"SQLite database file does not exist: {sqlite_db_path}")
    sys.exit(1)

print(f"SQLite database file exists and is {os.path.getsize(sqlite_db_path) / (1024*1024):.2f} MB")

# Try to connect to the SQLite database
try:
    conn = sqlite3.connect(sqlite_db_path)
    print(f"Successfully connected to SQLite database")
    
    # Get table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in the database: {[table[0] for table in tables]}")
    
    # Check if emails table exists
    if ('emails',) in tables:
        # Count documents in the emails table
        cursor.execute("SELECT COUNT(*) FROM emails")
        count = cursor.fetchone()[0]
        print(f"Found {count} documents in the emails table")
        
        # Get a sample document
        cursor.execute("SELECT id, thread_id, message_index FROM emails LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print(f"Sample document ID: {sample[0]}")
            print(f"Sample thread ID: {sample[1]}")
            print(f"Sample message index: {sample[2]}")
            
            # Check if a specific document exists
            test_id = f"full_{sample[1]}_{sample[2]}"
            cursor.execute("SELECT COUNT(*) FROM emails WHERE id = ?", (test_id,))
            exists = cursor.fetchone()[0] > 0
            print(f"Document with ID '{test_id}' exists: {exists}")
            
            # Try without the 'full_' prefix
            test_id = f"{sample[1]}_{sample[2]}"
            cursor.execute("SELECT COUNT(*) FROM emails WHERE id = ?", (test_id,))
            exists = cursor.fetchone()[0] > 0
            print(f"Document with ID '{test_id}' exists: {exists}")
            
            # Check document ID format
            cursor.execute("SELECT id FROM emails LIMIT 10")
            ids = [row[0] for row in cursor.fetchall()]
            print(f"Document ID format examples: {ids}")
    else:
        print("No 'emails' table found in the database")
    
    conn.close()
except Exception as e:
    print(f"Error connecting to SQLite database: {str(e)}")

print("Check complete")
