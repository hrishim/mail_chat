# RAG Implementation Notes

This document captures the implementation details, learnings, and insights about the Retrieval Augmented Generation (RAG) system used in the mail_chat project.

## Overview

The mail_chat project implements RAG using:
- NVIDIA AI Endpoints for embeddings
- FAISS vector store from `langchain_community.vectorstores`
- Document handling via `langchain_core.documents`
- Both Simple RAG and Conversational Chain approaches

## Key Components

### Reranking System
- Implements cosine similarity scoring for document relevance
- I intend to use NVIDIAEmbeddings from `langchain_nvidia_ai_endpoints`

### Vector Store
Notes for when you explore different chunking and embedding strategies

- Uses FAISS from `langchain_community.vectorstores`
- Stores and retrieves email chunks efficiently
- Enables semantic search capabilities

## Implementation Details

[More details to be added as the implementation evolves]
