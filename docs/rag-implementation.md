# RAG Implementation Notes

This document captures the implementation details, learnings, and insights about the Retrieval Augmented Generation (RAG) system used in the mail_chat project.

## Overview

The mail_chat project implements RAG using:
- NVIDIA AI Endpoints for embeddings
- FAISS vector store from `langchain_community.vectorstores`
- Document handling via `langchain_core.documents`
- Both Simple RAG and Conversational Chain approaches

## Key Components

### Picking the right data store

### Chroma
Chroma is an **open-source vector database** specifically designed for storing and retrieving vector embeddings, along with **associated metadata**. It is widely used in AI applications such as semantic search, recommendation systems, and optimizing large language models (LLMs).

The RAG flow can be structured so that it can use Chroma as the vector store. The LLM can be used to generate the query that will be used to access Chroma database. The output can then be processed to retrieve the answer.

So a query like "When did I last sent email to Gopal Srinivasan?" can be translated to a database query like 
```
"query='Gopal Srinivasan' filter=Comparison(comparator=<Comparator.EQ: 'eq'>, attribute='to', value='Gopal Srinivasan') limit=None
```
using the LLM.

Langchain has `load_query_constructor_runnable` which can be used to transform the query into a generic database query. This is intermediate database query is then translated to the actual query to be sent to the vector store once again using the LLM. In LangChain this is done by `SelfQueryRetriever`.

Example:

```python
from langchain_community.query_constructors.chroma import ChromaTranslator

retriever = SelfQueryRetriever(
    query_constructor=query_constructor,
    vectorstore=vectorstore,
    structured_query_translator=ChromaTranslator(),
)
```



### FAISS
FAISS is not a traditional database, but rather a specialized library designed for efficient similarity search and clustering of high-dimensional data. It is focused primarily on 

1. Vector indexing: FAISS creates optimized data structures to organize and store high-dimensional vectors efficiently.
2. Similarity search: It provides fast and accurate search capabilities for finding the most similar vectors to a query.
3. Scalability: It is designed to handle large datasets and can scale to support millions of vectors.
4 GPU Acceleration: Many of its algorithms are implemented to run efficiently on GPUs, making it suitable for large-scale applications.

However, It lacks persistence and metadata filtering capabilities.
This means, queries like these cannot be answered
- "show me all emails from 2024"
- "when did I last receive an email from abc@xxy.org"

Unlike traditional databases that excel at exact matches and structured queries, FAISS is specifically tailored for approximate nearest neighbor search in vector spaces. It's particularly useful for applications involving embeddings, such as recommendation systems, image similarity, and natural language processing tasks where finding similar items based on their vector representations is crucial
Essentially, FAISS is not ideal for the mail chat application.

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
