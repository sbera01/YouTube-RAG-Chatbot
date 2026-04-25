# YouTube RAG Chatbot

This project is an end-to-end YouTube question-answering assistant that uses RAG to generate grounded answers from video transcripts. You paste a YouTube link, it builds a local FAISS index from transcript chunks, and then answers user questions through a Streamlit chat interface using HuggingFace models.

## Problem
Users often need quick answers from long YouTube videos, but manually searching through transcripts is slow and inefficient. Most chatbots also answer without grounding, which can lead to hallucinations.

## Solution
This project builds a Retrieval-Augmented Generation (RAG) chatbot for YouTube videos. A user pastes a YouTube URL, the system extracts the transcript, indexes it in FAISS, and answers questions using only retrieved context. This makes responses faster, more accurate, and traceable to source content.

## Features
- Accepts a YouTube URL and extracts video ID automatically in backend.
- Enforces English transcript workflow using TRANSCRIPT_LANGUAGE=en.
- Fetches transcript via LangChain YoutubeLoader, then stores it locally for reuse.
- Splits transcript into chunks for semantic retrieval.
- Generates HuggingFace embeddings and stores vectors in local FAISS index.
- Uses similarity search to retrieve top relevant context.
- Uses LLaMA 3.1 via HuggingFace Endpoint for final grounded response generation.
- Streamlit UI for indexing and chat interaction.

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS (local vector store)
- HuggingFace Embeddings
- HuggingFace Inference Endpoint (LLaMA 3.1)
- LangChain YoutubeLoader

## Arc
Architecture flow:

```text
YouTube URL
    ↓
Transcript Extraction
    ↓
Chunking
    ↓
Embeddings (HF model)
    ↓
FAISS Vector Store
    ↓
User Query
    ↓
Similarity Search
    ↓
LLM (LLaMA)
    ↓
Final Answer
```


