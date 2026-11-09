# RAG Document Assistant

## Project Overview
A Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from uploaded PDF documents.

## Technologies
- Python
- Google Gemini API
- ChromaDB
- Sentence Transformers
- Streamlit
- PyPDF

## Features
- PDF text extraction
- Text chunking with overlap
- Semantic embeddings
- Vector storage using ChromaDB
- Relevant document retrieval
- Context-grounded Gemini responses
- Source filename and page references
- RAG vs No-RAG comparison

## Documents Used
1. Python Interview Notes
2. Python Complete Notes Outline
3. Reusable ML Pipeline with scikit-learn
4. Streamlit Playwright Final Guide

Total: 4 PDF documents, 95 pages and 99 chunks processed.

## Project Structure
    RAG-System/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── documents/
    └── src/
        ├── chunking.py
        ├── ingest.py
        ├── retrieval.py
        └── generation.py

## Setup Instructions

1. Install dependencies:
   pip install -r requirements.txt

2. Create a .env file in the project root and add:
   GEMINI_API_KEY=your_api_key_here

3. Add your PDF documents to the documents folder.

4. Ingest documents:
   python src/ingest.py

5. Start the application:
   streamlit run app.py

## RAG Workflow
1. Extract text from PDF documents.
2. Split text into overlapping chunks.
3. Generate embeddings using all-MiniLM-L6-v2.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks for a question.
6. Pass retrieved context to Gemini.
7. Generate a grounded answer with source references.

## RAG vs No-RAG
The application compares an answer generated using retrieved PDF context with a normal Gemini answer without the PDF context.

## Test Results
Successfully processed 4 PDF documents into 99 chunks. Tested semantic retrieval and Gemini-based answer generation with source references.

## Author
P SHADIK KHAN
