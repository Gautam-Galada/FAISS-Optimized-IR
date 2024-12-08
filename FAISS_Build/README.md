# Information Retrieval System with FAISS and LLM Integration

## **Overview**
This project implements an advanced Information Retrieval (IR) system leveraging **FAISS** (Facebook AI Similarity Search), **HuggingFace Embeddings**, and a Flask-based user interface. The system scrapes, preprocesses, and indexes Wikipedia data, enabling fast and scalable query-based document retrieval. It uses an LLM (e.g., Ollama or llama3) to enhance conversational AI capabilities for contextual responses.

## **Key Features**
- **Efficient Retrieval**: Utilizes FAISS for scalable vector search with Maximum Marginal Relevance (MMR).
- **Preprocessing**: Cleans and normalizes text data to ensure high-quality embeddings.
- **Embeddings**: Generates semantic embeddings with HuggingFace's `sentence-transformers/paraphrase-MiniLM-L6-v2`.
- **Flask Integration**: Provides an intuitive interface for interacting with the system.
- **Topic Metadata**: Tracks and analyzes query distribution by topic for insightful analytics.

## **File Structure**
```plaintext
project/
│
├── data/
│   ├── cleaned_wiki_articles.json       # Input JSON file with cleaned data
│   └── split_documents.pkl              # Pickle file for split document chunks
│
├── embeddings/
│   ├── faiss_index                      # FAISS index directory
│
├── scripts/
│   ├── preprocess.py                    # Script for chunking and saving documents
│   ├── build_faiss.py                   # Script for building FAISS index
│   └── query_faiss.py                   # Script for querying the FAISS database
│
├── requirements.txt                     # Dependencies for the project
└── README.md                            # Project documentation
