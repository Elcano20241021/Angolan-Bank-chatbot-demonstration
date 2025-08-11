# 🏦 Sol AI Virtual Assistant
**RAG chatbot powered by LangChain, OpenAI, and Hugging Face 🤖**
<img width="1240" height="1840" alt="RAG_architecture" src="https://github.com/user-attachments/assets/d93090ce-c1ae-4098-9aa8-d6e6812c367d" />
<img width="540" height="238" alt="web_scraping" src="https://github.com/user-attachments/assets/1e6aba0b-5580-4b40-84fc-0e5e5cb68434" />


Banco Sol AI Virtual Assistant is an intelligent customer support chatbot built with Retrieval-Augmented Generation (RAG) and LLM integration, designed to provide accurate, contextual, and cost-efficient answers about Banco Sol's products and services.

| ⚠️ This project is for demonstration purposes only and is not connected to Banco Sol’s internal systems.

## Project Overview
Large Language Models (LLMs) are powerful but rely on static training data, which can make them outdated or inaccurate.
RAG overcomes this limitation by retrieving verified information from official documents before generating a response.

This project combines:

- LangChain for RAG orchestration;
- OpenAI GPT-3.5 Turbo for response generation;
- Hugging Face sentence transformers for embeddings;
- Chroma for vector storage;
- Streamlit for the chatbot interface

Users can upload PDF, TXT, DOCX, or CSV files, and the chatbot will retrieve the most relevant chunks before answering.

## Architecture

**Workflow**
- Document Loading → Load official Banco Sol brochures or web-scraped text.
- Splitting → Break documents into token-friendly chunks.
- Embeddings → Convert chunks into dense vector representations.
- Storage → Store vectors in Chroma DB for retrieval.
- Synonym Normalization → Map queries like "seguro saúde" → "seguro vida".
- Retriever → Search for top-matching chunks in the vector store.
- Context Assembly → Deduplicate & compress retrieved context.
- LLM Response → Generate grounded, context-aware answers.

## Web Scraping Integration

Some documents are fetched directly from Banco Sol’s public website and converted to structured data before indexing.

## Features
- Local PDF Knowledge Base – Automated ingestion of Banco Sol PDFs (data_pdfs/) with metadata for category, subcategory, and product.
- Multilingual Embeddings – Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for semantic search.
- Synonym Expansion – Normalizes user queries with a configurable synonym dictionary for better retrieval.
- Token-Optimized Context – Deduplicates and trims context to stay within token limits.
- Fallback Handling – Escalates to human support when needed.
- Streamlit UI – User-friendly web interface with mascot branding.

## Main Files
- app.py – Main Streamlit application with RAG pipeline, synonym handling, and UI.
- Webscrap_Banco_Sol.ipynb – Web scraping script to collect and process Banco Sol website content.
- requirements.txt – Python dependencies for local setup.

📂 Project Structure
📦 sol-assistant
 ┣ 📂 data_pdfs/               # Official bank documents & brochures
 ┣ 📂 chroma_db_bancosol/      # Persistent Chroma vector store
 ┣ mascote_banco_sol.png       # Chatbot mascot/logo
 ┣ app.py                      # Main Streamlit application
 ┣ requirements.txt            # Project dependencies
 ┗ README.md                   # Documentation
 
## License
This project is for educational and demonstration purposes only. Banco Sol branding is property of Banco Sol.

