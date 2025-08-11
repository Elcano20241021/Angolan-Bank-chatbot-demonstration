# 🏦 Sol AI Virtual Assistant
**RAG chatbot powered by LangChain, OpenAI, and Hugging Face 🤖**

Banco Sol AI Virtual Assistant is an intelligent customer support chatbot built with Retrieval-Augmented Generation (RAG) and LLM integration, designed to provide accurate, contextual, and cost-efficient answers about Banco Sol's products and services.

| ⚠️ This project is for demonstration purposes only and is not connected to Banco Sol’s internal systems.

## Project Overview
Large Language Models (LLMs) are powerful but rely on static training data, which can make them outdated or inaccurate.
RAG overcomes this limitation by retrieving verified information from official documents before generating a response.

This project combines:

LangChain for RAG orchestration

OpenAI GPT-3.5 Turbo for response generation

Hugging Face sentence transformers for embeddings

Chroma for vector storage

Streamlit for the chatbot interface

Users can upload PDF, TXT, DOCX, or CSV files, and the chatbot will retrieve the most relevant chunks before answering.

## Architecture

Workflow:

Document Loading → Load official Banco Sol brochures/web scraped text.

Splitting → Break documents into token-friendly chunks.

Embeddings → Convert chunks into vectors.

Storage → Save vectors in Chroma DB.

Synonym Normalization → Map user queries like “seguro saúde” → “seguro vida”.

Retriever → Search for top matching chunks.

Context Assembly → Deduplicate & compress context.

LLM Response → Generate grounded answers.

## Web Scraping Integration

Some documents are fetched directly from Banco Sol’s public website and converted to structured data before indexing.

🚀 Features
🔍 Retrieval-Augmented Generation (RAG) → Always retrieves context before answering.

📂 PDF & Web Knowledge Base → Indexed brochures & scraped web content.

🗣 Synonym Matching → Understands popular terms used by customers.

⚙️ Context Compression → Saves tokens while preserving answer quality.

💬 Multi-turn Conversation → Keeps short-term chat history.

📞 Escalation Handling → Suggests contact with human agents if data is unavailable.

🎯 Lead Qualification → Uses follow-up prompts to identify customer intent.

🛠 Tech Stack
Frontend/UI: Streamlit

RAG Orchestration: LangChain

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Vector Store: Chroma

LLM: OpenAI GPT-3.5 Turbo

PDF Processing: LangChain PDF loaders

Synonym Handling: Regex-based normalization

Token Management: tiktoken

📂 Project Structure
bash
Copy
Edit
📦 banco-sol-assistant
 ┣ 📂 data_pdfs/               # Official bank documents
 ┣ 📂 chroma_db_bancosol/      # Persistent vector store
 ┣ mascote_banco_sol.png       # Chatbot mascot
 ┣ app.py                      # Main Streamlit app
 ┣ requirements.txt            # Dependencies
 ┗ README.md                   # Documentation
 
 ## Installation
bash
Copy
Edit
git clone https://github.com/<your-username>/banco-sol-assistant.git
cd banco-sol-assistant
pip install -r requirements.txt
Create a .env file:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
Run:

bash
Copy
Edit
streamlit run app.py

## License
This project is for educational and demonstration purposes only. Banco Sol branding is property of Banco Sol.

