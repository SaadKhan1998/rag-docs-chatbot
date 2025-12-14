# 📄 Retrieval-Augmented Generation (RAG) System for Document Q&A

This project implements an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** that enables semantic question answering over private PDF documents using Large Language Models (LLMs) and a vector database.

The system ingests PDFs, converts their content into vector embeddings, retrieves the most relevant document chunks for a user query, and generates **grounded answers with source attribution** via a FastAPI backend.

---

## 🚀 Features

- PDF document ingestion and preprocessing  
- Text chunking and vector embedding generation  
- Semantic search using a vector database (ChromaDB)  
- LLM-based answer generation constrained to retrieved context  
- REST API built with FastAPI  
- Source-aware responses (shows which document was used)  
- Designed to avoid hallucinations  

---

## 🧠 System Architecture

PDFs → Text Extraction → Chunking → Embeddings → Vector DB (Chroma)
↓
User Question → Embedding → Retrieval → Context Injection → LLM → Answer


---

## 🛠️ Tech Stack

- **Python**
- **LangChain (latest modular APIs)**
- **OpenAI (LLM + embeddings)**
- **ChromaDB** (vector database)
- **FastAPI** (API backend)
- **Uvicorn** (ASGI server)

---

## 📁 Project Structure

RAG/
├── app/
│ ├── api.py # FastAPI endpoints
│ ├── rag_pipeline.py # Core RAG logic
│ ├── ingest.py # PDF ingestion & embedding
│ └── config.py # Environment configuration
├── data/ # PDF documents
├── storage/ # Vector database (local)
├── .env.example # Environment variable template
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

### follow the steps
1️⃣ Create virtual environment
```bash
python -m venv rag_env
source rag_env/bin/activate   # Windows: rag_env\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Configure environment variables

Create a .env file:

OPENAI_API_KEY=your_openai_key_here
LLM_MODEL_NAME=gpt-4.1-mini
EMBEDDING_MODEL_NAME=text-embedding-3-small

📥 Ingest Documents

Place one or more PDF files into the data/ folder

Run ingestion:

python -m app.ingest


This will:

extract text from PDFs

split text into chunks

generate embeddings

store them in ChromaDB

▶️ Run the API
uvicorn app.api:app --reload


API will be available at:

http://127.0.0.1:8000


Interactive API docs:

http://127.0.0.1:8000/docs

❓ Example Query

Request

{
  "question": "What is the main topic of the document?"
}


Response

{
  "answer": "The document discusses ...",
  "sources": ["example_paper.pdf"]
}


If the answer is not found in the retrieved context, the system responds with:

"I don't know."

🔍 Key Design Decisions

No hallucination by default — answers are strictly constrained to retrieved document context

Persistent knowledge base — vector database acts as long-term memory

Modular architecture — retriever, LLM, and prompt logic are cleanly separated

Production-style setup — backend-first design, frontend-agnostic

📌 Limitations

No OCR support (image-based PDFs not handled)

Stateless Q&A (no chat memory yet)

Local vector database (not distributed)

🔮 Possible Extensions

Conversational memory (chat history)

OCR for scanned PDFs

Streamlit or web-based chatbot UI

Source highlighting with page numbers

Evaluation metrics for retrieval quality

📄 License

This project is for educational and portfolio purposes.


---

## ✅ What this README achieves
- Explains **what** you built  
- Explains **how it works**  
- Shows **engineering maturity**  
- Makes the project runnable by recruiters  
- Clearly signals **GenAI / AI Engineer competence**

---

### Next optional step
If you want, I can:
- generate a `.gitignore`
- review your repo before first push
- help you tag this as **v1.0**

Just tell me 👍