# 🚀 Fast S3 RAG with Streamlit, FAISS & Ollama

A **high-performance Retrieval-Augmented Generation (RAG)** application that:
- Indexes **PDFs stored in AWS S3**
- Creates embeddings using **HuggingFace**
- Stores vectors in **FAISS**
- Answers questions using **Ollama (Phi-3 Mini)**
- Falls back to **Web Search (DuckDuckGo)** if PDFs don’t contain the answer

Built with **speed and simplicity** in mind ⚡

---

## 🧠 Architecture Overview

PDFs (AWS S3)
↓
PyPDFLoader
↓
Text Chunking
↓
HuggingFace Embeddings
↓
FAISS Vector DB
↓
Retriever (Top-K)
↓
Ollama LLM (Phi-3)
↓
Web Search Fallback


---

## ✨ Features

- ☁️ Load **PDFs directly from AWS S3**
- ⚡ Parallel PDF loading (ThreadPoolExecutor)
- 📚 Fast semantic search using **FAISS**
- 🧠 **RAG-first** answering (no hallucination)
- 🌐 Web fallback if answer not found in PDFs
- 🤖 Local LLM using **Ollama**
- 🖥️ Clean **Streamlit UI**
- 💾 Cached embeddings & retriever for performance

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|------------|
| UI | Streamlit |
| Storage | AWS S3 |
| PDF Parsing | LangChain PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace (`intfloat/e5-small-v2`) |
| Vector DB | FAISS |
| LLM | Ollama (`phi3:mini`) |
| Web Search | DuckDuckGo Search |
| Language | Python |

---

