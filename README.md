**🧠  System with S3, FAISS & Ollama**

This project is an Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents to AWS S3 and interact with them through an autonomous AI agent.
The agent dynamically decides whether to retrieve information from internal PDFs, perform a web search, or generate an answer using its own reasoning, following a ReAct (Reason–Act–Observe) architecture.

The application is built with Streamlit for an interactive UI, FAISS for fast vector search, HuggingFace embeddings for semantic understanding, and Ollama (Phi-3) as the local LLM.

**🔑 Key Highlights**

🤖  AI (ReAct Architecture)
The LLM autonomously chooses which tool to use (PDF Search or Web Search) and iteratively reasons until a final answer is produced.

📂 Cloud-Based Knowledge Management
PDF documents are stored in AWS S3, allowing scalable and persistent document storage.

🔍 High-Performance RAG Pipeline

PDF parsing using LangChain loaders

Chunking with overlap for context preservation

Vector indexing using FAISS

🌐 Web Search Fallback
If relevant information is not found in PDFs, the agent can query the web using DuckDuckGo.

🧠 Local LLM Inference
Uses Ollama (Phi-3 Mini) for fast, private, and offline-friendly inference.

🧠 Short-Term Agent Memory
Maintains a reasoning scratchpad to track previous actions and observations.

**🖥️ Interactive Streamlit UI**

Upload / delete PDFs from S3


User Query
   ↓
Agent (LLM – ReAct Loop)
   ↓
Decides Action
   ├── PDF Search (FAISS + S3)
   ├── Web Search (DuckDuckGo)
   └── LLM Reasoning
   ↓
Final Answer


Re-index documents on demand

Switch between Auto  RAG-only, Web-only, and LLM-only modes
