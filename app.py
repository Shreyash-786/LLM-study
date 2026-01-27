import os
import streamlit as st
import boto3
import tempfile
import ollama

from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from duckduckgo_search import DDGS

# ================= CONFIG =================
S3_BUCKET_NAME = "testing-shree-data1"
S3_FOLDER_PREFIX = "pdf/"
VECTOR_DB_DIR = "faiss_index"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 2

# ================= EMBEDDINGS (CACHED) =================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",   # ⚡ very fast
        model_kwargs={"device": "cpu"}
    )

# ================= LOAD PDFs FROM S3 =================
def load_single_pdf(s3, key):
    pdf_obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    pdf_bytes = pdf_obj["Body"].read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    os.remove(tmp_path)
    return docs

def load_pdfs_from_s3():
    s3 = boto3.client("s3")
    documents = []

    response = s3.list_objects_v2(
        Bucket=S3_BUCKET_NAME,
        Prefix=S3_FOLDER_PREFIX
    )

    if "Contents" not in response:
        return documents

    pdf_keys = [
        obj["Key"] for obj in response["Contents"]
        if obj["Key"].endswith(".pdf")
    ]

    # ⚡ Parallel loading
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda k: load_single_pdf(s3, k), pdf_keys)

    for docs in results:
        documents.extend(docs)

    return documents

# ================= CREATE VECTOR DB =================
def create_vector_db():
    documents = load_pdfs_from_s3()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)

    return vectorstore

# ================= LOAD RETRIEVER =================
@st.cache_resource
def load_retriever():
    embeddings = get_embeddings()

    if not os.path.exists(VECTOR_DB_DIR):
        create_vector_db()

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ================= WEB SEARCH =================
def web_search(query):
    text = ""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=2)
        for r in results:
            text += f"{r['title']}\n{r['body']}\n\n"
    return text.strip()

# ================= OLLAMA =================
def ollama_llm(prompt):
    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 200}  # ⚡ speed limit
    )
    return response["message"]["content"]

# ================= ANSWER PIPELINE =================
def answer_question(question, retriever):
    # 1️⃣ FAST RAG
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    if context.strip():
        rag_prompt = f"""
Use ONLY the context. If answer is missing, say NOT FOUND.

Context:
{context}

Q: {question}
A:
"""
        rag_answer = ollama_llm(rag_prompt)

        if "NOT FOUND" not in rag_answer and len(rag_answer) > 25:
            return "📄 PDF", rag_answer

    # 2️⃣ WEB FALLBACK
    web_data = web_search(question)
    if web_data:
        verify_prompt = f"""
Does this answer fully solve the question?
Reply YES or NO.

Question:
{question}

Answer:
{web_data}
"""
        decision = ollama_llm(verify_prompt)
        if "YES" in decision:
            return "🌐 WEB", web_data

    # 3️⃣ PURE LLM
    final_prompt = f"Answer clearly:\n{question}"
    return "🤖 LLM", ollama_llm(final_prompt)

# ================= STREAMLIT UI =================
st.set_page_config("Fast S3 RAG", layout="wide")
st.title("LLM Study")

# st.write(
#     "PDFs → **AWS S3** → **FAISS** → **RAG** → Web → LLM\n\n"
#     "**Optimized for speed** 🚀"
# )

if st.button("☁️ Index PDFs from S3"):
    with st.spinner("Indexing PDFs (one-time)..."):
        create_vector_db()
        st.success("Index created successfully")

retriever = load_retriever()

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking fast..."):
        source, answer = answer_question(question, retriever)
        st.markdown(f"### Source: {source}")
        st.write(answer)
