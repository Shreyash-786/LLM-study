import os
import streamlit as st
import boto3
import tempfile
import ollama
import re

from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from duckduckgo_search import DDGS

# --- UNIVERSAL IMPORT FIX ---
# This block handles both old and new versions of LangChain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ================= CONFIGURATION =================
S3_BUCKET_NAME = "testing-shree-data1"
S3_FOLDER_PREFIX = "pdf/"
VECTOR_DB_DIR = "faiss_index"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
OLLAMA_MODEL = "phi3:mini"

# ================= 1. OPTIMIZED S3 CONNECTION =================
@st.cache_resource
def get_s3_client():
    return boto3.client("s3")

def s3_upload_file(uploaded_file):
    s3 = get_s3_client()
    try:
        file_key = f"{S3_FOLDER_PREFIX}{uploaded_file.name}"
        s3.upload_fileobj(uploaded_file, S3_BUCKET_NAME, file_key)
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        return True
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

def s3_delete_file(file_key):
    s3 = get_s3_client()
    try:
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        st.success(f"🗑️ Deleted: {file_key}")
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False

def s3_list_files():
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_FOLDER_PREFIX)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".pdf")]
        return []
    except Exception:
        return []

# ================= 2. RAG BACKEND =================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="intfloat/e5-small-v2",
        model_kwargs={"device": "cpu"}
    )

def load_single_pdf(s3_client, bucket, key):
    try:
        pdf_obj = s3_client.get_object(Bucket=bucket, Key=key)
        pdf_bytes = pdf_obj["Body"].read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.remove(tmp_path)
        return docs
    except Exception:
        return []

def create_vector_db():
    s3 = get_s3_client()
    pdf_keys = s3_list_files()
    
    if not pdf_keys:
        st.warning("No PDFs found in S3 to index.")
        return None

    documents = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda k: load_single_pdf(s3, S3_BUCKET_NAME, k), pdf_keys)
    
    for docs in results:
        documents.extend(docs)

    if not documents:
        return None

    # Use the safe import class
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_DIR)
    return vectorstore

@st.cache_resource
def load_retriever():
    embeddings = get_embeddings()
    if not os.path.exists(VECTOR_DB_DIR):
        create_vector_db()
    try:
        vectorstore = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    except:
        return None

# ================= 3. TOOLS & LLM =================
def ollama_llm(prompt):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 400, "temperature": 0} 
        )
        return response["message"]["content"]
    except Exception as e:
        return f"LLM Error: {e}"

def tool_pdf_search(query):
    retriever = load_retriever()
    if not retriever: return None
    try:
        docs = retriever.invoke(query)
        if not docs: return None
        return "\n".join([d.page_content[:600] for d in docs])
    except Exception as e:
        print(f"PDF Search Error: {e}")
        return None

def tool_web_search(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2)
            if results: return str(results)
    except Exception as e:
        print(f"Web Search Error: {e}")
        pass
    return None

# ================= 4. SMART AUTO LOGIC =================
def run_auto_fallback(question, status_box):
    """
    Logic: RAG -> Validate -> Web -> Validate -> LLM
    """
    try:
        # 1. Try PDF
        status_box.info("📄 Checking Internal PDFs...")
        pdf_context = tool_pdf_search(question)
        
        if pdf_context:
            validation = ollama_llm(f"Context: {pdf_context[:500]}\nQ: {question}\nDoes this answer the question? YES or NO.")
            if "YES" in validation.upper():
                status_box.success("✅ Found in PDFs.")
                return ollama_llm(f"Context: {pdf_context}\nQ: {question}\nAnswer:")
            else:
                status_box.warning("⚠️ PDF result irrelevant. Trying Web...")
        else:
            status_box.warning("⚠️ PDFs empty. Trying Web...")

        # 2. Try Web
        status_box.info("🌐 Searching Web...")
        web_context = tool_web_search(question)
        
        if web_context:
            validation = ollama_llm(f"Context: {web_context[:500]}\nQ: {question}\nDoes this answer the question? YES or NO.")
            if "YES" in validation.upper():
                status_box.success("✅ Found on Web.")
                return ollama_llm(f"Context: {web_context}\nQ: {question}\nAnswer:")
            else:
                status_box.error("❌ Web result irrelevant. Using Brain...")
        else:
            status_box.error("❌ Web search failed. Using Brain...")

        # 3. Fallback to LLM
        status_box.info("🧠 Generating from internal knowledge...")
        return ollama_llm(question)

    except Exception as e:
        return f"Auto Mode Crash: {e}"

# --- Manual Helpers ---
def run_manual_rag(q):
    ctx = tool_pdf_search(q)
    return ollama_llm(f"Context: {ctx}\nQ: {q}") if ctx else "No PDF info found."

def run_manual_web(q):
    ctx = tool_web_search(q)
    return ollama_llm(f"Context: {ctx}\nQ: {q}") if ctx else "No web info found."

def run_manual_llm(q):
    return ollama_llm(q)

# ================= 5. MAIN UI =================
def main():
    st.set_page_config("Agentic RAG", layout="wide")
    st.title("📂 Agentic RAG + File Manager")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🔍 Search Mode")
        search_mode = st.radio(
            "Select Source:",
            ["Auto", "RAG", "Web Search", "LLM"]
        )
        st.divider()

        st.header("S3 File Manager")
        uploaded_file = st.file_uploader("Upload New PDF", type=["pdf"])
        if uploaded_file and st.button("Confirm Upload"):
            if s3_upload_file(uploaded_file):
                st.cache_resource.clear()
                create_vector_db()
                st.rerun()

        st.subheader("Existing Files")
        files = s3_list_files()
        if files:
            for f in files:
                col1, col2 = st.columns([0.8, 0.2])
                col1.text(f.replace(S3_FOLDER_PREFIX, ""))
                if col2.button("🗑️", key=f):
                    if s3_delete_file(f):
                        st.cache_resource.clear()
                        create_vector_db()
                        st.rerun()
        else:
            st.info("No PDFs in S3.")

        if st.button("🔄 Force Re-Index"):
            create_vector_db()
            st.success("Index Updated!")

    # --- MAIN CHAT ---
    question = st.text_input("Ask a question:")
    
    if question:
        st.divider()
        status_box = st.container()
        answer = ""
        
        # Crash Handler: This catches errors and shows them in the UI
        try:
            with st.spinner(f"Running in {search_mode} mode..."):
                if search_mode == "Auto":
                    answer = run_auto_fallback(question, status_box)
                elif search_mode == "RAG":
                    status_box.info("📄 Searching S3 PDFs...")
                    answer = run_manual_rag(question)
                elif search_mode == "Web Search":
                    status_box.info("🌐 Searching the Web...")
                    answer = run_manual_web(question)
                elif search_mode == "LLM":
                    status_box.info("🧠 Using Internal Knowledge...")
                    answer = run_manual_llm(question)
            
            status_box.empty()
            st.subheader("💡 Answer")
            st.success(answer)
            
        except Exception as e:
            st.error(f"❌ Application Error: {e}")
            st.write("Tip: Check if Ollama is running (`ollama serve`) and internet is connected.")

if __name__ == "__main__":
    main()
