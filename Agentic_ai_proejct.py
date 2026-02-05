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

# ================= CONFIGURATION =================
S3_BUCKET_NAME = "testing-shree-data1"
S3_FOLDER_PREFIX = "pdf/"
VECTOR_DB_DIR = "faiss_index"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 3
OLLAMA_MODEL = "phi3:mini"

# ================= 1. S3 FILE MANAGEMENT =================
def s3_upload_file(uploaded_file):
    """Uploads a file from Streamlit to S3."""
    s3 = boto3.client("s3")
    try:
        file_key = f"{S3_FOLDER_PREFIX}{uploaded_file.name}"
        s3.upload_fileobj(uploaded_file, S3_BUCKET_NAME, file_key)
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        return True
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return False

def s3_delete_file(file_key):
    """Deletes a file from S3."""
    s3 = boto3.client("s3")
    try:
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        st.success(f"🗑️ Deleted: {file_key}")
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False

def s3_list_files():
    """Returns a list of PDF file keys in the S3 folder."""
    s3 = boto3.client("s3")
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_FOLDER_PREFIX)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".pdf")]
        return []
    except Exception:
        return []

# ================= 2. EMBEDDINGS & RAG BACKEND =================
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
    s3 = boto3.client("s3")
    pdf_keys = s3_list_files()
    
    if not pdf_keys:
        st.warning("No PDFs found in S3 to index.")
        return None

    documents = []
    # Parallel load
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda k: load_single_pdf(s3, S3_BUCKET_NAME, k), pdf_keys)
    
    for docs in results:
        documents.extend(docs)

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

# ================= 3. AGENT TOOLS & LLM =================
def ollama_llm(prompt):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 400, "temperature": 0} 
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

def tool_pdf_search(query):
    retriever = load_retriever()
    if not retriever: return "Index not ready."
    docs = retriever.invoke(query)
    if not docs: return "No details found in PDF."
    return "\n".join([d.page_content[:600] for d in docs])

def tool_web_search(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2)
            if results: return str(results)
    except: pass
    return "No web results."

tools_map = {"pdf_search": tool_pdf_search, "web_search": tool_web_search}

# ================= 4. MODE LOGIC =================

# --- MODE 1: AUTO (Agentic) ---
def run_agent(question, status_box):
    scratchpad = ""
    for step in range(3):
        prompt = f"""
You are a research assistant.
TOOLS: 
- pdf_search (Check internal docs FIRST)
- web_search (External info)

QUESTION: {question}
HISTORY: {scratchpad}

Reply strictly:
"FINAL ANSWER: [text]" OR "ACTION: [tool] QUERY: [text]"
"""
        response = ollama_llm(prompt).strip()
        
        if "FINAL ANSWER:" in response:
            return response.split("FINAL ANSWER:")[-1].strip()
        
        if "ACTION:" in response:
            try:
                parts = response.split("QUERY:")
                tool = parts[0].replace("ACTION:", "").strip()
                query = parts[1].strip()
                
                status_box.write(f"⚙️ **Auto-Pilot:** Using `{tool}` for *'{query}'*...")
                
                if tool in tools_map:
                    res = tools_map[tool](query)
                    scratchpad += f"Tried {tool} with {query}. Result: {res[:400]}...\n"
                else:
                    scratchpad += "Invalid tool.\n"
            except:
                pass
    return "I couldn't find a complete answer in 3 steps."

# --- MODE 2: RAG (PDFs Only) ---
def run_manual_rag(question):
    context = tool_pdf_search(question)
    prompt = f"Answer using ONLY this context:\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    return ollama_llm(prompt)

# --- MODE 3: WEB SEARCH ---
def run_manual_web(question):
    context = tool_web_search(question)
    prompt = f"Answer using these search results:\n\nRESULTS:\n{context}\n\nQUESTION: {question}"
    return ollama_llm(prompt)

# --- MODE 4: LLM ONLY ---
def run_manual_llm(question):
    return ollama_llm(question)

# ================= 5. MAIN UI =================
def main():
    st.set_page_config("Agentic RAG", layout="wide")
    st.title("📂 Agentic RAG + File Manager")

    # --- SIDEBAR ---
    with st.sidebar:
        # A. SEARCH MODE SELECTOR
        st.header("🔍 Search Mode")
        search_mode = st.radio(
            "Select Source:",
            ["Auto", "RAG", "Web Search", "LLM"]
        )
        st.info(f"Current Mode: **{search_mode}**")
        st.divider()

        # B. FILE MANAGER
        st.header("S3 File Manager")
        uploaded_file = st.file_uploader("Upload New PDF", type=["pdf"])
        if uploaded_file:
            if st.button("Confirm Upload"):
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
        
        with st.spinner(f"Running in {search_mode} mode..."):
            
            # --- MODE SWITCHING LOGIC ---
            if search_mode == "Auto":
                status_box.info("🧠 Agent is planning...")
                answer = run_agent(question, status_box)
                
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

if __name__ == "__main__":
    main()