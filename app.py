import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import time
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect environment - use Groq if API key exists, else Ollama
USE_GROQ = os.getenv("GROQ_API_KEY") is not None

if USE_GROQ:
    from langchain_groq import ChatGroq
else:
    from langchain_ollama import OllamaLLM

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="DocuChat AI ✨",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CUSTOM CSS =============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 35, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #f0f0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8), 0 0 40px rgba(240, 147, 251, 0.5); }
    }
    
    .hero-subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
    }
    
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    
    .source-box {
        background: rgba(255, 255, 255, 0.08);
        border-left: 4px solid #f093fb;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0 5px;
    }
    
    .badge-groq {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-ollama {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    button[title="Deploy this app"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    
    [data-testid="stToolbar"] {
        background: transparent !important;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 3rem !important;
    }
    
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[kind="header"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border-radius: 12px !important;
        padding: 0.7rem !important;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.6) !important;
        z-index: 999999 !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: white !important;
        border: none !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        width: 3rem !important;
        height: 3rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { 
            box-shadow: 0 4px 20px rgba(245, 87, 108, 0.6);
        }
        50% { 
            box-shadow: 0 4px 30px rgba(245, 87, 108, 0.9);
        }
    }
    
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg,
    button[kind="header"] svg {
        color: white !important;
        fill: white !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover,
    button[kind="header"]:hover {
        transform: scale(1.1) rotate(5deg) !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%) !important;
    }
    
    .stSpinner > div {
        border-color: #f093fb transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ============= SESSION STATE =============
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "show_landing" not in st.session_state:
    st.session_state.show_landing = True

# ============= HELPER FUNCTIONS =============
@st.cache_resource
def get_embeddings():
    """Get embeddings model - works both locally and in cloud"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_llm():
    """Get LLM - Groq for cloud, Ollama for local"""
    if USE_GROQ:
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        return OllamaLLM(model="llama3.2")

# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("# 🤖 DocuChat AI")
    st.markdown("##### *Your AI Document Companion*")
    
    if USE_GROQ:
        st.markdown('<span class="badge badge-groq">⚡ Powered by Groq</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-ollama">🦙 Powered by Ollama</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📁 Upload Your Docs")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if st.button("✨ Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("🪄 Working magic..."):
                all_docs = []
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    os.unlink(tmp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(all_docs)
                
                embeddings = get_embeddings()
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    collection_name=f"docs_{uuid.uuid4().hex[:8]}"
                )
                
                st.session_state.doc_count = len(uploaded_files)
                st.session_state.chunk_count = len(splits)
                st.session_state.show_landing = False
                
                st.success(f"✅ Ready to chat!")
                time.sleep(1)
                st.rerun()
        else:
            st.warning("⚠️ Upload PDFs first!")
    
    if st.session_state.doc_count > 0:
        st.markdown("---")
        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Docs", st.session_state.doc_count)
        with col2:
            st.metric("🧩 Chunks", st.session_state.chunk_count)
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🛠️ Powered By")
    if USE_GROQ:
        st.markdown("⚡ **Groq** - Lightning fast LLM")
    else:
        st.markdown("🦙 **Llama 3.2** - Local AI")
    st.markdown("🔗 **LangChain** - Framework")
    st.markdown("💾 **ChromaDB** - Vector DB")
    st.markdown("🤗 **HuggingFace** - Embeddings")
    st.markdown("⚡ **Streamlit** - UI")
    
    st.markdown("---")
    st.markdown("##### Made with 💜")

# ============= MAIN AREA =============
if st.session_state.show_landing and st.session_state.vectorstore is None:
    st.markdown('<h1 class="hero-title">✨ DocuChat AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Have intelligent conversations with your documents using AI 🚀</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">Secure</div>
            <div class="feature-desc">Your data is processed safely</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Lightning Fast</div>
            <div class="feature-desc">Instant answers from your PDFs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Completely Free</div>
            <div class="feature-desc">No subscriptions required</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 How It Works")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:2.5rem;">1️⃣</div>
            <div class="feature-title">Upload PDFs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:2.5rem;">2️⃣</div>
            <div class="feature-title">Process Docs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:2.5rem;">3️⃣</div>
            <div class="feature-title">Ask Questions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size:2.5rem;">4️⃣</div>
            <div class="feature-title">Get Answers</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:rgba(255,255,255,0.8); font-size:1.1rem;'>👈 Upload your PDFs in the sidebar to get started!</p>", unsafe_allow_html=True)

else:
    st.markdown('<h1 class="hero-title">💬 Chat with Your Docs</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Ask anything about your uploaded documents</p>', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📄 Source {i}</strong> - Page {source.get('page', 'N/A')}<br>
                            <em>{source['content'][:200]}...</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("💭 Ask me anything about your documents..."):
        if st.session_state.vectorstore is None:
            st.error("⚠️ Please upload and process documents first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🤔 Thinking..."):
                    llm = get_llm()
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        return_source_documents=True
                    )
                    
                    result = qa_chain.invoke({"query": prompt})
                    response = result["result"]
                    
                    message_placeholder = st.empty()
                    full_response = ""
                    for word in response.split():
                        full_response += word + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)
                    message_placeholder.markdown(full_response)
                    
                    sources = []
                    if "source_documents" in result:
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                page = doc.metadata.get('page', 'N/A')
                                content = doc.page_content
                                sources.append({"page": page, "content": content})
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>📄 Source {i}</strong> - Page {page}<br>
                                    <em>{content[:200]}...</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })