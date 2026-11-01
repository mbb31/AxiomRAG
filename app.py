import streamlit as st
import pdfplumber
import base64
import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

st.set_page_config(page_title="AxiomRAG", page_icon="🏛️", layout="wide")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- MAXIMUM UI/UX & CHATGPT LAYOUT CSS ---
st.markdown("""
<style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 95% !important;}
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a, .header-anchor { display: none !important; visibility: hidden !important; }

    /* Remove default chat message styling */
    .stChatMessage {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
        padding: 0 !important;
        margin-bottom: 15px !important;
    }
    .stChatMessage > div:first-child { display: none !important; }

    /* USER MESSAGE: Right aligned dark gray pill */
    .stChatMessage:has(.user-marker) {
        flex-direction: row-reverse !important;
    }
    .stChatMessage:has(.user-marker) > div:nth-child(2) {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border-radius: 24px 24px 6px 24px !important;
        padding: 12px 20px !important;
        max-width: 80% !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
        display: inline-block !important;
    }

    /* AI MESSAGE: Left aligned transparent text */
    .stChatMessage:has(.ai-marker) > div:nth-child(2) {
        background-color: transparent !important;
        color: #f8fafc !important;
        padding: 10px 0 !important;
        max-width: 95% !important;
    }

    /* ChatGPT-style Chat Input Container */
    [data-testid="stChatInput"] > div {
        border-radius: 50px !important; 
        padding: 5px 15px !important;
        border: 1px solid #475569 !important;
        background-color: #1e293b !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
    }

    /* Document Viewer & Dropzone */
    iframe {
        border-radius: 24px;
        border: 2px solid #334155;
        background-color: #0f172a;
    }
    [data-testid="stFileUploadDropzone"] {
        border-radius: 24px !important;
        border: 2px dashed #475569 !important;
        background-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "uploaded_file_names" not in st.session_state: st.session_state.uploaded_file_names = []
if "file_bytes_dict" not in st.session_state: st.session_state.file_bytes_dict = {}

col_doc, col_chat = st.columns([1, 1.2], gap="large")

with col_doc:
    st.markdown("<h2 style='color: #f8fafc; font-weight: 800; font-family: sans-serif;'>📂 Knowledge Base</h2>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload Financial PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        current_names = sorted([f.name for f in uploaded_files])
        
        if st.session_state.uploaded_file_names != current_names:
            with st.spinner("Processing & Vectorizing Dataset..."):
                st.session_state.uploaded_file_names = current_names
                st.session_state.file_bytes_dict = {f.name: f.getvalue() for f in uploaded_files}
                
                all_chunks = []
                for file in uploaded_files:
                    pdf = pdfplumber.open(file)
                    page_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    shredder = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    all_chunks.extend(shredder.split_text(page_text))
                
                translator = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
                unique_id = uuid.uuid4().hex
                st.session_state.vector_db = Chroma.from_texts(
                    texts=all_chunks, 
                    embedding=translator,
                    collection_name=f"axiom_{unique_id}"
                )
                st.session_state.chat_history = [] 
            st.success(f"✨ Successfully linked {len(uploaded_files)} document(s)")
        
        if len(uploaded_files) > 1:
            selected_preview = st.selectbox("Select document to preview:", st.session_state.uploaded_file_names)
        else:
            selected_preview = st.session_state.uploaded_file_names[0]
            
        if selected_preview in st.session_state.file_bytes_dict:
            base64_pdf = base64.b64encode(st.session_state.file_bytes_dict[selected_preview]).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="650px"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

with col_chat:
    st.markdown("<h2 style='color: #f8fafc; font-weight: 800; font-family: sans-serif;'>💬 Analysis Interface</h2>", unsafe_allow_html=True)
        
    # CRITICAL FIX: Fixed height scrollable container pushes the input box to the bottom!
    chat_container = st.container(height=580, border=False)
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("<p style='text-align: center; color: #94a3b8; margin-top: 20vh;'>Upload a document and ask a question to begin.</p>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            marker = "<span class='user-marker'></span>" if msg["role"] == "user" else "<span class='ai-marker'></span>"
            with st.chat_message(msg["role"]):
                st.markdown(marker + msg["content"], unsafe_allow_html=True)
            
    # Input floats permanently below the scrollable container
    if user_question := st.chat_input("Message AxiomRAG..."):
        if not st.session_state.vector_db:
            st.error("Please upload at least one document on the left first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(f"<span class='user-marker'></span>{user_question}", unsafe_allow_html=True)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing vectors..."):
                        llm = OllamaLLM(model="llama3.1", base_url=OLLAMA_URL)
                        clues = st.session_state.vector_db.similarity_search(user_question, k=8)
                        context = "\n\n".join([c.page_content for c in clues])
                        
                        prompt = f"""You are a precise data extraction AI. Extract the direct answer to the user's question from the provided CONTEXT. 
                        
                        CRITICAL RULES:
                        1. Only use the provided CONTEXT. Do not use outside knowledge.
                        2. Do NOT include conversational filler like "Based on the document" or "I can find the answer". Provide the exact answer immediately.
                        3. If the answer is completely missing from the CONTEXT, reply exactly with: "I cannot find the relevant information in the uploaded document."

                        CONTEXT:
                        {context}

                        QUESTION: {user_question}
                        ANSWER:"""
                        
                        answer = llm.invoke(prompt)
                        st.markdown(f"<span class='ai-marker'></span>{answer}", unsafe_allow_html=True)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})