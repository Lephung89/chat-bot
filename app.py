
import streamlit as st
import os
from pathlib import Path
import json
import time
import re
import glob
import pickle
import tempfile
import logging
from datetime import datetime
from dotenv import load_dotenv
import base64
import gdown
import schedule
import threading
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Thiết lập logging
logging.basicConfig(filename='chatbot.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Hàm chuyển ảnh sang base64
def get_base64_of_image():
    """Convert image to base64 string"""
    try:
        path = "logo.jpg"
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Không tìm thấy file logo: {path}")
        logging.error(f"Không tìm thấy file logo: {path}")
        return ""
    except Exception as e:
        st.error(f"Lỗi đọc file logo: {str(e)}")
        logging.error(f"Lỗi đọc file logo: {str(e)}")
        return ""
# Cấu hình trang Streamlit
st.set_page_config(
    page_title="ChatBot Tư Vấn - Đại học Luật TP.HCM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .main-header {
        background: linear-gradient(90deg, #1e72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .header-logo {
        height: 100px;
        margin-bottom: 1rem;
    }
    .header-title {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .header-title img {
        height: 50px;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header h3 {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.8;
    }
    .main-header p {
        font-size: 0.9rem;
        opacity: 0.0.7;
    }
    .chat-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .user-message {
        background: #e6f3ff;
        border-left: 4px solid #007bff;
    }
    .assistant-message {
        background: #f0f9ff;
        border-left: 4px solid #28a745;
    }
    .sidebar-info {
        background: #ffffff;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .metric-card {
        background: flex#fff;
        padding: 1rem;
        border-radius: 4px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
    }
    .info-card {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .category-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        margin: 0.1rem;
    }
    .badge-tuyensinh { background: #007bff; color: white; }
    .badge-hocphi { background: #28a745; color: white; }
    .badge-ch { background: #17a2b8; color: white; }
    .badge-sinhhot { background: #ffc107; color: black; }
    .badge-hotro { background: #dc3545; color: white; }
    .badge-totnghiep { background: #6c757d; color: white; }
    .quick-actions { display: grid; gap: 1rem; }
    .action-button {
        background: #007bff;
        color: white;
        padding: 0.8rem;
        border-radius: 4px;
        text-align: center;
        text-decoration: none;
    }
    .action-button:hover {
        background: #0056b3;
    }
    .footer {
        background: #333;
        color: white;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
    .footer-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .footer-section h4 {
        color: #007bff;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 4px;
    }
    .chat-input {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GDRIVE_VECTORSTORE_ID = os.getenv("GDRIVE_VECTORSTORE_ID")
GDRIVE_METADATA_ID = os.getenv("GDRIVE_METADATA_ID")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

# Paths
DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"
CACHE_PATH = "cache"
CHAT_HISTORY_PATH = os.path.join(CACHE_PATH, "chat_history.json")

# Create directories
for path in [DOCUMENTS_PATH, VECTORSTORE_PATH, CACHE_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Prompt template
PROMPT_TEMPLATE = """
Bạn là chuyên viên tư vấn tuyển sinh của Đại học Luật TP.HCM. Trả lời câu hỏi ngắn gọn, chính xác, ưu tiên thông tin từ tài liệu. Nếu không có thông tin, trả lời dựa trên kiến thức chung, nêu rõ và khuyến khích liên hệ trực tiếp.

**Thông tin liên hệ**:
- Hotline: 1900 5555 14 hoặc 0879 987 654
- Email: tuyensinh@hcmulaw.edu.vn
- Website: www.hcmulaw.edu.vn
- Địa chỉ: 123 Nguyễn Văn Cừ, Quận 5, TP.HCM

**Nguyên tắc**:
1. Trả lời ngắn gọn, đúng trọng tâm.
2. Ưu tiên thông tin từ tài liệu: {context}.
3. Nếu không có thông tin, nêu rõ và hướng dẫn liên hệ.
4. Thân thiện, chuyên nghiệp.

**Lịch sử hội thoại**: {chat_history}
**Câu hỏi**: {question}

**Trả lời**:
"""

# Google Drive API service
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_gdrive_service():
    try:
        service_account_info = os.getenv("SERVICE_ACCOUNT_JSON")
        if service_account_info:
            logging.info("Sử dụng SERVICE_ACCOUNT_JSON từ .env")
            with open("temp_service_account.json", "w") as f:
                json.dump(json.loads(service_account_info), f)
            creds = Credentials.from_service_account_file("temp_service_account.json", scopes=SCOPES)
            os.remove("temp_service_account.json")
        else:
            logging.info("Sử dụng file service_account.json")
            creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
        service = build("drive", "v3", credentials=creds)
        logging.info("Khởi tạo Google Drive API thành công")
        return service
    except Exception as e:
        logging.error(f"Lỗi khởi tạo Google Drive API: {str(e)}")
        return None

# Upload file to Google Drive
def upload_to_gdrive(file_path, file_id=None, folder_id=None):
    service = get_gdrive_service()
    if not service:
        return False
    try:
        file_name = os.path.basename(file_path)
        file_metadata = {"name": file_name}
        if folder_id:
            file_metadata["parents"] = [folder_id]
        media = MediaFileUpload(file_path)
        if file_id:
            file = service.files().update(fileId=file_id, media_body=media, fields="id").execute()
        else:
            file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        logging.info(f"Uploaded {file_name} to Google Drive, ID: {file.get('id')}")
        return file.get("id")
    except Exception as e:
        logging.error(f"Error uploading {file_name}: {str(e)}")
        return False

# Download file from Google Drive
def download_from_gdrive(file_id, output_path):
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=True)
        logging.info(f"Downloaded file from Google Drive: {file_id}")
        return True
    except Exception as e:
        logging.error(f"Error downloading file {file_id}: {str(e)}")
        return False

# Initialize embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize sentence transformer
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Get document files
def get_document_files():
    files = []
    for ext in ["*.pdf", "*.docx", "*.txt"]:
        files.extend(glob.glob(os.path.join(DOCUMENTS_PATH, "**", ext), recursive=True))
    logging.info(f"Found {len(files)} files in documents")
    return files

# File hash for change detection
def get_file_hash(file_path):
    stat = os.stat(file_path)
    return f"{stat.st_mtime}_{stat.st_size}"

# Load vector store
def load_cached_vectorstore():
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, "vectorstore.pkl")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    try:
        if download_from_gdrive(GDRIVE_VECTORSTORE_ID, vectorstore_path) and download_from_gdrive(GDRIVE_METADATA_ID, metadata_path):
            with open(vectorstore_path, "rb") as f:
                vectorstore = pickle.load(f)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logging.info("Loaded vector store from Google Drive")
            return vectorstore, metadata
    except Exception as e:
        logging.error(f"Error loading vector store: {str(e)}")
    finally:
        if os.path.exists(vectorstore_path):
            os.remove(vectorstore_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        os.rmdir(temp_dir)
    return None, None

# Save vector store
def save_vectorstore_cache(vector_store, metadata):
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, "vectorstore.pkl")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    try:
        with open(vectorstore_path, "wb") as f:
            pickle.dump(vector_store, f)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        vectorstore_id = upload_to_gdrive(vectorstore_path, GDRIVE_VECTORSTORE_ID, GDRIVE_FOLDER_ID)
        metadata_id = upload_to_gdrive(metadata_path, GDRIVE_METADATA_ID, GDRIVE_FOLDER_ID)
        if vectorstore_id and metadata_id:
            logging.info("Saved vector store to Google Drive")
            return True
        return False
    except Exception as e:
        logging.error(f"Error saving vector store: {str(e)}")
        return False
    finally:
        try:
            os.remove(vectorstore_path)
            os.remove(metadata_path)
            os.rmdir(temp_dir)
        except:
            pass

# Check if vector store needs rebuild
def need_rebuild_vectorstore():
    files = get_document_files()
    if not files:
        return True, [], []
    current_metadata = {file_path: get_file_hash(file_path) for file_path in files}
    cached_vectorstore, cached_metadata = load_cached_vectorstore()
    if not cached_metadata:
        return True, current_metadata, files
    if set(current_metadata.keys()) != set(cached_metadata.get("files", {}).keys()):
        return True, current_metadata, files
    return False, current_metadata, files

# Process documents
def process_documents(file_paths):
    documents = []
    processed_files = []
    failed_files = []
    for file_path in file_paths:
        try:
            extension = os.path.splitext(file_path)[1].lower()
            stat = os.stat(file_path)
            if extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                failed_files.append(f"{file_path} (unsupported format)")
                continue
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(file_path)
                doc.metadata["file_type"] = extension[1:]
                doc.metadata["file_size"] = stat.st_size
                doc.metadata["last_modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            documents.extend(docs)
            processed_files.append(file_path)
        except Exception as e:
            failed_files.append(f"{file_path} ({str(e)}")
            logging.error(f"Error processing {file_path}: {str(e)}")
    return documents, processed_files, failed_files

# Create vector store
def create_vector_store(documents):
    if not documents:
        return None
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return None
    vector_store = FAISS.from_documents(chunks, load_embeddings())
    return chunks, vector_store

# Initialize vector store
@st.cache_resource
def create_vector_store(documents):
    try:
        # Khởi tạo splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        # Chia tài liệu
        split_docs = splitter.split_documents(documents)
        
        # Tạo embeddings
        embeddings = load_embeddings()
        
        # Tạo vector store
        vector_store = FAISS.from_documents(split_docs, embeddings)
        logging.info("Tạo vector store thành công")
        return vector_store
    except Exception as e:
        logging.error(f"Lỗi tạo vector store: {str(e)}")
        return None

# Classify question
def classify_question(question):
    transformer = load_sentence_transformer()
    categories = {
        "Tuyển sinh": ["Thủ tục đăng ký xét tuyển?", "Điểm chuẩn là bao nhiêu?", "Hồ sơ cần gì?"],
        "Học phí": ["Học phí bao nhiêu?", "Có miễn giảm học phí không?", "Học bổng như thế nào?"],
        "Học hỏi": ["Ngành học của trường?", "Chương trình học gồm gì?", "Thời khóa biểu ra sao?"],
        "Hỗ trợ": ["Có ký túc xá không?", "Thư viện mở khi nào?", "Hỗ trợ tư vấn tâm lý?"]
    }
    question_emb = transformer.encode(question)
    max_sim = 0
    category = "Không rõ"
    for cat, examples in categories.items():
        ex_embs = transformer.encode(examples)
        sims = util.cos_sim(question_emb, ex_embs)[0]
        if max(sims) > 0.5 and max(sims) > max_sim:
            max_sim = max(sims)
            category = cat
        return category
    logging.info(f"Question not classified: {question}")
    return "Không rõ"

# Category badge
def get_category_badge(category):
    badges = {
        "Tuyển sinh": "badge-tuyensinh",
        "Học phí": "badge-blue",
        "Học hỏi": "badge-info",
        "Hỗ trợ": "badge-red",
        "Không rõ": "badge-gray"
    }
    return f'<span class="badge {badges.get(category, "badge-gray")}">{category}</span>'

# Conversational chain
def create_conversation_chain(vector_store, llm):
    prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
	)
    memory = ConversationChain.from_llm(
        llm=llm,
        vectorstore=vector_store.as_retriever(),
        memory=ConversationBufferWindowMemory(k=4),
        return_prompt=prompt
    )
    return chain

# Initialize LLM
@st.cache_resource
def get_gemini_llm():
    return GoogleGenerativeAI(model="gemini-pro", api_key="GEMINI_API_KEY")

# Answer from LLM
def answer_from_api(prompt, llm, category):
    enhanced = f"""
    Bạn tư vấn {category} của Đại học Luật TP.HCM.
    Câu hỏi: {prompt}
    Hướng dẫn:
    - Trả lời ngắn gọn, đúng trọng tâm.
    - Nếu không rõ, nêu rõ và khuyến khích liên hệ.
    Thông tin liên hệ:
    - Hotline: 1900 5555 14
    - Email: tuyensinh@hcmulh.edu.vn
    - Website: www.hcmulaw.edu.vn
    """
    try:
        answer = llm.invoke(enhanced)
        return answer
    except Exception as e:
        return f"Lỗi hệ thống. Liên hệ: tuyensinh@hcmulaw.edu.vn. ({str(e)})"

# Clean response
def clean_response(response):
    replacements = {
        "[HOTLINE]": "1900 5555 14",
        "[EMAIL]": "tuyensinh@hcmulh.edu.vn",
        "[WEBSITE]": "www.hcmulaw.edu.vn"
    }
    for k, v in replacements.items():
        answer = response.replace(k, v)
    return response

# Save chat history
def save_chat_history(question, response, category, feedback=None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "response": response,
        "category": category,
        "feedback": feedback
    }
    try:
        with open(CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
            json.dump([entry], f, ensure_ascii=False)
        logging.info(f"Saved chat history: {question}")
    except Exception as e:
        logging.error(f"Error saving chat history: {str(e)}")

# Display metrics
def display_metrics(stats):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("total_files", 0)}</div><div class="metric-label">Tổng tài liệu</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("processed_files", 0)}</div><div class="metric-label">Đã xử lý</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats.get("failed_files", 0)}</div><div class="metric-label">Lỗi</div>', unsafe_allow_html=True)

# Quick actions
def display_quick_actions():
    actions = [
        "Thủ tục đăng ký xét tuyển?",
        "Học phí bao nhiêu?",
        "Ngành học của trường?",
        "Có ký túc xá không?",
    ]

    
    for i, action in enumerate(actions):
        if st.button(action, key=f"quick{i}"):
            st.session_state.messages.append({"role": "user", "content": action})
            st.rerun()

# Display sources
def display_sources(sources):
    if not sources:
        st.markdown("Không tìm thấy nguồn tài liệu")
        return
    with st.expander("Nguồn tham khảo"):
        for i, src in enumerate(sources[:3]):
            st.markdown(f"{i+1}. {src.metadata['source_file']} ({src.metadata['file_type']})")

# Admin login
def check_admin_login():
    return st.session_state.get("admin_login", False)

# Admin login form
def admin_login_form():
    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.admin_login = True
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials")

# Main function
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "admin_login" not in st.session_state:
        st.session_state.admin_login = False

    st.markdown(f"""
    <div class="main-header">
        <img src="data:image/jpeg;base64,{get_base64_of_image()}" alt="logo-img" class="logo">
        <h1>Chatbot Tư Vấn Tuyển Sinh</h1>
        <p>Đại học Luật TP.HCM</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Thông tin hệ thống")
        if get_gdrive_service():
            st.markdown("<div class='success-card'>Kết nối Google Drive OK</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warning-card'>Lỗi kết nối Google Drive</div>", unsafe_allow_html=True)
        if check_admin_login():
            if st.button("Rebuild Vector Store"):
                initialize_vector_store.clear()
                st.session_state.messages = []
                st.rerun()
            if st.button("Logout"):
                st.session_state.admin_login = False
                st.rerun()
        else:
            admin_login_form()

    vector_store, _, stats = initialize_vector_store()
    llm = get_gemini_llm()
    chain = create_conversation_chain(vector_store, llm) if vector_store else None

    if not st.session_state.messages:
        display_quick_actions()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(get_category_badge(msg.get("category", "Không rõ")))
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍", key=f"like{i}"):
                        save_chat_history(msg.get("question", ""), msg["content"], msg.get("category", ""), "like")
                        st.success("Thanks for feedback!")
                    elif st.button("👎", key=f"dislike{i}"):
                        save_chat_history(msg.get("question", ""), msg["content"], msg.get("category", ""), "dislike")
                        st.warning("Thanks for feedback!")

                prompt = st.chat_input("Hỏi gì đó...")
                if prompt:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    category = classify_question(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(get_category_badge(category))
                        try:
                            if chain:
                                response = chain.invoke({"question": prompt})
                                answer = response["answer"]
                                display_sources(response.get("source_documents", []))
                            else:
                                answer = answer_from_gemini(prompt, llm, category)
                                st.markdown("Không có tài liệu tham khảo")
                                answer = clean_response(answer)
                                st.markdown(answer)
                                save_chat_history(prompt, answer, category)
                                st.session_state.messages.append({"role": "assistant", "content": answer, "category": category, "question": prompt})
                        except Exception as e:
                                st.error(f"Lỗi hệ thống: {str(e)}")
                                save_chat_history(prompt, str(e), answer, category)
                                st.session_state.messages.append({"role": "assistant", "content": str(e), "category": category, "question": prompt})

    st.markdown("""
    <div class="footer">
        <p>© 2023 Đại học Luật TP.HCM</p>
        <p>Hotline: 1900 5555 14 | Email: tuyensinh@hsp.edu.vn</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()