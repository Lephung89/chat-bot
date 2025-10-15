import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import os
import pickle
import json
from datetime import datetime
import gdown
import tempfile
import glob
from pathlib import Path
from dotenv import load_dotenv
import base64

def get_base64_of_image(path):
    """Convert image to base64 string"""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return ""

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Tư Vấn - Đại học Luật TPHCM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tối ưu hơn
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 60, 114, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
    }
    
    .category-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .badge-tuyensinh { background: #e3f2fd; color: #1565c0; }
    .badge-hocphi { background: #e8f5e8; color: #2e7d32; }
    .badge-chuongtrinh { background: #f3e5f5; color: #6a1b9a; }
    
    .info-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 1px solid #ffb74d;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .footer {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Load biến môi trường
load_dotenv()
gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
GDRIVE_VECTORSTORE_ID = st.secrets.get("GDRIVE_VECTORSTORE_ID") or os.getenv("GDRIVE_VECTORSTORE_ID")
GDRIVE_METADATA_ID = st.secrets.get("GDRIVE_METADATA_ID") or os.getenv("GDRIVE_METADATA_ID")

# Cấu hình đường dẫn
DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"

for path in [DOCUMENTS_PATH, VECTORSTORE_PATH]:
    Path(path).mkdir(exist_ok=True)

# Template prompt
COUNSELING_PROMPT_TEMPLATE = """
Bạn là chuyên gia tư vấn tuyển sinh Trường Đại học Luật Thành phố Hồ Chí Minh.

THÔNG TIN LIÊN HỆ CHÍNH THỨC:
- Hotline tuyển sinh: 1900 5555 14 hoặc 0879 5555 14
- Email: tuyensinh@hcmulaw.edu.vn
- Điện thoại: (028) 39400 989
- Địa chỉ: 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM
- Website: www.hcmulaw.edu.vn

Nguyên tắc trả lời:
1. Thân thiện, chuyên nghiệp
2. Cung cấp thông tin chính xác về Đại học Luật TPHCM
3. KHÔNG sử dụng placeholder, luôn dùng thông tin liên hệ cụ thể ở trên
4. Nếu không chắc chắn, khuyến khích liên hệ trực tiếp

Thông tin tham khảo: {context}
Lịch sử hội thoại: {chat_history}
Câu hỏi: {question}

Trả lời (tiếng Việt):
"""

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = load_embeddings()

def download_from_gdrive(file_id, output_path):
    """Download file từ Google Drive"""
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=True)
        return True
    except Exception as e:
        st.warning(f"⚠️ Không thể tải từ GDrive: {e}")
        return False

def get_document_files():
    """Lấy danh sách file trong documents"""
    files = []
    for ext in ['*.pdf', '*.docx', '*.txt']:
        files.extend(glob.glob(os.path.join(DOCUMENTS_PATH, '**', ext), recursive=True))
    return files

def get_file_hash(file_path):
    """Tạo hash cho file"""
    stat = os.stat(file_path)
    return f"{stat.st_mtime}_{stat.st_size}"

def load_cached_vectorstore():
    """Load vector store từ Google Drive"""
    if not GDRIVE_VECTORSTORE_ID or not GDRIVE_METADATA_ID:
        return None, {}
    
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, "vectorstore.pkl")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    
    try:
        if not download_from_gdrive(GDRIVE_VECTORSTORE_ID, vectorstore_path):
            return None, {}
        if not download_from_gdrive(GDRIVE_METADATA_ID, metadata_path):
            return None, {}
        
        with open(vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Cleanup
        os.remove(vectorstore_path)
        os.remove(metadata_path)
        os.rmdir(temp_dir)
        
        return vectorstore, metadata
        
    except Exception as e:
        st.error(f"❌ Lỗi load vectorstore: {e}")
        return None, {}

def process_documents(file_paths):
    """Xử lý documents"""
    documents = []
    processed = []
    failed = []
    
    for file_path in file_paths:
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                failed.append(f"{file_path} (không hỗ trợ)")
                continue
            
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = os.path.basename(file_path)
                doc.metadata['processed_time'] = datetime.now().isoformat()
            
            documents.extend(docs)
            processed.append(file_path)
            
        except Exception as e:
            failed.append(f"{file_path} ({str(e)})")
    
    return documents, processed, failed

def create_vector_store(documents):
    """Tạo vector store"""
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=['\n\n', '\n', '.', '!', '?', ';', ':', ' ']
    )
    texts = text_splitter.split_documents(documents)
    texts = [t for t in texts if len(t.page_content.strip()) > 50]
    
    if not texts:
        return None
    
    return FAISS.from_documents(texts, embeddings)

@st.cache_resource
def initialize_vectorstore():
    """Khởi tạo vectorstore"""
    # Thử load từ cache trước
    vectorstore, metadata = load_cached_vectorstore()
    if vectorstore:
        return vectorstore, metadata.get('stats', {})
    
    # Nếu không có cache, xử lý local files
    current_files = get_document_files()
    if not current_files:
        st.warning("⚠️ Không tìm thấy file nào")
        return None, {}
    
    with st.spinner("🔄 Đang xử lý tài liệu..."):
        documents, processed, failed = process_documents(current_files)
        
        if not documents:
            st.error("❌ Không thể xử lý file")
            return None, {}
        
        vectorstore = create_vector_store(documents)
        
        if vectorstore:
            stats = {
                'total_files': len(current_files),
                'processed_files': len(processed),
                'failed_files': len(failed),
                'total_chunks': vectorstore.index.ntotal,
                'last_updated': datetime.now().isoformat()
            }
            return vectorstore, stats
    
    return None, {}

def classify_question(question):
    """Phân loại câu hỏi"""
    question_lower = question.lower()
    
    categories = {
        "Tuyển sinh": ["tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "xét tuyển"],
        "Học phí": ["học phí", "chi phí", "miễn giảm", "học bổng"],
        "Chương trình đào tạo": ["chương trình", "môn học", "tín chỉ", "ngành"],
    }
    
    for category, keywords in categories.items():
        if any(kw in question_lower for kw in keywords):
            return category
    return "Khác"

def get_category_badge(category):
    """Tạo badge cho category"""
    badge_map = {
        "Tuyển sinh": "badge-tuyensinh",
        "Học phí": "badge-hocphi",
        "Chương trình đào tạo": "badge-chuongtrinh"
    }
    badge_class = badge_map.get(category, "badge-tuyensinh")
    return f'<span class="category-badge {badge_class}">{category}</span>'

def create_conversational_chain(vector_store, llm):
    """Tạo chain"""
    prompt = PromptTemplate(
        template=COUNSELING_PROMPT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )
    
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

@st.cache_resource
def get_gemini_llm():
    """Khởi tạo Gemini LLM với cấu hình tối ưu"""
    try:
        return GoogleGenerativeAI(
            model="gemini-1.5-flash",  # Hoặc thử "models/gemini-1.5-flash"
            google_api_key=gemini_api_key,
            temperature=0.3,
            max_output_tokens=2000
        )
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo Gemini: {e}")
        # Fallback sang model khác nếu cần
        return GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=gemini_api_key,
            temperature=0.3
        )

def answer_from_external_api(prompt, llm, question_category):
    """Trả lời từ API"""
    enhanced_prompt = f"""
Bạn là chuyên gia tư vấn {question_category.lower()} của Đại học Luật TPHCM.

THÔNG TIN LIÊN HỆ (BẮT BUỘC SỬ DỤNG):
- Hotline: 1900 5555 14 hoặc 0879 5555 14
- Email: tuyensinh@hcmulaw.edu.vn
- Điện thoại: (028) 39400 989
- Địa chỉ: 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM
- Website: www.hcmulaw.edu.vn

Câu hỏi: {prompt}

QUY TẮC:
- KHÔNG dùng placeholder như [Số điện thoại], [Email]
- Luôn dùng thông tin cụ thể ở trên
- Kết thúc bằng thông tin liên hệ nếu cần

Trả lời thân thiện, chuyên nghiệp:
"""
    
    try:
        response = llm.invoke(enhanced_prompt)
        
        # Thay thế placeholder còn sót
        replacements = {
            "[Số điện thoại": "1900 5555 14 hoặc 0879 5555 14",
            "[Email": "tuyensinh@hcmulaw.edu.vn",
            "[Website": "www.hcmulaw.edu.vn"
        }
        
        for placeholder, actual in replacements.items():
            if placeholder in response:
                response = response.replace(placeholder + "]", actual)
        
        return response
        
    except Exception as e:
        return f"""
Xin lỗi, hệ thống gặp sự cố. Vui lòng liên hệ:

📞 Hotline: 1900 5555 14 hoặc 0879 5555 14
📧 Email: tuyensinh@hcmulaw.edu.vn
🌐 Website: www.hcmulaw.edu.vn
📍 Địa chỉ: 2 Nguyễn Tất Thành, P.12, Q.4, TP.HCM

Lỗi: {str(e)}
"""

def display_quick_questions():
    """Hiển thị câu hỏi gợi ý"""
    st.markdown("### 💡 Câu hỏi thường gặp")
    
    questions = [
        "📝 Thủ tục đăng ký xét tuyển?",
        "💰 Học phí của trường?",
        "📚 Các ngành học?",
        "🏠 Trường có ký túc xá không?",
        "🎓 Cơ hội việc làm?",
        "📞 Thông tin liên hệ?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                clean_q = q.split(" ", 1)[1]
                st.session_state.messages.append({"role": "user", "content": clean_q})
                st.session_state.process_question = clean_q
                st.rerun()

def main():
    # Kiểm tra API key
    if not gemini_api_key:
        st.error("⚠️ Thiếu GEMINI_API_KEY! Vui lòng cấu hình trong Streamlit Secrets")
        st.stop()
    
    # Khởi tạo session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True

    # Header
    logo_base64 = get_base64_of_image("logo.jpg")
    st.markdown(f"""
    <div class="main-header">
        {f'<img src="data:image/jpg;base64,{logo_base64}" style="width:80px;border-radius:50%;margin-bottom:1rem;">' if logo_base64 else ''}
        <h1>🤖 Chatbot Tư Vấn Tuyển Sinh</h1>
        <h3>Trường Đại học Luật TP. Hồ Chí Minh</h3>
        <p>💬 Hỗ trợ 24/7 | 🎓 Tư vấn chuyên nghiệp</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        
        # Thông tin hệ thống
        with st.expander("📊 Thông tin hệ thống", expanded=False):
            st.info(f"""
            **Trạng thái:**
            - ✅ Gemini API: {'Đã kết nối' if gemini_api_key else '❌ Chưa cấu hình'}
            - 📁 Documents: {len(get_document_files())} files
            """)
        
        # Nút làm mới
        if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📞 Liên hệ
        **Hotline:** 1900 5555 14  
        **Email:** tuyensinh@hcmulaw.edu.vn  
        **Web:** www.hcmulaw.edu.vn
        """)

    # Khởi tạo vectorstore
    with st.spinner("🔄 Đang khởi tạo hệ thống..."):
        vectorstore, stats = initialize_vectorstore()
        llm = get_gemini_llm()
        chain = create_conversational_chain(vectorstore, llm) if vectorstore else None

    # Hiển thị câu hỏi gợi ý nếu là lần đầu
    if not st.session_state.messages and st.session_state.first_visit:
        display_quick_questions()
        
        st.markdown("""
        <div class="info-card">
            <h4>💡 Hướng dẫn sử dụng:</h4>
            <ul>
                <li>🎯 Chọn câu hỏi gợi ý hoặc nhập câu hỏi của bạn</li>
                <li>💬 Đặt câu hỏi cụ thể để được tư vấn chính xác</li>
                <li>📞 Liên hệ trực tiếp nếu cần hỗ trợ khẩn cấp</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "category" in msg:
                st.markdown(get_category_badge(msg["category"]), unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Xử lý input
    prompt = None
    if hasattr(st.session_state, 'process_question'):
        prompt = st.session_state.process_question
        del st.session_state.process_question
    else:
        prompt = st.chat_input("💬 Hãy đặt câu hỏi...")

    if prompt:
        st.session_state.first_visit = False
        
        # Hiển thị câu hỏi
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Phân loại và trả lời
        category = classify_question(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(get_category_badge(category), unsafe_allow_html=True)
            
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    if chain:
                        response = chain({"question": prompt})
                        answer = response["answer"]
                    else:
                        answer = answer_from_external_api(prompt, llm, category)
                    
                    st.markdown(answer)
                    
                except Exception as e:
                    answer = f"""
❌ **Lỗi hệ thống**

Vui lòng liên hệ:
📞 Hotline: 1900 5555 14 hoặc 0879 5555 14
📧 Email: tuyensinh@hcmulaw.edu.vn

Lỗi: {str(e)}
"""
                    st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "category": category
            })

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>🏛️ Trường Đại học Luật TP. Hồ Chí Minh</h4>
        <p>📍 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM</p>
        <p>📞 Hotline: 1900 5555 14 | Email: tuyensinh@hcmulaw.edu.vn</p>
        <p>🌐 www.hcmulaw.edu.vn | 📘 facebook.com/hcmulaw</p>
        <p style="margin-top:1rem;opacity:0.8;font-size:0.85em;">
            🤖 Chatbot v2.0 | Phát triển bởi Lvphung - CNTT
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
