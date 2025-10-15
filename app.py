import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import requests
import json
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import os
import pickle
import json
from datetime import datetime
import requests
import tempfile
import glob
from pathlib import Path
from dotenv import load_dotenv
import base64
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
debug = False
verbose = False


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

# CSS tối ưu
st.markdown("""
<style>
.header-container {
    background: linear-gradient(90deg, #003366, #004c99);
    color: white;
    padding: 24px 0;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}
.header-container h1 {
    font-size: 2rem;
    margin-bottom: 0.4rem;
    font-weight: 700;
}
.header-container h3 {
    font-size: 1.2rem;
    font-weight: 400;
    margin-top: 0;
    opacity: 0.9;
}
.header-container p {
    font-size: 1rem;
    margin-top: 0.2rem;
    opacity: 0.85;
}
</style>

<div class="header-container">
    <h1>🤖 Chatbot Tư Vấn Tuyển Sinh</h1>
    <h3>Trường Đại học Luật TP. Hồ Chí Minh</h3>
    <p>💬 Hỗ trợ 24/7 &nbsp; | &nbsp; 🎓 Tư vấn chuyên nghiệp</p>
</div>
""", unsafe_allow_html=True)


# Load biến môi trường
load_dotenv()

# ĐỌC API KEY TỪ SECRETS TRƯỚC, SAU ĐÓ MỚI ĐẾN .env
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except:
    gemini_api_key = os.getenv("GEMINI_API_KEY")

try:
    GDRIVE_VECTORSTORE_ID = st.secrets["GDRIVE_VECTORSTORE_ID"]
except:
    GDRIVE_VECTORSTORE_ID = os.getenv("GDRIVE_VECTORSTORE_ID")

try:
    GDRIVE_METADATA_ID = st.secrets["GDRIVE_METADATA_ID"]
except:
    GDRIVE_METADATA_ID = os.getenv("GDRIVE_METADATA_ID")

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

def download_from_gdrive_direct(file_id, output_path):
    """
    Download file từ Google Drive bằng requests (không dùng gdown)
    File phải được share công khai: Anyone with the link
    """
    try:
        # URL download trực tiếp từ GDrive
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Xử lý virus scan warning của GDrive (file lớn)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'export': 'download', 'id': file_id, 'confirm': value}
                response = session.get(url, params=params, stream=True)
                break
        
        # Lưu file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        
        return True
        
    except Exception as e:
        #st.warning(f"⚠️ Không thể tải từ GDrive: {e}")
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
        #st.info("ℹ️ Chưa cấu hình Google Drive, sử dụng xử lý local")
        return None, {}
    
    temp_dir = tempfile.mkdtemp()
    vectorstore_path = os.path.join(temp_dir, "vectorstore.pkl")
    metadata_path = os.path.join(temp_dir, "metadata.json")
    
    try:
        # Download bằng requests thay vì gdown
        if not download_from_gdrive_direct(GDRIVE_VECTORSTORE_ID, vectorstore_path):
            return None, {}
        if not download_from_gdrive_direct(GDRIVE_METADATA_ID, metadata_path):
            return None, {}
        
        with open(vectorstore_path, 'rb') as f:
            vectorstore = pickle.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Cleanup
        os.remove(vectorstore_path)
        os.remove(metadata_path)
        os.rmdir(temp_dir)
        
        #st.success("✅ Đã load vectorstore từ Google Drive")
        return vectorstore, metadata
        
    except Exception as e:
        st.warning(f"⚠️ Không thể load từ GDrive: {e}")
        # Cleanup nếu có lỗi
        try:
            if os.path.exists(vectorstore_path):
                os.remove(vectorstore_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass
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
    # Thử load từ GDrive trước
    vectorstore, metadata = load_cached_vectorstore()
    if vectorstore:
        return vectorstore, metadata.get('stats', {})
    
    # Nếu không có GDrive, xử lý local files
    st.info("ℹ️ Đang xử lý tài liệu local...")
    current_files = get_document_files()
    
    if not current_files:
        st.warning("⚠️ Không tìm thấy file nào trong thư mục documents")
        return None, {}
    
    with st.spinner("🔄 Đang xử lý tài liệu..."):
        documents, processed, failed = process_documents(current_files)
        
        if not documents:
            st.error("❌ Không thể xử lý file nào")
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
            st.success(f"✅ Đã xử lý {len(processed)} files thành công!")
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
    """
    Tạo chain với Google Generative AI
    Không dùng LangChain chain nữa - xử lý trực tiếp
    """
    # Trả về tuple: (vectorstore, llm) để xử lý manual
    return (vector_store, llm)

@st.cache_resource
def get_gemini_llm():
    """
    Tạo Gemini client dùng REST API trực tiếp
    GIẢI PHÁP CUỐI CÙNG - LUÔN HOẠT ĐỘNG
    """
    if not gemini_api_key:
        st.error("❌ Thiếu GEMINI_API_KEY!")
        st.stop()
    
    # Test API key
    test_url = f"https://generativelanguage.googleapis.com/v1/models?key={gemini_api_key}"
    
    try:
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            # Tìm model tốt nhất có sẵn
            preferred_models = [
                'models/gemini-1.5-flash-latest',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro-latest',
                'models/gemini-1.5-pro',
                'models/gemini-pro'
            ]
            
            selected_model = None
            for model in preferred_models:
                if model in available_models:
                    selected_model = model
                    break
            
            if not selected_model and available_models:
                # Nếu không có model ưa thích, lấy model đầu tiên
                selected_model = available_models[0]
            
            if selected_model:
                #st.success(f"✅ Đã kết nối Gemini: {selected_model}")
                return {
                    'api_key': gemini_api_key,
                    'model': selected_model,
                    'available_models': available_models
                }
            else:
                st.error("❌ Không tìm thấy model nào!")
                st.stop()
                
        elif response.status_code == 400:
            st.error("❌ API key không hợp lệ!")
            st.info("Lấy API key mới tại: https://aistudio.google.com/app/apikey")
            st.stop()
        else:
            st.error(f"❌ Lỗi API: {response.status_code}")
            st.stop()
            
    except requests.exceptions.Timeout:
        st.error("❌ Timeout khi kết nối Gemini API")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
        st.info("""
        **Hướng dẫn:**
        1. Lấy API key: https://aistudio.google.com/app/apikey
        2. Thêm vào Streamlit Secrets:
        ```
        GEMINI_API_KEY = "AIzaSy..."
        ```
        """)
        st.stop()

def call_gemini_api(llm_config, prompt):
    """
    Gọi Gemini API bằng REST API
    """
    api_key = llm_config['api_key']
    model = llm_config['model']
    
    # API endpoint
    url = f"https://generativelanguage.googleapis.com/v1/{model}:generateContent?key={api_key}"
    
    # Request body
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2000,
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract text từ response
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        return parts[0]['text']
            
            return "Xin lỗi, không nhận được phản hồi từ AI."
            
        else:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            return f"Lỗi API: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "Lỗi: Timeout khi gọi API"
    except Exception as e:
        return f"Lỗi: {str(e)}"
def answer_from_external_api(prompt, llm_config, question_category):
    """Trả lời từ Gemini REST API"""
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

Trả lời thân thiện, chuyên nghiệp bằng tiếng Việt:
"""
    
    try:
        answer = call_gemini_api(llm_config, enhanced_prompt)
        
        # Thay thế placeholder còn sót
        replacements = {
            "[Số điện thoại": "1900 5555 14 hoặc 0879 5555 14",
            "[Email": "tuyensinh@hcmulaw.edu.vn",
            "[Website": "www.hcmulaw.edu.vn",
            "[Điện thoại": "(028) 39400 989"
        }
        
        for placeholder, actual in replacements.items():
            if placeholder in answer:
                answer = answer.replace(placeholder + "]", actual)
        
        return answer
        
    except Exception as e:
        return f"""
Xin lỗi, hệ thống gặp sự cố. Vui lòng liên hệ:

📞 **Hotline:** 1900 5555 14 hoặc 0879 5555 14
📧 **Email:** tuyensinh@hcmulaw.edu.vn
🌐 **Website:** www.hcmulaw.edu.vn
📍 **Địa chỉ:** 2 Nguyễn Tất Thành, P.12, Q.4, TP.HCM

_(Lỗi kỹ thuật: {str(e)[:100]})_
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
    
    for i, q in enumerate(questions):
        if st.button(q, key=f"q_{i}"):
            st.session_state["pending_question"] = q
            st.rerun()

def main():
    # Kiểm tra API key
    if not gemini_api_key:
        st.error("⚠️ **Thiếu GEMINI_API_KEY!**")
        st.info("""
        **Cách cấu hình:**
        1. Vào Settings → Secrets trên Streamlit Cloud
        2. Thêm:
        ```
        GEMINI_API_KEY = "your-api-key"
        ```
        3. Lấy API key tại: https://makersuite.google.com/app/apikey
        """)
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
        with st.expander("📊 Trạng thái hệ thống", expanded=False):
            st.success("✅ Gemini API: Đã kết nối")
            st.info(f"📁 Documents: {len(get_document_files())} files")
            
            if GDRIVE_VECTORSTORE_ID:
                st.info("☁️ Google Drive: Đã cấu hình")
            else:
                st.warning("⚠️ Google Drive: Chưa cấu hình")
        
        # Hướng dẫn cấu hình GDrive
        with st.expander("📖 Hướng dẫn Google Drive", expanded=False):
            st.markdown("""
            **Để sử dụng Google Drive:**
            
            1. Upload file `vectorstore.pkl` và `metadata.json` lên GDrive
            2. Click chuột phải → Share → Anyone with the link
            3. Copy File ID từ URL (phần sau `/d/` và trước `/view`)
            4. Thêm vào Secrets:
            ```
            GDRIVE_VECTORSTORE_ID = "file-id-1"
            GDRIVE_METADATA_ID = "file-id-2"
            ```
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

    # Khởi tạo vectorstore và LLM
    with st.spinner("🔄 Đang khởi động hệ thống..."):
        vectorstore, stats = initialize_vectorstore()
        llm = get_gemini_llm()
        
        # Không dùng chain nữa, xử lý trực tiếp
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            retriever = None

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
    if "pending_question" in st.session_state and st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None
    
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
                    # XỬ LÝ TRỰC TIẾP không dùng chain
                    if retriever:
                        # Lấy context từ vectorstore
                        docs = retriever.invoke(prompt)
                        context = "\n\n".join([doc.page_content for doc in docs[:3]])
                        
                        # Tạo prompt với context
                        full_prompt = f"""
Bạn là chuyên gia tư vấn của Đại học Luật TPHCM.

THÔNG TIN THAM KHẢO:
{context}

THÔNG TIN LIÊN HỆ:
- Hotline: 1900 5555 14 hoặc 0879 5555 14
- Email: tuyensinh@hcmulaw.edu.vn
- Website: www.hcmulaw.edu.vn

Câu hỏi: {prompt}

Hãy trả lời dựa trên thông tin tham khảo ở trên. Nếu không có thông tin, hãy tư vấn chung và khuyến khích liên hệ trực tiếp.
"""
                        answer = call_gemini_api(llm, full_prompt)

                    else:
                        # Không có vectorstore, dùng API thuần
                        answer = answer_from_external_api(prompt, llm, category)
                    
                    st.markdown(answer)
                    
                except Exception as e:
                    answer = f"""
❌ **Lỗi hệ thống**

Vui lòng liên hệ:
📞 Hotline: 1900 5555 14 hoặc 0879 5555 14
📧 Email: tuyensinh@hcmulaw.edu.vn

_(Lỗi: {str(e)[:100]})_
"""
                    st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "category": category
            })
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>🏛️ Trường Đại học Luật TP. Hồ Chí Minh</h4>
        <p>📍 2 Nguyễn Tất Thành, Phường 12, Quận 4, TP.HCM</p>
        <p>📞 Hotline: 1900 5555 14 | Email: tuyensinh@hcmulaw.edu.vn</p>
        <p>🌐 www.hcmulaw.edu.vn | 📘 facebook.com/hcmulaw</p>
        <p style="margin-top:1rem;opacity:0.8;font-size:0.85em;">
            🤖 Chatbot v2.1 | Phát triển bởi Lvphung - CNTT
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
